#!/usr/bin/env python3
"""Pure Python ROS2 deploy node matching the C++ me_act_inference_node."""

from __future__ import annotations

import os
import random
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from ros_robot_controller_msgs.msg import GetBusServoCmd, ServoPosition, ServosPosition
from ros_robot_controller_msgs.srv import GetBusServoState

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from act_pipeline_py import ActPipelinePy


def stamp_to_ns(stamp: TimeMsg) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class RunState(str, Enum):
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    ESTOP = "ESTOP"
    FAULT = "FAULT"


@dataclass
class RawImageFrame:
    image: np.ndarray
    encoding: str
    stamp_ns: int


@dataclass
class SyncedFrame:
    rgb_bgr: np.ndarray
    depth_raw: np.ndarray
    rgb_encoding: str
    depth_encoding: str
    rgb_stamp_ns: int
    depth_stamp_ns: int
    synced_stamp_ns: int
    frame_id: int = 0


@dataclass
class PendingInference:
    frame: SyncedFrame
    state_query_started_ns: int
    tick: int
    generation: int


@dataclass
class ServoStateSnapshot:
    qpos: list[float]
    observed: list[bool]
    missing_ids: list[int] = field(default_factory=list)


class MeActInferenceNodePy(Node):
    def __init__(self) -> None:
        super().__init__("me_act_inference_node_py")
        self.bridge = CvBridge()
        self._rng = random.Random()
        self._frame_counter = 0
        self._tick_id = 0
        self._control_generation = 0
        self._logged_image_info = False
        self._state = RunState.IDLE
        self._initialize_until_ns = 0
        self._latest_frame: Optional[SyncedFrame] = None
        self._pending_state_request: Optional[PendingInference] = None
        self._inference_in_progress = False
        self._last_known_qpos: list[float] = []
        self._last_log_ns: dict[str, int] = {}
        self._frame_lock = threading.Lock()
        self._schedule_lock = threading.Lock()
        self._pipeline_lock = threading.Lock()
        self._pending_state_lock = threading.Lock()
        self._qpos_lock = threading.Lock()
        self._sync_lock = threading.Lock()
        self._rgb_queue: deque[RawImageFrame] = deque()
        self._depth_queue: deque[RawImageFrame] = deque()

        self._declare_parameters()
        self._load_parameters()

        self.pipeline = ActPipelinePy(self.deploy_dir, self.device)

        self.control_callback_group = ReentrantCallbackGroup()
        self.servo_client_callback_group = ReentrantCallbackGroup()

        self.servo_command_pub = self.create_publisher(ServosPosition, self.servo_command_topic, 10)
        self.servo_state_client = self.create_client(
            GetBusServoState,
            self.servo_state_service,
            callback_group=self.servo_client_callback_group,
        )

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self._on_rgb_image,
            10,
            callback_group=self.control_callback_group,
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self._on_depth_image,
            10,
            callback_group=self.control_callback_group,
        )

        self.start_srv = self.create_service(Trigger, "~/start", self._handle_start)
        self.stop_srv = self.create_service(Trigger, "~/stop", self._handle_stop)
        self.estop_srv = self.create_service(Trigger, "~/emergency_stop", self._handle_estop)
        self.initialize_srv = self.create_service(Trigger, "~/initialize", self._handle_initialize)

        self.control_timer = self.create_timer(
            self.control_period_ms / 1000.0,
            self._on_control_timer,
            callback_group=self.control_callback_group,
        )

        self._state = RunState.RUNNING if self.enable_inference_on_start else RunState.IDLE
        if self.enable_me_block and not self.pipeline.uses_memory_image_input():
            self.get_logger().warning(
                "enable_me_block=true, but exported ACT is a single-image model. me_block will be ignored."
            )
        if self.enable_me_block and self.pipeline.uses_memory_image_input() and not self.pipeline.has_me_block():
            raise RuntimeError("enable_me_block=true, but deploy_dir does not contain me_block_inference.pt.")

        self.get_logger().info(
            "me_act_inference_node_py ready. state=%s deploy_dir=%s enable_me_block=%s"
            % (
                self._state.value,
                self.deploy_dir,
                "true" if self.enable_me_block else "false",
            )
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("deploy_dir", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("rgb_topic", "/depth_cam/rgb/image_raw")
        self.declare_parameter("depth_topic", "/depth_cam/depth/image_raw")
        self.declare_parameter("servo_command_topic", "/ros_robot_controller/bus_servo/set_position")
        self.declare_parameter("servo_state_service", "/ros_robot_controller/bus_servo/get_state")
        self.declare_parameter("control_period_ms", 100)
        self.declare_parameter("command_duration_ms", 300)
        self.declare_parameter("init_command_duration_ms", 1500)
        self.declare_parameter("max_frame_age_ms", 250)
        self.declare_parameter("max_state_image_skew_ms", 150)
        self.declare_parameter("servo_state_timeout_ms", 5000)
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("enable_inference_on_start", False)
        self.declare_parameter("enable_me_block", False)
        self.declare_parameter("validate_servo_ids", False)
        self.declare_parameter("debug_dump_dir", "")
        self.declare_parameter("debug_dump_every_n", 0)
        self.declare_parameter("servo_ids", [1, 2, 3, 4, 5, 10])
        self.declare_parameter("init_center", [500, 500, 180, 190, 500, 300])
        self.declare_parameter("init_random_range", 40)
        self.declare_parameter("physical_min", [0, 100, 50, 50, 50, 150])
        self.declare_parameter("physical_max", [1000, 800, 650, 900, 950, 700])

    def _load_parameters(self) -> None:
        self.deploy_dir = self.get_parameter("deploy_dir").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.servo_command_topic = self.get_parameter("servo_command_topic").get_parameter_value().string_value
        self.servo_state_service = self.get_parameter("servo_state_service").get_parameter_value().string_value
        self.control_period_ms = int(self.get_parameter("control_period_ms").value)
        self.command_duration_ms = int(self.get_parameter("command_duration_ms").value)
        self.init_command_duration_ms = int(self.get_parameter("init_command_duration_ms").value)
        self.max_frame_age_ms = int(self.get_parameter("max_frame_age_ms").value)
        self.max_state_image_skew_ms = int(self.get_parameter("max_state_image_skew_ms").value)
        self.servo_state_timeout_ms = int(self.get_parameter("servo_state_timeout_ms").value)
        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self.enable_inference_on_start = bool(self.get_parameter("enable_inference_on_start").value)
        self.enable_me_block = bool(self.get_parameter("enable_me_block").value)
        self.validate_servo_ids = bool(self.get_parameter("validate_servo_ids").value)
        self.debug_dump_dir = self.get_parameter("debug_dump_dir").get_parameter_value().string_value
        self.debug_dump_every_n = int(self.get_parameter("debug_dump_every_n").value)
        self.servo_ids = [int(v) for v in self.get_parameter("servo_ids").value]
        self.init_center = [int(v) for v in self.get_parameter("init_center").value]
        self.init_random_range = int(self.get_parameter("init_random_range").value)
        self.physical_min = [int(v) for v in self.get_parameter("physical_min").value]
        self.physical_max = [int(v) for v in self.get_parameter("physical_max").value]

        if not self.deploy_dir:
            raise RuntimeError("Parameter deploy_dir must not be empty.")
        if not (len(self.servo_ids) == len(self.init_center) == len(self.physical_min) == len(self.physical_max) == 6):
            raise RuntimeError("servo_ids/init_center/physical_min/physical_max must all have length 6.")
        if self.debug_dump_every_n < 0:
            raise RuntimeError("debug_dump_every_n must be >= 0.")
        self._reset_last_known_qpos([float(v) for v in self.init_center])

    def _join_vector(self, values) -> str:
        return ", ".join(str(v) for v in values)

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _should_log_throttle(self, key: str, period_sec: float) -> bool:
        now_ns = self._now_ns()
        period_ns = int(period_sec * 1e9)
        last_ns = self._last_log_ns.get(key, 0)
        if now_ns - last_ns >= period_ns:
            self._last_log_ns[key] = now_ns
            return True
        return False

    def _on_rgb_image(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = RawImageFrame(
                image=image.copy(),
                encoding=msg.encoding,
                stamp_ns=stamp_to_ns(msg.header.stamp),
            )
            self._push_and_try_sync("rgb", frame)
        except Exception as exc:  # pragma: no cover - ROS callback safety
            if self._should_log_throttle("rgb_convert", 2.0):
                self.get_logger().error(f"RGB conversion failed: {exc}")

    def _on_depth_image(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            frame = RawImageFrame(
                image=np.array(image, copy=True),
                encoding=msg.encoding,
                stamp_ns=stamp_to_ns(msg.header.stamp),
            )
            self._push_and_try_sync("depth", frame)
        except Exception as exc:  # pragma: no cover - ROS callback safety
            if self._should_log_throttle("depth_convert", 2.0):
                self.get_logger().error(f"Depth conversion failed: {exc}")

    def _push_and_try_sync(self, source: str, frame: RawImageFrame) -> None:
        with self._sync_lock:
            if source == "rgb":
                self._rgb_queue.append(frame)
                while len(self._rgb_queue) > self.sync_queue_size:
                    self._rgb_queue.popleft()
                if not self._depth_queue:
                    return
                best_index = min(range(len(self._depth_queue)), key=lambda idx: abs(self._depth_queue[idx].stamp_ns - frame.stamp_ns))
                other = self._depth_queue[best_index]
                del self._depth_queue[best_index]
                self._rgb_queue.pop()
                rgb = frame
                depth = other
            else:
                self._depth_queue.append(frame)
                while len(self._depth_queue) > self.sync_queue_size:
                    self._depth_queue.popleft()
                if not self._rgb_queue:
                    return
                best_index = min(range(len(self._rgb_queue)), key=lambda idx: abs(self._rgb_queue[idx].stamp_ns - frame.stamp_ns))
                other = self._rgb_queue[best_index]
                del self._rgb_queue[best_index]
                self._depth_queue.pop()
                rgb = other
                depth = frame

        synced = SyncedFrame(
            rgb_bgr=rgb.image,
            depth_raw=depth.image,
            rgb_encoding=rgb.encoding,
            depth_encoding=depth.encoding,
            rgb_stamp_ns=rgb.stamp_ns,
            depth_stamp_ns=depth.stamp_ns,
            synced_stamp_ns=max(rgb.stamp_ns, depth.stamp_ns),
            frame_id=self._frame_counter + 1,
        )
        self._frame_counter = synced.frame_id

        if not self._logged_image_info:
            self._logged_image_info = True
            depth_channels = 1 if synced.depth_raw.ndim == 2 else synced.depth_raw.shape[2]
            self.get_logger().info(
                "First image pair: rgb encoding=%s shape=%dx%d channels=%d | depth encoding=%s shape=%dx%d channels=%d dtype=%s"
                % (
                    synced.rgb_encoding,
                    synced.rgb_bgr.shape[1],
                    synced.rgb_bgr.shape[0],
                    synced.rgb_bgr.shape[2],
                    synced.depth_encoding,
                    synced.depth_raw.shape[1],
                    synced.depth_raw.shape[0],
                    depth_channels,
                    str(synced.depth_raw.dtype),
                )
            )

        with self._frame_lock:
            self._latest_frame = synced

    def _on_control_timer(self) -> None:
        state = self._state
        if state in (RunState.ESTOP, RunState.IDLE, RunState.FAULT):
            return

        now_ns = self._now_ns()
        if state == RunState.INITIALIZING:
            if now_ns < self._get_initialize_until_ns():
                return
            with self._pipeline_lock:
                self.pipeline.reset_memory()
            self._tick_id = 0
            self._state = RunState.RUNNING
            self.get_logger().info("Initialization finished. Switching to RUNNING.")
            return

        if self._has_timed_out_pending_state_request(now_ns):
            self._enter_fault("Servo states request timeout")
            return
        if self._has_active_control_work():
            return

        frame = self._get_latest_frame()
        if frame is None:
            if self._should_log_throttle("no_frame", 2.0):
                self.get_logger().warning("No synced RGB/depth frame available yet.")
            return

        if (now_ns - frame.synced_stamp_ns) > self.max_frame_age_ms * 1_000_000:
            if self._should_log_throttle("old_frame", 2.0):
                self.get_logger().warning("Latest frame is too old. Skip this tick.")
            return

        if self._state != RunState.RUNNING:
            return

        generation = self._control_generation
        self._tick_id += 1
        tick = self._tick_id
        state_query_started_ns = self._now_ns()
        self._send_servo_state_request(frame, tick, generation, state_query_started_ns)

    def _run_inference_with_state(
        self,
        pending: PendingInference,
        servo_state: ServoStateSnapshot,
        state_received_ns: int,
    ) -> None:
        if self._state != RunState.RUNNING or self._control_generation != pending.generation:
            return

        skew_ms = (state_received_ns - pending.frame.synced_stamp_ns) / 1e6
        if abs(skew_ms) > self.max_state_image_skew_ms:
            self.get_logger().warning(
                f"Tick {pending.tick} skipped due to image/state skew: {skew_ms:.1f} ms"
            )
            return

        try:
            infer_started_ns = self._now_ns()
            with self._pipeline_lock:
                trajectory = self.pipeline.predict(
                    pending.frame.rgb_bgr,
                    pending.frame.depth_raw,
                    servo_state.qpos,
                    use_me_block=self.enable_me_block,
                )
            infer_finished_ns = self._now_ns()
            if self._state != RunState.RUNNING or self._control_generation != pending.generation:
                return
            if not trajectory or len(trajectory[0]) != len(self.servo_ids):
                self._enter_fault("ACT returned an empty trajectory or wrong action dimension.")
                return

            publish_frame_age_ms = (infer_finished_ns - pending.frame.synced_stamp_ns) / 1e6
            if publish_frame_age_ms > self.max_frame_age_ms:
                self.get_logger().warning(
                    f"Tick {pending.tick} skipped because inference output is stale: "
                    f"frame_age={publish_frame_age_ms:.1f} ms, limit={self.max_frame_age_ms} ms"
                )
                return

            self._maybe_dump_debug_tick(pending, servo_state, trajectory[0], infer_finished_ns)
            self._publish_servo_command(trajectory[0], self.command_duration_ms, servo_state.observed)

            self.get_logger().info(
                "tick=%d frame=%d frame_age=%.1fms state_wait=%.1fms infer=%.1fms qpos=[%s] observed=[%s] cmd0=[%s]"
                % (
                    pending.tick,
                    pending.frame.frame_id,
                    publish_frame_age_ms,
                    (state_received_ns - pending.state_query_started_ns) / 1e6,
                    (infer_finished_ns - infer_started_ns) / 1e6,
                    self._join_vector(servo_state.qpos),
                    self._join_vector(servo_state.observed),
                    self._join_vector(trajectory[0]),
                )
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            self._enter_fault(str(exc))

    def _get_latest_frame(self) -> Optional[SyncedFrame]:
        with self._frame_lock:
            return self._latest_frame

    def _get_initialize_until_ns(self) -> int:
        with self._schedule_lock:
            return self._initialize_until_ns

    def _set_initialize_until_ns(self, value: int) -> None:
        with self._schedule_lock:
            self._initialize_until_ns = value

    def _has_active_control_work(self) -> bool:
        with self._pending_state_lock:
            return self._pending_state_request is not None or self._inference_in_progress

    def _has_timed_out_pending_state_request(self, now_ns: int) -> bool:
        with self._pending_state_lock:
            if self._pending_state_request is None:
                return False
            elapsed_ns = now_ns - self._pending_state_request.state_query_started_ns
            if elapsed_ns <= self.servo_state_timeout_ms * 1_000_000:
                return False
            self._pending_state_request = None
            return True

    def _clear_pending_state_request(self) -> None:
        with self._pending_state_lock:
            self._pending_state_request = None
            self._inference_in_progress = False

    def _invalidate_control_work(self) -> None:
        self._control_generation += 1
        self._clear_pending_state_request()

    def _begin_inference_for_response(self, tick: int, generation: int) -> bool:
        with self._pending_state_lock:
            if self._pending_state_request is None:
                return False
            if self._pending_state_request.tick != tick or self._pending_state_request.generation != generation:
                return False
            self._pending_state_request = None
            self._inference_in_progress = True
            return True

    def _finish_inference_for_response(self) -> None:
        with self._pending_state_lock:
            self._inference_in_progress = False

    def _send_servo_state_request(
        self,
        frame: SyncedFrame,
        tick: int,
        generation: int,
        state_query_started_ns: int,
    ) -> None:
        if not self.servo_state_client.service_is_ready():
            if self._should_log_throttle("service_unready", 2.0):
                self.get_logger().warning(f"Servo state service not available: {self.servo_state_service}")
            return

        request = GetBusServoState.Request()
        request.cmd = []
        for servo_id in self.servo_ids:
            cmd = GetBusServoCmd()
            cmd.id = int(servo_id)
            cmd.get_id = int(1 if self.validate_servo_ids else 0)
            cmd.get_position = int(1)
            request.cmd.append(cmd)

        if self._state != RunState.RUNNING or self._control_generation != generation:
            return

        request_context = PendingInference(
            frame=frame,
            state_query_started_ns=state_query_started_ns,
            tick=tick,
            generation=generation,
        )
        with self._pending_state_lock:
            self._pending_state_request = request_context

        future = self.servo_state_client.call_async(request)
        future.add_done_callback(lambda fut, ctx=request_context: self._on_servo_state_response(ctx, fut))

    def _on_servo_state_response(self, request_context: PendingInference, future) -> None:
        if not self._begin_inference_for_response(request_context.tick, request_context.generation):
            return

        state_received_ns = self._now_ns()
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - runtime safety
            self._enter_fault(f"GetBusServoState future failed: {exc}")
            return

        if response is None or not response.state:
            self.get_logger().warning("GetBusServoState returned no servo state. Skip this tick.")
            self._finish_inference_for_response()
            return
        if not response.success:
            self.get_logger().warning(
                "GetBusServoState returned success=false; trying to use any partial state in the response."
            )

        servo_state = self._extract_servo_positions(response)
        if servo_state is None:
            self.get_logger().warning("Failed to parse any usable servo position. Skip this tick.")
            self._finish_inference_for_response()
            return

        self._run_inference_with_state(request_context, servo_state, state_received_ns)
        self._finish_inference_for_response()

    def _extract_servo_positions(self, response) -> Optional[ServoStateSnapshot]:
        if not response.state:
            return None
        if len(response.state) < len(self.servo_ids):
            self.get_logger().warning(
                f"Expected {len(self.servo_ids)} servo states, but got {len(response.state)}. "
                "Missing servos will keep their previous command."
            )
        elif len(response.state) > len(self.servo_ids):
            self.get_logger().warning(
                f"Expected {len(self.servo_ids)} servo states, but got {len(response.state)}. Extra states will be ignored."
            )

        with self._qpos_lock:
            qpos = list(self._last_known_qpos)
        if len(qpos) != len(self.servo_ids):
            qpos = [float(v) for v in self.init_center]

        snapshot = ServoStateSnapshot(qpos=qpos, observed=[False] * len(self.servo_ids))
        position_by_id: dict[int, float] = {}
        ordered_positions: list[tuple[int, float]] = []
        usable_state_count = min(len(response.state), len(self.servo_ids))

        for index in range(usable_state_count):
            bus_state = response.state[index]
            if not bus_state.position:
                self.get_logger().warning(
                    f"Servo state {index} has no position field. That servo will not be commanded this tick."
                )
                continue

            position = float(bus_state.position[-1])
            ordered_positions.append((index, position))

            if self.validate_servo_ids:
                if not bus_state.present_id:
                    self.get_logger().warning(
                        f"validate_servo_ids=true, but servo state {index} has no present_id field. "
                        "It will only be used if no IDs are returned."
                    )
                else:
                    present_id = int(bus_state.present_id[-1])
                    position_by_id[present_id] = position

        observed_count = 0
        if self.validate_servo_ids and position_by_id:
            for index, servo_id in enumerate(self.servo_ids):
                if servo_id not in position_by_id:
                    snapshot.missing_ids.append(servo_id)
                    continue
                snapshot.qpos[index] = position_by_id[servo_id]
                snapshot.observed[index] = True
                observed_count += 1
        else:
            for index, position in ordered_positions:
                snapshot.qpos[index] = position
                snapshot.observed[index] = True
                observed_count += 1
            for index in range(usable_state_count, len(self.servo_ids)):
                snapshot.missing_ids.append(self.servo_ids[index])

        for index in range(usable_state_count):
            if not snapshot.observed[index] and self.servo_ids[index] not in snapshot.missing_ids:
                snapshot.missing_ids.append(self.servo_ids[index])

        if observed_count == 0:
            self.get_logger().warning("GetBusServoState response contained no usable servo positions.")
            return None

        with self._qpos_lock:
            if len(self._last_known_qpos) != len(self.servo_ids):
                self._last_known_qpos = list(snapshot.qpos)
            for index, observed in enumerate(snapshot.observed):
                if observed:
                    self._last_known_qpos[index] = snapshot.qpos[index]

        if snapshot.missing_ids:
            self.get_logger().warning(
                "Partial servo state: observed=%d/%d missing_ids=[%s]. Missing servos will not receive a command this tick."
                % (observed_count, len(self.servo_ids), self._join_vector(snapshot.missing_ids))
            )

        return snapshot

    def _publish_servo_command(
        self,
        action: list[float],
        duration_ms: int,
        command_mask: Optional[list[bool]] = None,
    ) -> None:
        msg = ServosPosition()
        msg.duration = float(duration_ms) / 1000.0
        msg.position = []

        for index, servo_id in enumerate(self.servo_ids):
            if command_mask is not None and (index >= len(command_mask) or not command_mask[index]):
                continue
            servo = ServoPosition()
            servo.id = int(servo_id)
            servo.position = int(self._clamp_to_physical_range(int(round(action[index])), index))
            msg.position.append(servo)

        if not msg.position:
            self.get_logger().warning("No servo command published because no servo state was observed this tick.")
            return

        self.servo_command_pub.publish(msg)

    def _reset_last_known_qpos(self, qpos: list[float]) -> None:
        with self._qpos_lock:
            self._last_known_qpos = list(qpos)

    def _should_dump_debug_tick(self, tick: int) -> bool:
        return bool(self.debug_dump_dir) and self.debug_dump_every_n > 0 and tick % self.debug_dump_every_n == 0

    def _maybe_dump_debug_tick(
        self,
        pending: PendingInference,
        servo_state: ServoStateSnapshot,
        command: list[float],
        infer_finished_ns: int,
    ) -> None:
        if not self._should_dump_debug_tick(pending.tick):
            return

        try:
            os.makedirs(self.debug_dump_dir, exist_ok=True)
            stem = f"tick_{pending.tick}_frame_{pending.frame.frame_id}"
            base = Path(self.debug_dump_dir) / stem

            cv2.imwrite(str(base) + "_rgb_bgr.png", pending.frame.rgb_bgr)
            cv2.imwrite(str(base) + "_depth_raw.png", pending.frame.depth_raw)
            four_channel = self.pipeline.build_debug_four_channel_image(pending.frame.rgb_bgr, pending.frame.depth_raw)
            cv2.imwrite(str(base) + "_four_channel_bgra.png", four_channel)

            depth_min = float(np.min(pending.frame.depth_raw))
            depth_max = float(np.max(pending.frame.depth_raw))
            with open(str(base) + "_meta.txt", "w", encoding="utf-8") as meta:
                meta.write(f"tick={pending.tick}\n")
                meta.write(f"frame={pending.frame.frame_id}\n")
                meta.write(f"frame_age_ms={(infer_finished_ns - pending.frame.synced_stamp_ns) / 1e6}\n")
                meta.write(f"rgb_encoding={pending.frame.rgb_encoding}\n")
                meta.write(
                    f"rgb_shape={pending.frame.rgb_bgr.shape[1]}x{pending.frame.rgb_bgr.shape[0]}x{pending.frame.rgb_bgr.shape[2]}\n"
                )
                depth_channels = 1 if pending.frame.depth_raw.ndim == 2 else pending.frame.depth_raw.shape[2]
                meta.write(f"depth_encoding={pending.frame.depth_encoding}\n")
                meta.write(
                    f"depth_shape={pending.frame.depth_raw.shape[1]}x{pending.frame.depth_raw.shape[0]}x{depth_channels}\n"
                )
                meta.write(f"depth_type={pending.frame.depth_raw.dtype}\n")
                meta.write(f"depth_min={depth_min}\n")
                meta.write(f"depth_max={depth_max}\n")
                meta.write(f"qpos=[{self._join_vector(servo_state.qpos)}]\n")
                meta.write(f"observed=[{self._join_vector(servo_state.observed)}]\n")
                meta.write(f"cmd0=[{self._join_vector(command)}]\n")

            self.get_logger().info(f"Debug dump written: {base}")
        except Exception as exc:  # pragma: no cover - runtime safety
            self.get_logger().warning(f"Failed to write debug dump: {exc}")

    def _clamp_to_physical_range(self, value: int, index: int) -> int:
        return max(self.physical_min[index], min(value, self.physical_max[index]))

    def _sample_initialization_pose(self) -> list[float]:
        pose = []
        for index, center in enumerate(self.init_center):
            candidate = center + self._rng.randint(-self.init_random_range, self.init_random_range)
            pose.append(float(self._clamp_to_physical_range(candidate, index)))
        return pose

    def _handle_start(self, request, response):
        del request
        if self._state == RunState.FAULT:
            response.success = False
            response.message = "Node is in FAULT state. Restart node or reinitialize after fixing the cause."
            return response
        if self._state == RunState.INITIALIZING:
            response.success = False
            response.message = "Node is initializing. Wait until initialization finishes."
            return response
        self._state = RunState.RUNNING
        response.success = True
        response.message = "Inference started."
        return response

    def _handle_stop(self, request, response):
        del request
        self._state = RunState.IDLE
        self._invalidate_control_work()
        with self._pipeline_lock:
            self.pipeline.reset_memory()
        response.success = True
        response.message = "Inference stopped."
        return response

    def _handle_estop(self, request, response):
        del request
        self._state = RunState.ESTOP
        self._invalidate_control_work()
        with self._pipeline_lock:
            self.pipeline.reset_memory()
        response.success = True
        response.message = "Emergency stop activated. No more motion commands will be sent."
        self.get_logger().warning("Emergency stop activated.")
        return response

    def _handle_initialize(self, request, response):
        del request
        try:
            self._invalidate_control_work()
            pose = self._sample_initialization_pose()
            self._publish_servo_command(pose, self.init_command_duration_ms)
            self._reset_last_known_qpos(pose)
            with self._pipeline_lock:
                self.pipeline.reset_memory()
            self._set_initialize_until_ns(self._now_ns() + (self.init_command_duration_ms + 300) * 1_000_000)
            self._state = RunState.INITIALIZING
            response.success = True
            response.message = "Initialization command sent."
            self.get_logger().info("Initialization pose sent: [%s]" % self._join_vector(pose))
        except Exception as exc:  # pragma: no cover - runtime safety
            response.success = False
            response.message = str(exc)
            self._enter_fault(str(exc))
        return response

    def _enter_fault(self, reason: str) -> None:
        self._state = RunState.FAULT
        self._invalidate_control_work()
        with self._pipeline_lock:
            self.pipeline.reset_memory()
        self.get_logger().error(f"Entering FAULT state: {reason}")


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = MeActInferenceNodePy()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
