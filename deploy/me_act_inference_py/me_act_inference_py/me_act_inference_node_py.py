#!/usr/bin/env python3
"""Single-threaded discrete ACT deploy node.

Design goal:
- collect image/state observations asynchronously into queues;
- match them using the same logic as the current data sampling path;
- once one valid pair is found, stop collection, clear queues, run inference,
  send cmd0 with a short duration, then immediately start the next cycle.
"""

from __future__ import annotations

import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import message_filters
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from cv_bridge import CvBridge
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


def now_ns() -> int:
    return time.time_ns()


class RunState(str, Enum):
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    ESTOP = "ESTOP"
    FAULT = "FAULT"


@dataclass
class SyncedFrame:
    rgb_bgr: np.ndarray
    depth_raw: np.ndarray
    rgb_stamp_ns: int
    depth_stamp_ns: int
    synced_stamp_ns: int
    frame_id: int


@dataclass
class ServoStateSnapshot:
    qpos: list[float]
    observed: list[bool]
    state_est_stamp_ns: int
    request_started_ns: int
    response_received_ns: int
    missing_ids: list[int]


@dataclass
class MatchedSample:
    frame: SyncedFrame
    servo: ServoStateSnapshot


class MeActInferenceNodePy(Node):
    def __init__(self) -> None:
        super().__init__("me_act_inference_node_py")
        self.bridge = CvBridge()
        self._rng = random.Random()
        self._state = RunState.IDLE
        self._initialize_until_ns = 0
        self._frame_counter = 0
        self._cycle_counter = 0
        self._last_servo_poll_wall_ns = 0
        self._active_request_started_wall_ns = 0
        self._servo_request_in_flight = False
        self._active_request_id: Optional[int] = None
        self._next_request_id = 0
        self._request_timeout_ids: set[int] = set()
        self._collection_enabled = False
        self._logged_first_image = False

        self._declare_parameters()
        self._load_parameters()

        if not self.deploy_dir:
            raise RuntimeError("Parameter deploy_dir must not be empty.")
        if not (
            len(self.servo_ids)
            == len(self.init_center)
            == len(self.physical_min)
            == len(self.physical_max)
            == 6
        ):
            raise RuntimeError("servo_ids/init_center/physical_min/physical_max must all have length 6.")

        self.pipeline = ActPipelinePy(self.deploy_dir, self.device)

        self.frame_queue: deque[SyncedFrame] = deque(maxlen=self.frame_queue_size)
        self.servo_cache: deque[ServoStateSnapshot] = deque(maxlen=self.servo_cache_maxlen)
        self._last_known_qpos = [float(v) for v in self.init_center]

        self.servo_command_pub = self.create_publisher(ServosPosition, self.servo_command_topic, 10)
        self.servo_state_client = self.create_client(GetBusServoState, self.servo_state_service)

        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        self.image_sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=self.image_sync_queue_size,
            slop=self.image_sync_slop_s,
        )
        self.image_sync.registerCallback(self._on_synced_images)

        self.start_srv = self.create_service(Trigger, "~/start", self._handle_start)
        self.stop_srv = self.create_service(Trigger, "~/stop", self._handle_stop)
        self.estop_srv = self.create_service(Trigger, "~/emergency_stop", self._handle_estop)
        self.initialize_srv = self.create_service(Trigger, "~/initialize", self._handle_initialize)

        while not self.servo_state_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().info(f"Waiting for servo state service: {self.servo_state_service}")

        self._state = RunState.RUNNING if self.enable_inference_on_start else RunState.IDLE
        self._collection_enabled = self._state == RunState.RUNNING

        self.get_logger().info("=" * 72)
        self.get_logger().info("Discrete single-thread ACT deploy node ready")
        self.get_logger().info(f"state={self._state.value}")
        self.get_logger().info(f"servo_poll_hz={self.servo_poll_hz:.1f}")
        self.get_logger().info(f"image_sync_queue_size={self.image_sync_queue_size}")
        self.get_logger().info(f"image_sync_slop_s={self.image_sync_slop_s:.3f}")
        self.get_logger().info(f"max_rgb_depth_skew_ms={self.max_rgb_depth_skew_ms}")
        self.get_logger().info(f"max_img_state_skew_ms={self.max_img_state_skew_ms}")
        self.get_logger().info(f"command_duration_ms={self.command_duration_ms}")
        self.get_logger().info("=" * 72)

    def _declare_parameters(self) -> None:
        self.declare_parameter("deploy_dir", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("rgb_topic", "/depth_cam/rgb/image_raw")
        self.declare_parameter("depth_topic", "/depth_cam/depth/image_raw")
        self.declare_parameter("servo_command_topic", "/ros_robot_controller/bus_servo/set_position")
        self.declare_parameter("servo_state_service", "/ros_robot_controller/bus_servo/get_state")
        self.declare_parameter("command_duration_ms", 20)
        self.declare_parameter("init_command_duration_ms", 1500)
        self.declare_parameter("enable_inference_on_start", False)
        self.declare_parameter("enable_me_block", False)
        self.declare_parameter("validate_servo_ids", False)
        self.declare_parameter("debug_dump_dir", "")
        self.declare_parameter("debug_dump_every_n", 0)
        self.declare_parameter("servo_poll_hz", 30.0)
        self.declare_parameter("servo_request_timeout_ms", 200)
        self.declare_parameter("image_sync_queue_size", 20)
        self.declare_parameter("image_sync_slop_s", 0.03)
        self.declare_parameter("frame_queue_size", 20)
        self.declare_parameter("servo_cache_maxlen", 512)
        self.declare_parameter("max_img_state_skew_ms", 25)
        self.declare_parameter("max_rgb_depth_skew_ms", 25)
        self.declare_parameter("loop_sleep_ms", 1)
        self.declare_parameter("servo_ids", [1, 2, 3, 4, 5, 10])
        self.declare_parameter("init_center", [500, 500, 180, 190, 500, 300])
        self.declare_parameter("init_random_range", 40)
        self.declare_parameter("physical_min", [0, 100, 50, 50, 50, 150])
        self.declare_parameter("physical_max", [1000, 800, 650, 900, 950, 700])

    def _load_parameters(self) -> None:
        self.deploy_dir = str(self.get_parameter("deploy_dir").value)
        self.device = str(self.get_parameter("device").value)
        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.servo_command_topic = str(self.get_parameter("servo_command_topic").value)
        self.servo_state_service = str(self.get_parameter("servo_state_service").value)
        self.command_duration_ms = int(self.get_parameter("command_duration_ms").value)
        self.init_command_duration_ms = int(self.get_parameter("init_command_duration_ms").value)
        self.enable_inference_on_start = bool(self.get_parameter("enable_inference_on_start").value)
        self.enable_me_block = bool(self.get_parameter("enable_me_block").value)
        self.validate_servo_ids = bool(self.get_parameter("validate_servo_ids").value)
        self.debug_dump_dir = str(self.get_parameter("debug_dump_dir").value)
        self.debug_dump_every_n = int(self.get_parameter("debug_dump_every_n").value)
        self.servo_poll_hz = float(self.get_parameter("servo_poll_hz").value)
        self.servo_request_timeout_ms = int(self.get_parameter("servo_request_timeout_ms").value)
        self.image_sync_queue_size = int(self.get_parameter("image_sync_queue_size").value)
        self.image_sync_slop_s = float(self.get_parameter("image_sync_slop_s").value)
        self.frame_queue_size = int(self.get_parameter("frame_queue_size").value)
        self.servo_cache_maxlen = int(self.get_parameter("servo_cache_maxlen").value)
        self.max_img_state_skew_ms = int(self.get_parameter("max_img_state_skew_ms").value)
        self.max_rgb_depth_skew_ms = int(self.get_parameter("max_rgb_depth_skew_ms").value)
        self.loop_sleep_ms = int(self.get_parameter("loop_sleep_ms").value)
        self.servo_ids = [int(v) for v in self.get_parameter("servo_ids").value]
        self.init_center = [int(v) for v in self.get_parameter("init_center").value]
        self.init_random_range = int(self.get_parameter("init_random_range").value)
        self.physical_min = [int(v) for v in self.get_parameter("physical_min").value]
        self.physical_max = [int(v) for v in self.get_parameter("physical_max").value]

        self.servo_poll_period_ns = int(1e9 / max(self.servo_poll_hz, 1e-6))

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _join_vector(self, values) -> str:
        return ", ".join(str(v) for v in values)

    def _clamp_to_physical_range(self, value: int, index: int) -> int:
        return max(self.physical_min[index], min(value, self.physical_max[index]))

    def _sample_initialization_pose(self) -> list[float]:
        pose = []
        for index, center in enumerate(self.init_center):
            candidate = center + self._rng.randint(-self.init_random_range, self.init_random_range)
            pose.append(float(self._clamp_to_physical_range(candidate, index)))
        return pose

    def _set_collection_enabled(self, enabled: bool) -> None:
        self._collection_enabled = bool(enabled)
        if not enabled:
            self.frame_queue.clear()
            self.servo_cache.clear()

    def _on_synced_images(self, rgb_msg: Image, depth_msg: Image) -> None:
        if not self._collection_enabled or self._state != RunState.RUNNING:
            return
        try:
            rgb_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as exc:
            self.get_logger().warning(f"Image conversion failed: {exc}")
            return

        frame = SyncedFrame(
            rgb_bgr=np.array(rgb_bgr, copy=True),
            depth_raw=np.array(depth_raw, copy=True),
            rgb_stamp_ns=stamp_to_ns(rgb_msg.header.stamp),
            depth_stamp_ns=stamp_to_ns(depth_msg.header.stamp),
            synced_stamp_ns=max(stamp_to_ns(rgb_msg.header.stamp), stamp_to_ns(depth_msg.header.stamp)),
            frame_id=self._frame_counter + 1,
        )
        self._frame_counter = frame.frame_id
        self.frame_queue.append(frame)

        if not self._logged_first_image:
            self._logged_first_image = True
            depth_channels = 1 if frame.depth_raw.ndim == 2 else frame.depth_raw.shape[2]
            self.get_logger().info(
                "First synced pair: rgb=%dx%d depth=%dx%d depth_channels=%d dtype=%s"
                % (
                    frame.rgb_bgr.shape[1],
                    frame.rgb_bgr.shape[0],
                    frame.depth_raw.shape[1],
                    frame.depth_raw.shape[0],
                    depth_channels,
                    str(frame.depth_raw.dtype),
                )
            )

    def _build_servo_request(self) -> GetBusServoState.Request:
        request = GetBusServoState.Request()
        request.cmd = []
        for servo_id in self.servo_ids:
            cmd = GetBusServoCmd()
            cmd.id = int(servo_id)
            cmd.get_id = int(1 if self.validate_servo_ids else 0)
            cmd.get_position = 1
            request.cmd.append(cmd)
        return request

    def _maybe_send_servo_request(self) -> None:
        if not self._collection_enabled or self._state != RunState.RUNNING:
            return
        if self._servo_request_in_flight:
            return
        now_wall_ns = now_ns()
        if now_wall_ns - self._last_servo_poll_wall_ns < self.servo_poll_period_ns:
            return

        self._last_servo_poll_wall_ns = now_wall_ns
        self._servo_request_in_flight = True
        request_id = self._next_request_id
        self._next_request_id += 1
        self._active_request_id = request_id
        request_started_ns = now_wall_ns
        self._active_request_started_wall_ns = request_started_ns
        future = self.servo_state_client.call_async(self._build_servo_request())
        future.add_done_callback(
            lambda fut, req_id=request_id, req_started=request_started_ns: self._on_servo_response(
                fut, req_id, req_started
            )
        )

    def _handle_servo_request_timeout(self) -> None:
        if not self._servo_request_in_flight or self._active_request_id is None:
            return
        elapsed_ms = (now_ns() - self._active_request_started_wall_ns) / 1e6
        if elapsed_ms < self.servo_request_timeout_ms:
            return
        self._request_timeout_ids.add(self._active_request_id)
        self._active_request_id = None
        self._active_request_started_wall_ns = 0
        self._servo_request_in_flight = False
        self.get_logger().warning(
            f"Servo request timeout after {elapsed_ms:.1f} ms, dropping this request and continuing."
        )

    def _extract_servo_snapshot(
        self,
        response,
        request_started_ns: int,
        response_received_ns: int,
    ) -> Optional[ServoStateSnapshot]:
        if response is None or not response.state:
            return None

        qpos = list(self._last_known_qpos)
        observed = [False] * len(self.servo_ids)
        missing_ids: list[int] = []
        position_by_id: dict[int, float] = {}
        ordered_positions: list[tuple[int, float]] = []
        usable_state_count = min(len(response.state), len(self.servo_ids))

        for index in range(usable_state_count):
            bus_state = response.state[index]
            if not bus_state.position:
                continue
            position = float(bus_state.position[-1])
            ordered_positions.append((index, position))
            if self.validate_servo_ids and getattr(bus_state, "present_id", None):
                if bus_state.present_id:
                    present_id = int(bus_state.present_id[-1])
                    position_by_id[present_id] = position

        if self.validate_servo_ids and position_by_id:
            for index, servo_id in enumerate(self.servo_ids):
                if servo_id in position_by_id:
                    qpos[index] = position_by_id[servo_id]
                    observed[index] = True
                else:
                    missing_ids.append(servo_id)
        else:
            for index, position in ordered_positions:
                qpos[index] = position
                observed[index] = True
            for index in range(len(ordered_positions), len(self.servo_ids)):
                missing_ids.append(self.servo_ids[index])

        if not any(observed):
            return None

        self._last_known_qpos = list(qpos)
        request_response_ns = max(response_received_ns - request_started_ns, 0)
        state_est_stamp_ns = request_started_ns + request_response_ns // 2
        return ServoStateSnapshot(
            qpos=qpos,
            observed=observed,
            state_est_stamp_ns=state_est_stamp_ns,
            request_started_ns=request_started_ns,
            response_received_ns=response_received_ns,
            missing_ids=missing_ids,
        )

    def _on_servo_response(self, future, request_id: int, request_started_ns: int) -> None:
        response_received_ns = now_ns()
        if request_id in self._request_timeout_ids:
            self._request_timeout_ids.discard(request_id)
            return
        if request_id != self._active_request_id:
            return
        self._active_request_id = None
        self._active_request_started_wall_ns = 0
        self._servo_request_in_flight = False
        if not self._collection_enabled or self._state != RunState.RUNNING:
            return
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warning(f"Servo query failed: {exc}")
            return

        snapshot = self._extract_servo_snapshot(response, request_started_ns, response_received_ns)
        if snapshot is None:
            return
        self.servo_cache.append(snapshot)

    def _try_match_sample(self) -> Optional[MatchedSample]:
        if not self.frame_queue or not self.servo_cache:
            return None

        best_candidate = None
        for frame_index, frame in enumerate(self.frame_queue):
            rgb_depth_skew_ms = abs(frame.rgb_stamp_ns - frame.depth_stamp_ns) / 1e6
            if rgb_depth_skew_ms > self.max_rgb_depth_skew_ms:
                continue
            for servo_index, servo in enumerate(self.servo_cache):
                img_state_skew_ms = abs(servo.state_est_stamp_ns - frame.synced_stamp_ns) / 1e6
                if img_state_skew_ms > self.max_img_state_skew_ms:
                    continue
                if best_candidate is None or img_state_skew_ms < best_candidate[0]:
                    best_candidate = (img_state_skew_ms, frame_index, servo_index)

        if best_candidate is None:
            return None

        _, frame_index, servo_index = best_candidate
        frame = self.frame_queue[frame_index]
        servo = self.servo_cache[servo_index]
        return MatchedSample(frame=frame, servo=servo)

    def _run_one_discrete_cycle(self, sample: MatchedSample) -> None:
        self._set_collection_enabled(False)
        cycle_started_ns = now_ns()
        try:
            trajectory = self.pipeline.predict(
                sample.frame.rgb_bgr,
                sample.frame.depth_raw,
                sample.servo.qpos,
                use_me_block=self.enable_me_block,
            )
        except Exception as exc:
            self._state = RunState.FAULT
            self.get_logger().error(f"Inference failed. Entering FAULT: {exc}")
            return

        if not trajectory or len(trajectory[0]) != len(self.servo_ids):
            self._state = RunState.FAULT
            self.get_logger().error("ACT returned empty trajectory or wrong action dimension. Entering FAULT.")
            return

        self._publish_servo_command(trajectory[0], self.command_duration_ms, sample.servo.observed)
        self._cycle_counter += 1
        rgb_depth_skew_ms = abs(sample.frame.rgb_stamp_ns - sample.frame.depth_stamp_ns) / 1e6
        img_state_skew_ms = abs(sample.servo.state_est_stamp_ns - sample.frame.synced_stamp_ns) / 1e6
        cycle_ms = (now_ns() - cycle_started_ns) / 1e6
        self.get_logger().info(
            "cycle=%d frame=%d rgb_depth=%.1fms img_state=%.1fms servo_rr=%.1fms infer+publish=%.1fms qpos=[%s] cmd0=[%s]"
            % (
                self._cycle_counter,
                sample.frame.frame_id,
                rgb_depth_skew_ms,
                img_state_skew_ms,
                (sample.servo.response_received_ns - sample.servo.request_started_ns) / 1e6,
                cycle_ms,
                self._join_vector(f"{v:.1f}" for v in sample.servo.qpos),
                self._join_vector(f"{v:.1f}" for v in trajectory[0]),
            )
        )
        self._set_collection_enabled(True)

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
            self.get_logger().warning("No servo command published because no servo state was observed.")
            return
        self.servo_command_pub.publish(msg)

    def _handle_start(self, request, response):
        del request
        if self._state == RunState.FAULT:
            response.success = False
            response.message = "Node is in FAULT state. Reinitialize or restart after fixing the cause."
            return response
        if self._state == RunState.INITIALIZING:
            response.success = False
            response.message = "Node is initializing. Wait until initialization finishes."
            return response
        self._state = RunState.RUNNING
        self._set_collection_enabled(True)
        response.success = True
        response.message = "Inference started."
        return response

    def _handle_stop(self, request, response):
        del request
        self._state = RunState.IDLE
        self._set_collection_enabled(False)
        response.success = True
        response.message = "Inference stopped."
        return response

    def _handle_estop(self, request, response):
        del request
        self._state = RunState.ESTOP
        self._set_collection_enabled(False)
        response.success = True
        response.message = "Emergency stop activated. No more motion commands will be sent."
        self.get_logger().warning("Emergency stop activated.")
        return response

    def _handle_initialize(self, request, response):
        del request
        try:
            pose = self._sample_initialization_pose()
            self._publish_servo_command(pose, self.init_command_duration_ms)
            self._last_known_qpos = list(pose)
            self.pipeline.reset_memory()
            self._initialize_until_ns = self._now_ns() + (self.init_command_duration_ms + 300) * 1_000_000
            self._state = RunState.INITIALIZING
            self._set_collection_enabled(False)
            response.success = True
            response.message = "Initialization command sent."
            self.get_logger().info("Initialization pose sent: [%s]" % self._join_vector(pose))
        except Exception as exc:
            self._state = RunState.FAULT
            self._set_collection_enabled(False)
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f"Initialization failed: {exc}")
        return response

    def run(self) -> None:
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)

            if self._state in (RunState.IDLE, RunState.ESTOP, RunState.FAULT):
                if self.loop_sleep_ms > 0:
                    time.sleep(self.loop_sleep_ms / 1000.0)
                continue

            if self._state == RunState.INITIALIZING:
                if self._now_ns() >= self._initialize_until_ns:
                    self.pipeline.reset_memory()
                    self._state = RunState.RUNNING
                    self._set_collection_enabled(True)
                    self.get_logger().info("Initialization finished. Switching to RUNNING.")
                if self.loop_sleep_ms > 0:
                    time.sleep(self.loop_sleep_ms / 1000.0)
                continue

            self._handle_servo_request_timeout()
            self._maybe_send_servo_request()
            sample = self._try_match_sample()
            if sample is not None:
                self._run_one_discrete_cycle(sample)

            if self.loop_sleep_ms > 0:
                time.sleep(self.loop_sleep_ms / 1000.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MeActInferenceNodePy()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
