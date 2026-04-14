#!/usr/bin/env python3
"""Compatible but simplified ROS2 ACT deploy node.

What stays compatible:
- node name: me_act_inference_node_py
- services:
  - ~/start
  - ~/stop
  - ~/emergency_stop
  - ~/initialize
- run states:
  - IDLE / INITIALIZING / RUNNING / ESTOP / FAULT

What is intentionally simplified:
- latest RGB / latest Depth cached by callbacks
- fixed minimum control period loop
- async servo state request with timeout
- no RGB/Depth timestamp sync
- no frame-age / skew validation
- no pending-request bookkeeping / generation logic
- each successful loop:
    request servo state -> use latest images -> inference -> publish
- print actual closed-loop duration for each successful cycle
"""

from __future__ import annotations

import random
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
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


class RunState(str, Enum):
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    ESTOP = "ESTOP"
    FAULT = "FAULT"


class MeActInferenceNodePy(Node):
    def __init__(self) -> None:
        super().__init__("me_act_inference_node_py")
        self.bridge = CvBridge()
        self._rng = random.Random()

        self._declare_parameters()
        self._load_parameters()

        if not self.deploy_dir:
            raise RuntimeError("Parameter deploy_dir must not be empty.")
        if not (len(self.servo_ids) == len(self.init_center) == len(self.physical_min) == len(self.physical_max) == 6):
            raise RuntimeError(
                "servo_ids/init_center/physical_min/physical_max must all have length 6."
            )

        self.pipeline = ActPipelinePy(self.deploy_dir, self.device)

        self.rgb: Optional[np.ndarray] = None
        self.depth_raw: Optional[np.ndarray] = None

        self._state = RunState.RUNNING if self.enable_inference_on_start else RunState.IDLE
        self._initialize_until_ns = 0
        self._last_known_qpos = [float(v) for v in self.init_center]
        self._last_control_time = time.time()
        self._loop_idx = 0

        self.servo_command_pub = self.create_publisher(ServosPosition, self.servo_command_topic, 10)
        self.servo_state_client = self.create_client(GetBusServoState, self.servo_state_service)

        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self._on_rgb, 2)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self._on_depth, 2)

        self.start_srv = self.create_service(Trigger, "~/start", self._handle_start)
        self.stop_srv = self.create_service(Trigger, "~/stop", self._handle_stop)
        self.estop_srv = self.create_service(Trigger, "~/emergency_stop", self._handle_estop)
        self.initialize_srv = self.create_service(Trigger, "~/initialize", self._handle_initialize)

        while not self.servo_state_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().info(f"Waiting for servo state service: {self.servo_state_service}")

        self.get_logger().info("=" * 68)
        self.get_logger().info("Compatible simple ACT deploy node ready")
        self.get_logger().info(f"state={self._state.value}")
        self.get_logger().info(f"control_period_ms={self.control_period_ms}")
        self.get_logger().info(f"servo_query_timeout_ms={self.servo_query_timeout_ms}")
        self.get_logger().info(f"command_duration_ms={self.command_duration_ms}")
        self.get_logger().info(f"init_command_duration_ms={self.init_command_duration_ms}")
        self.get_logger().info("=" * 68)

    def _declare_parameters(self) -> None:
        self.declare_parameter("deploy_dir", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("rgb_topic", "/depth_cam/rgb/image_raw")
        self.declare_parameter("depth_topic", "/depth_cam/depth/image_raw")
        self.declare_parameter("servo_command_topic", "/ros_robot_controller/bus_servo/set_position")
        self.declare_parameter("servo_state_service", "/ros_robot_controller/bus_servo/get_state")
        self.declare_parameter("control_period_ms", 300)
        self.declare_parameter("servo_query_timeout_ms", 50)
        self.declare_parameter("command_duration_ms", 300)
        self.declare_parameter("init_command_duration_ms", 1500)
        self.declare_parameter("enable_inference_on_start", False)
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
        self.control_period_ms = int(self.get_parameter("control_period_ms").value)
        self.servo_query_timeout_ms = int(self.get_parameter("servo_query_timeout_ms").value)
        self.command_duration_ms = int(self.get_parameter("command_duration_ms").value)
        self.init_command_duration_ms = int(self.get_parameter("init_command_duration_ms").value)
        self.enable_inference_on_start = bool(self.get_parameter("enable_inference_on_start").value)
        self.servo_ids = [int(v) for v in self.get_parameter("servo_ids").value]
        self.init_center = [int(v) for v in self.get_parameter("init_center").value]
        self.init_random_range = int(self.get_parameter("init_random_range").value)
        self.physical_min = [int(v) for v in self.get_parameter("physical_min").value]
        self.physical_max = [int(v) for v in self.get_parameter("physical_max").value]

    def _on_rgb(self, msg: Image) -> None:
        try:
            self.rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().warning(f"RGB conversion failed: {exc}")

    def _on_depth(self, msg: Image) -> None:
        try:
            self.depth_raw = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as exc:
            self.get_logger().warning(f"Depth conversion failed: {exc}")

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

    def _publish_servo_command(self, action: list[float], duration_ms: int) -> None:
        msg = ServosPosition()
        msg.duration = float(duration_ms) / 1000.0
        msg.position = []

        for index, servo_id in enumerate(self.servo_ids):
            servo = ServoPosition()
            servo.id = int(servo_id)
            servo.position = int(self._clamp_to_physical_range(int(round(action[index])), index))
            msg.position.append(servo)

        self.servo_command_pub.publish(msg)

    def _read_servo_once(self) -> Optional[list[float]]:
        request = GetBusServoState.Request()
        request.cmd = []
        for servo_id in self.servo_ids:
            cmd = GetBusServoCmd()
            cmd.id = int(servo_id)
            cmd.get_position = 1
            request.cmd.append(cmd)

        future = self.servo_state_client.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=self.servo_query_timeout_ms / 1000.0,
        )

        response = future.result()
        if response is None or not response.state:
            return None

        qpos: list[float] = []
        usable = min(len(response.state), len(self.servo_ids))
        for index in range(usable):
            bus_state = response.state[index]
            if not bus_state.position:
                return None
            qpos.append(float(bus_state.position[-1]))

        if len(qpos) != len(self.servo_ids):
            return None

        self._last_known_qpos = list(qpos)
        return qpos

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
        response.success = True
        response.message = "Inference started."
        return response

    def _handle_stop(self, request, response):
        del request
        self._state = RunState.IDLE
        response.success = True
        response.message = "Inference stopped."
        return response

    def _handle_estop(self, request, response):
        del request
        self._state = RunState.ESTOP
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
            response.success = True
            response.message = "Initialization command sent."
            self.get_logger().info("Initialization pose sent: [%s]" % self._join_vector(pose))
        except Exception as exc:
            self._state = RunState.FAULT
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f"Initialization failed: {exc}")
        return response

    def run(self) -> None:
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)

            if self._state in (RunState.IDLE, RunState.ESTOP, RunState.FAULT):
                continue

            if self._state == RunState.INITIALIZING:
                if self._now_ns() >= self._initialize_until_ns:
                    self.pipeline.reset_memory()
                    self._state = RunState.RUNNING
                    self.get_logger().info("Initialization finished. Switching to RUNNING.")
                continue

            now = time.time()
            if now - self._last_control_time < self.control_period_ms / 1000.0:
                continue
            self._last_control_time = now

            if self.rgb is None or self.depth_raw is None:
                self.get_logger().warning("Skip: latest image not ready.")
                continue

            loop_start = time.time()

            qpos = self._read_servo_once()
            if qpos is None:
                self.get_logger().warning("Skip: servo query timeout or invalid response.")
                continue

            rgb = self.rgb
            depth = self.depth_raw

            try:
                trajectory = self.pipeline.predict(rgb, depth, qpos)
            except Exception as exc:
                self._state = RunState.FAULT
                self.get_logger().error(f"Inference failed. Entering FAULT: {exc}")
                continue

            if not trajectory or len(trajectory[0]) != len(self.servo_ids):
                self.get_logger().error("ACT returned empty trajectory or wrong action dimension. Entering FAULT.")
                self._state = RunState.FAULT
                continue

            self._publish_servo_command(trajectory[0], self.command_duration_ms)

            loop_ms = (time.time() - loop_start) * 1000.0
            self._loop_idx += 1
            self.get_logger().info(
                "loop=%d state=%s qpos=[%s] cmd0=[%s] closed_loop_ms=%.1f"
                % (
                    self._loop_idx,
                    self._state.value,
                    ", ".join(f"{x:.1f}" for x in qpos),
                    ", ".join(f"{x:.1f}" for x in trajectory[0]),
                    loop_ms,
                )
            )


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
