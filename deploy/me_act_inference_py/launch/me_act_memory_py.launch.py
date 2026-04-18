from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "deploy_dir",
                default_value="/home/ubuntu/my_models/me_act/deploy_artifacts_memory",
                description="Directory containing act_inference.pt, deploy_config.yml, and optional me_block_inference.pt.",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cuda",
                description="Torch device used by ACT and online me_block. Use cpu only for debugging.",
            ),
            DeclareLaunchArgument(
                "rgb_topic",
                default_value="/depth_cam/rgb/image_raw",
                description="RGB image topic.",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/depth_cam/depth/image_raw",
                description="Depth image topic.",
            ),
            DeclareLaunchArgument(
                "servo_command_topic",
                default_value="/ros_robot_controller/bus_servo/set_position",
                description="Servo command topic.",
            ),
            DeclareLaunchArgument(
                "servo_state_service",
                default_value="/ros_robot_controller/bus_servo/get_state",
                description="Servo state service.",
            ),
            DeclareLaunchArgument(
                "command_duration_ms",
                default_value="20",
                description="Duration for published cmd0 servo command in milliseconds.",
            ),
            DeclareLaunchArgument(
                "init_command_duration_ms",
                default_value="1500",
                description="Initialization command duration in milliseconds.",
            ),
            DeclareLaunchArgument(
                "servo_poll_hz",
                default_value="30.0",
                description="Asynchronous servo polling frequency.",
            ),
            DeclareLaunchArgument(
                "servo_request_timeout_ms",
                default_value="200",
                description="Timeout for one servo state request.",
            ),
            DeclareLaunchArgument(
                "image_sync_queue_size",
                default_value="20",
                description="ApproximateTimeSynchronizer queue size.",
            ),
            DeclareLaunchArgument(
                "image_sync_slop_s",
                default_value="0.03",
                description="RGB/depth sync slop in seconds.",
            ),
            DeclareLaunchArgument(
                "frame_queue_size",
                default_value="20",
                description="Matched frame candidate queue size.",
            ),
            DeclareLaunchArgument(
                "servo_cache_maxlen",
                default_value="512",
                description="Servo state cache size for matching.",
            ),
            DeclareLaunchArgument(
                "max_img_state_skew_ms",
                default_value="25",
                description="Maximum allowed image/servo physical skew in milliseconds.",
            ),
            DeclareLaunchArgument(
                "max_rgb_depth_skew_ms",
                default_value="25",
                description="Maximum allowed RGB/depth physical skew in milliseconds.",
            ),
            DeclareLaunchArgument(
                "loop_sleep_ms",
                default_value="1",
                description="Sleep inserted at the end of each outer loop iteration.",
            ),
            DeclareLaunchArgument(
                "enable_inference_on_start",
                default_value="false",
                description="Start in RUNNING state immediately after launch.",
            ),
            DeclareLaunchArgument(
                "validate_servo_ids",
                default_value="false",
                description="Read servo IDs and reorder qpos by servo_ids for debugging.",
            ),
            DeclareLaunchArgument(
                "debug_dump_dir",
                default_value="",
                description="Directory for online RGB/depth/BGRA debug dumps. Empty disables dumping.",
            ),
            DeclareLaunchArgument(
                "debug_dump_every_n",
                default_value="0",
                description="Dump every N successful inference ticks. 0 disables dumping.",
            ),
            Node(
                package="me_act_inference_py",
                executable="me_act_inference_node_py",
                name="me_act_inference_node_py",
                output="screen",
                parameters=[
                    {
                        "deploy_dir": LaunchConfiguration("deploy_dir"),
                        "device": LaunchConfiguration("device"),
                        "rgb_topic": LaunchConfiguration("rgb_topic"),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        "servo_command_topic": LaunchConfiguration("servo_command_topic"),
                        "servo_state_service": LaunchConfiguration("servo_state_service"),
                        "command_duration_ms": ParameterValue(LaunchConfiguration("command_duration_ms"), value_type=int),
                        "init_command_duration_ms": ParameterValue(LaunchConfiguration("init_command_duration_ms"), value_type=int),
                        "servo_poll_hz": ParameterValue(LaunchConfiguration("servo_poll_hz"), value_type=float),
                        "servo_request_timeout_ms": ParameterValue(LaunchConfiguration("servo_request_timeout_ms"), value_type=int),
                        "image_sync_queue_size": ParameterValue(LaunchConfiguration("image_sync_queue_size"), value_type=int),
                        "image_sync_slop_s": ParameterValue(LaunchConfiguration("image_sync_slop_s"), value_type=float),
                        "frame_queue_size": ParameterValue(LaunchConfiguration("frame_queue_size"), value_type=int),
                        "servo_cache_maxlen": ParameterValue(LaunchConfiguration("servo_cache_maxlen"), value_type=int),
                        "max_img_state_skew_ms": ParameterValue(LaunchConfiguration("max_img_state_skew_ms"), value_type=int),
                        "max_rgb_depth_skew_ms": ParameterValue(LaunchConfiguration("max_rgb_depth_skew_ms"), value_type=int),
                        "loop_sleep_ms": ParameterValue(LaunchConfiguration("loop_sleep_ms"), value_type=int),
                        "enable_inference_on_start": ParameterValue(LaunchConfiguration("enable_inference_on_start"), value_type=bool),
                        "enable_me_block": True,
                        "validate_servo_ids": ParameterValue(LaunchConfiguration("validate_servo_ids"), value_type=bool),
                        "debug_dump_dir": LaunchConfiguration("debug_dump_dir"),
                        "debug_dump_every_n": ParameterValue(LaunchConfiguration("debug_dump_every_n"), value_type=int),
                        "servo_ids": [1, 2, 3, 4, 5, 10],
                        "init_center": [500, 500, 180, 190, 500, 300],
                        "init_random_range": 40,
                        "physical_min": [0, 100, 50, 50, 50, 150],
                        "physical_max": [1000, 800, 650, 900, 950, 700],
                    }
                ],
            )
        ]
    )
