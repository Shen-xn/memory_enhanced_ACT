from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "device",
                default_value="cuda",
                description="Torch device used by ACT. Use cpu only for debugging.",
            ),
            Node(
                package="me_act_inference",
                executable="me_act_inference_node",
                name="me_act_inference_node",
                output="screen",
                parameters=[
                    {
                        "deploy_dir": "/home/ubuntu/my_models/me_act/deploy_artifacts_baseline",
                        "device": LaunchConfiguration("device"),
                        "rgb_topic": "/depth_cam/rgb/image_raw",
                        "depth_topic": "/depth_cam/depth/image_raw",
                        "servo_command_topic": "/ros_robot_controller/bus_servo/set_position",
                        "servo_state_service": "/ros_robot_controller/bus_servo/get_state",
                        "control_period_ms": 100,
                        "command_duration_ms": 300,
                        "init_command_duration_ms": 1500,
                        "max_frame_age_ms": 250,
                        "max_state_image_skew_ms": 150,
                        "servo_state_timeout_ms": 500,
                        "sync_queue_size": 10,
                        "enable_inference_on_start": False,
                        "enable_me_block": False,
                        "validate_servo_ids": False,
                        "servo_ids": [1, 2, 3, 4, 5, 10],
                        "init_center": [500, 500, 180, 190, 500, 300],
                        "init_random_range": 40,
                        "physical_min": [0, 0, 0, 0, 0, 100],
                        "physical_max": [1000, 1000, 1000, 1000, 1000, 700],
                    }
                ],
            )
        ]
    )
