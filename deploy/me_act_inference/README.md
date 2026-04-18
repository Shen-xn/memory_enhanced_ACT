# me_act_inference

ROS2 deployment package for the baseline single-image ACT runtime.

## Scope

This package now supports only the baseline deploy path:

```text
rgb + depth + servo_state(service) -> BGRA preprocess -> act_inference.pt -> cmd0 -> bus_servo/set_position
```

Removed from this package:

- `me_block`
- `memory_image`
- online memory-specific launches

## Launch

C++ node:

```bash
ros2 launch me_act_inference me_act_baseline.launch.py
```

Python node:

```bash
ros2 launch me_act_inference_py me_act_baseline_py.launch.py
```

Use CPU only for debugging:

```bash
ros2 launch me_act_inference me_act_baseline.launch.py device:=cpu
ros2 launch me_act_inference_py me_act_baseline_py.launch.py device:=cpu
```

## Expected artifact layout

```text
deploy_artifacts_baseline/
  act_inference.pt
  deploy_config.yml
```

`deploy_config.yml` controls image geometry and tensor layout. The node always builds a BGRA input and feeds it to the exported ACT wrapper.

## Topics and services

Subscribed topics:

- `/depth_cam/rgb/image_raw`
- `/depth_cam/depth/image_raw`

Service client:

- `/ros_robot_controller/bus_servo/get_state`

Published topic:

- `/ros_robot_controller/bus_servo/set_position`

Node services:

- `~/initialize`
- `~/start`
- `~/stop`
- `~/emergency_stop`

## Key parameters

- `deploy_dir`: directory containing `act_inference.pt` and `deploy_config.yml`
- `device`: `cuda` or `cpu`
- `control_period_ms`: C++ node control tick
- `command_duration_ms`: duration of the published servo command
- `init_command_duration_ms`: initialization motion duration
- `servo_state_timeout_ms`: timeout for one servo-state request
- `validate_servo_ids`: optional debug path that reorders qpos by returned servo id

`initialize`, `stop`, and `emergency_stop` all reset the internal deploy pipeline state before the next run.

## Replay node

`me_act_replay_node` replays local task data and publishes servo commands from `states_filtered.csv`.

Example:

```bash
ros2 run me_act_inference me_act_replay_node --ros-args \
  -p task_dir:=/path/to/task_xxx \
  -p rgb_dirname:=rgb \
  -p depth_dirname:=depth_normalized \
  -p states_filename:=states_filtered.csv \
  -p publish_period_ms:=200 \
  -p command_duration_ms:=220 \
  -p start_on_launch:=true
```
