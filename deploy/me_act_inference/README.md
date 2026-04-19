# me_act_inference

ROS2 deployment package for the single-image ACT runtime.

## Scope

这个包现在支持同一套 deploy 产物接口下的两种模型：

1. **Baseline ACT**
2. **PCA 正交分解方法**

统一推理路径：

```text
rgb + depth + servo_state(service) -> BGRA preprocess -> act_inference.pt -> cmd0 -> bus_servo/set_position
```

已移除：

- `me_block`
- `memory_image`
- 旧在线 memory 专用 launch

## Launch

C++ node:

```bash
ros2 launch me_act_inference me_act_baseline.launch.py
```

Python node:

```bash
ros2 launch me_act_inference_py me_act_baseline_py.launch.py
```

只用 CPU 调试：

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

`deploy_config.yml` 会写明：

- 图像几何 / 输入张量信息
- `predict_delta_qpos`
- `delta_qpos_scale`
- `use_phase_pca_supervision`
- `use_phase_token`
- `phase_pca_dim`

节点本身不需要手工区分 baseline 还是 PCA 模型，它只需要加载导出的 `act_inference.pt`。

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
