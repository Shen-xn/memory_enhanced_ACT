# me_act_inference

这个目录是 Jetson/ROS2 侧的部署节点包，目标路径通常是：

```bash
~/ros2_ws/src/me_act_inference
```

## 运行模式

### RGB baseline

RGB baseline 的 artifact 会在 `deploy_config.yml` 里写入 `image_channels: 3`。ROS 节点仍然构造 BGRA 输入，TorchScript wrapper 会忽略 depth 通道，所以相机同步和 C++ preprocessing 不需要分叉。

### baseline ACT

```text
rgb + depth + servo_state(service) -> preprocess(BGRA) -> ACT -> action_seq[0] -> bus_servo/set_position
```

启动：

```bash
ros2 launch me_act_inference me_act_baseline.launch.py
```

### ACT + online me_block

```text
rgb + depth + servo_state(service) -> preprocess(BGRA) -> me_block -> memory_image -> ACT -> action_seq[0] -> bus_servo/set_position
```

启动：

```bash
ros2 launch me_act_inference me_act_memory.launch.py
```

## GPU

两个 launch 默认使用 CUDA：

```bash
ros2 launch me_act_inference me_act_memory.launch.py
ros2 launch me_act_inference me_act_baseline.launch.py
```

如果要临时用 CPU 调试：

```bash
ros2 launch me_act_inference me_act_memory.launch.py device:=cpu
ros2 launch me_act_inference me_act_baseline.launch.py device:=cpu
```

节点启动时如果设置了 `device:=cuda`，但当前 LibTorch 不支持 CUDA，会直接报错。先在 Jetson 上确认：

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

编译时也要让 CMake 找到同一个 PyTorch/LibTorch：

```bash
export Torch_DIR=$(python3 -c "import torch, os; print(os.path.join(torch.utils.cmake_prefix_path, 'Torch'))")
colcon build --symlink-install \
  --packages-select me_act_inference \
  --cmake-args -DTorch_DIR=$Torch_DIR
```

## Artifact 同步

ROS 节点不重新实现 me_block 后处理，所有模型结构和 memory update 都来自导出的 TorchScript：

```text
deploy_artifacts_memory/
  act_inference.pt
  me_block_inference.pt
  deploy_config.yml
```

所以修改 me_block 之后，需要重新导出 deployment artifacts。当前 me_block 后处理保留 blur，已经移除 opening；导出的 `me_block_inference.pt` 会带上这个逻辑。

baseline artifact 目录不应该包含 `me_block_inference.pt`：

```text
deploy_artifacts_baseline/
  act_inference.pt
  deploy_config.yml
```

## 节点接口

订阅：

- `/depth_cam/rgb/image_raw`
- `/depth_cam/depth/image_raw`

调用 service：

- `/ros_robot_controller/bus_servo/get_state`

发布：

- `/ros_robot_controller/bus_servo/set_position`

控制 service：

- `~/initialize`
- `~/start`
- `~/stop`
- `~/emergency_stop`

## 关键参数

- `deploy_dir`: TorchScript artifact 目录。
- `device`: `cuda` 或 `cpu`。
- `enable_me_block`: memory launch 为 `true`，baseline launch 为 `false`。
- `validate_servo_ids`: 默认 `false`。调试 qpos 顺序时可临时设为 `true`，节点会额外读取舵机 ID，并按 `servo_ids` 重排状态；代价是每个控制 tick 多一次 ID 查询。
- `control_period_ms`: 控制 tick 周期。
- `command_duration_ms`: 每次舵机命令写入的运动时长。
- `init_command_duration_ms`: 初始化动作时长，默认 1500ms。
- `servo_state_timeout_ms`: 查询舵机状态的超时时间。

`initialize`、`stop`、`emergency_stop` 都会重置在线 memory state。节点内部会对推理和 memory reset 加锁，避免并发访问同一份 recurrent memory。

## 调试顺序

1. 确认 `deploy_dir` 指向正确 artifact 目录。
2. 确认 `device:=cuda` 时 PyTorch/LibTorch 真的支持 CUDA。
3. 确认相机 topic 和舵机 service 存在。
4. 调用初始化：

```bash
ros2 service call /me_act_inference_node/initialize std_srvs/srv/Trigger "{}"
```

5. 开始推理：

```bash
ros2 service call /me_act_inference_node/start std_srvs/srv/Trigger "{}"
```

6. 停止：

```bash
ros2 service call /me_act_inference_node/stop std_srvs/srv/Trigger "{}"
```

7. 急停：

```bash
ros2 service call /me_act_inference_node/emergency_stop std_srvs/srv/Trigger "{}"
```

## 演示回放节点

`me_act_replay_node` 用于播放本地任务数据并按 `states_filtered.csv` 逐帧下发舵机位置，方便验证数据与动作是否一致。

启动示例：
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

常用参数：
- `task_dir`: 任务目录（包含 rgb/depth_normalized/states_filtered.csv）
- `publish_period_ms`: 回放周期
- `command_duration_ms`: 下发到舵机的持续时长
- `loop`: 是否循环回放
- `start_on_launch`: 启动即开始回放
