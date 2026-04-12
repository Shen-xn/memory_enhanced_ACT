# me_act_inference

这个目录就是放进 Jetson ROS2 工作区 `src/` 的完整包，例如：

```bash
~/ros2_ws/src/me_act_inference
```

## 两种运行模式

### 1. baseline ACT

```text
rgb + depth + servo_state(service) -> preprocess(BGRA) -> ACT -> action_seq[0] -> bus_servo/set_position
```

启动：

```bash
ros2 launch me_act_inference me_act_baseline.launch.py
```

### 2. me_act（在线 `me_block`）

```text
rgb + depth + servo_state(service) -> preprocess(BGRA) -> me_block -> memory_image -> ACT -> action_seq[0] -> bus_servo/set_position
```

启动：

```bash
ros2 launch me_act_inference me_act_memory.launch.py
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

## 编译

这个包按 ROS2 Humble 口径维护。舵机状态 service 使用 Humble 支持的 `create_client(..., rmw_qos_profile_services_default, callback_group)` 写法，不依赖新版本里的 `rclcpp::ClientOptions`。

如果环境已经写进 `~/.zshrc`，直接：

```bash
export Torch_DIR=$(python3 -c "import torch, os; print(os.path.join(torch.utils.cmake_prefix_path, 'Torch'))")
colcon build --symlink-install \
  --packages-select me_act_inference \
  --cmake-args -DTorch_DIR=$Torch_DIR
```

如果没有自动 source，再先：

```bash
source /opt/ros/humble/setup.zsh
source /home/ubuntu/third_party_ros2/third_party_ws/install/setup.zsh
source /home/ubuntu/third_party_ros2/orbbec_ws/install/setup.zsh
source /home/ubuntu/ros2_ws/install/setup.zsh
```

## 启动前一定要改

### baseline

改：

- [`launch/me_act_baseline.launch.py`](./launch/me_act_baseline.launch.py)

至少确认：

- `deploy_dir`
- `device`
- `servo_state_timeout_ms`

例如：

```text
/home/ubuntu/my_models/me_act/deploy_artifacts_baseline
```

### memory 版

改：

- [`launch/me_act_memory.launch.py`](./launch/me_act_memory.launch.py)

至少确认：

- `deploy_dir`
- `device`
- `servo_state_timeout_ms`

例如：

```text
/home/ubuntu/my_models/me_act/deploy_artifacts_memory
```

## 调试顺序

1. 启动节点
2. 初始化：

```bash
ros2 service call /me_act_inference_node/initialize std_srvs/srv/Trigger "{}"
```

3. 开始：

```bash
ros2 service call /me_act_inference_node/start std_srvs/srv/Trigger "{}"
```

4. 停止：

```bash
ros2 service call /me_act_inference_node/stop std_srvs/srv/Trigger "{}"
```

5. 急停：

```bash
ros2 service call /me_act_inference_node/emergency_stop std_srvs/srv/Trigger "{}"
```

## 初始化说明

初始化中心姿态：

```text
[500, 560, 120, 180, 500, 240]
```

每次初始化都会加 `±100` 随机扰动，再裁剪到：

```text
min = [0, 0, 0, 0, 0, 100]
max = [1000, 1000, 1000, 1000, 1000, 700]
```

## 最重要的检查项

- baseline launch 默认 `enable_me_block = False`
- memory launch 默认 `enable_me_block = True`
- 如果 memory launch 打开了 `enable_me_block`，但导出目录里没有 `me_block_inference.pt`，节点会直接启动失败
- baseline 和 memory 版的导出目录不要混用
- 双图 ACT 必须是重新按当前代码口径训练出来的模型
- 舵机状态查询是异步的：控制 timer 只发送请求，response 回调回来后才做 ACT 推理和发命令。这样避免 timer 回调里同步等 service response 导致 executor 假超时或死锁
- `servo_state_timeout_ms` 默认 500ms。你手动测试 get_state 大约 30-40ms，所以正常情况下不应该触发；如果触发，优先检查 service 回调是否被其它节点阻塞、消息字段是否变了，或者 executor 是否被单线程 launch/嵌套 spin 改坏
- `initialize`、`stop`、`emergency_stop` 都会重置在线 memory state；节点内部会对推理和重置加锁，避免并发访问同一份 memory

## 你后面自己排查时优先看哪几处

1. `deploy_dir` 是否指对
2. `ros2 topic list` / `ros2 service list` 是否包含相机和舵机接口
3. `initialize` 是否成功
4. 节点日志里是否出现：
   - frame 太旧
   - image/state skew 太大
   - servo state service timeout
5. 如果换平台后消息字段不一致，优先改：

- [`src/me_act_inference_node.cpp`](./src/me_act_inference_node.cpp)
