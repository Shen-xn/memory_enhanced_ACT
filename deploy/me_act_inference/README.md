# me_act_inference

这个目录就是要放进 Jetson ROS2 工作区 `src/` 里的完整包。

例如：

```bash
~/ros2_ws/src/me_act_inference
```

## 当前功能

当前节点是 baseline ACT 在线推理节点：

```text
rgb + depth + servo_state(service) -> preprocess(BGRA) -> ACT -> action_seq[0] -> bus_servo/set_position
```

当前不做：

- `me_block`
- 时间聚合
- 关力矩急停

急停只是停止继续发新动作。

## 节点接口

订阅：

- `/depth_cam/rgb/image_raw`
- `/depth_cam/depth/image_raw`

调用 service：

- `/ros_robot_controller/bus_servo/get_state`

发布：

- `/ros_robot_controller/bus_servo/set_position`

额外控制 service：

- `~/initialize`
- `~/start`
- `~/stop`
- `~/emergency_stop`

## 初始化

初始化中心姿态：

```text
[500, 560, 120, 180, 500, 240]
```

每次初始化都会加 `±100` 随机扰动，再裁剪到：

```text
min = [0, 0, 0, 0, 0, 100]
max = [1000, 1000, 1000, 1000, 1000, 700]
```

## 编译

如果这些环境已经写进 `~/.zshrc`，直接在工作区根目录执行：

```bash
export Torch_DIR=$(python3 -c "import torch, os; print(os.path.join(torch.utils.cmake_prefix_path, 'Torch'))")
colcon build --symlink-install \
  --packages-select me_act_inference \
  --cmake-args -DTorch_DIR=$Torch_DIR
```

如果没有自动 source，再先 source：

```bash
source /opt/ros/humble/setup.zsh
source /home/ubuntu/third_party_ros2/third_party_ws/install/setup.zsh
source /home/ubuntu/third_party_ros2/orbbec_ws/install/setup.zsh
source /home/ubuntu/ros2_ws/install/setup.zsh
```

## 启动前要改

修改：

- [`launch/me_act_baseline.launch.py`](./launch/me_act_baseline.launch.py)

至少确认：

- `deploy_dir`
- `device`

其中 `deploy_dir` 要指向你 Jetson 上真实的导出目录，比如：

```text
/home/ubuntu/my_models/me_act/deploy_artifacts_baseline
```

## 启动

```bash
ros2 launch me_act_inference me_act_baseline.launch.py
```

## 调试顺序

先初始化：

```bash
ros2 service call /me_act_inference_node/initialize std_srvs/srv/Trigger "{}"
```

再开始：

```bash
ros2 service call /me_act_inference_node/start std_srvs/srv/Trigger "{}"
```

停止：

```bash
ros2 service call /me_act_inference_node/stop std_srvs/srv/Trigger "{}"
```

急停：

```bash
ros2 service call /me_act_inference_node/emergency_stop std_srvs/srv/Trigger "{}"
```

## 说明

当前代码已经按这套接口对齐过：

- `ros_robot_controller_msgs/srv/GetBusServoState`
- `ros_robot_controller_msgs/msg/ServosPosition`
- `ros_robot_controller_msgs/msg/ServoPosition`

如果你后面换平台，最常改的地方还是：

- [`src/me_act_inference_node.cpp`](./src/me_act_inference_node.cpp)
