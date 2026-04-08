# `me_act_inference`

这个目录就是后面直接放到 Jetson `src/` 里的 **完整 ROS2 包**。

你只需要把整个目录复制到：

```bash
~/me_act_ws/src/me_act_inference
```

然后编译即可，不需要再额外拼接 `cpp/` 或 `ros2/` 目录。

## 当前功能

当前是 baseline ACT 在线推理节点：

`rgb + depth + servo_state(service) -> 预处理 -> ACT -> action_seq[0] -> /ros_robot_controller/bus_servo/set_position`

特点：

- 不接 `me_block`
- 不做时间聚合
- 用 service 读取舵机状态
- 有急停和重初始化
- 急停时只是不再发新动作，不关力矩

## 目录说明

- `CMakeLists.txt`
- `package.xml`
- `include/act_pipeline.h`
- `src/act_pipeline.cpp`
- `src/me_act_inference_node.cpp`
- `launch/me_act_baseline.launch.py`

## Jetson 上怎么放

把：

- `deploy/me_act_inference/`

复制成：

```bash
~/me_act_ws/src/me_act_inference
```

另外把模型导出目录单独放，比如：

```bash
~/me_act_models/deploy_artifacts_baseline
```

## 编译

```bash
cd ~/me_act_ws
source /opt/ros/humble/setup.bash
export Torch_DIR=$(python3 -c "import torch, os; print(os.path.join(torch.utils.cmake_prefix_path, 'Torch'))")
colcon build --symlink-install --cmake-args -DTorch_DIR=$Torch_DIR
```

## 启动前先改

先改：

- `launch/me_act_baseline.launch.py`

至少确认：

- `deploy_dir`
- `device`

是你 Jetson 上的真实值。

## 节点接口

订阅：

- `/depth_cam/rgb/image_raw`
- `/depth_cam/depth/image_raw`

调用 service：

- `/ros_robot_controller/bus_servo/get_state`

发布：

- `/ros_robot_controller/bus_servo/set_position`

额外服务：

- `~/start`
- `~/stop`
- `~/emergency_stop`
- `~/initialize`

## 初始化

初始化中心姿态：

- `[500, 560, 120, 180, 500, 240]`

每次 `~/initialize` 时，在每个关节上叠加 `±100` 范围随机扰动，并裁剪到：

- `min = [0, 0, 0, 0, 0, 100]`
- `max = [1000, 1000, 1000, 1000, 1000, 700]`

## 重要提醒

当前代码是按你资料里的接口名写的。  
如果 Jetson 上 `ros_robot_controller_msgs` 里的字段名和这里假设的不完全一致，通常只需要改：

- `src/me_act_inference_node.cpp`

里很小一段消息映射代码。
