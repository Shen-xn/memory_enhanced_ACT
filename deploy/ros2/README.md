# ROS2 Baseline 节点

这个包是 baseline ACT 的 ROS2 在线推理节点。

当前链路：

`rgb + depth + servo_state(service) -> 预处理 -> ACT -> action_seq[0] -> /ros_robot_controller/bus_servo/set_position`

## 设计原则

- 先只做 baseline，不接 me_block
- 不做时间聚合
- 舵机状态优先用 service 主动读取
- 提供急停和重初始化
- 急停时“不再下发新动作”，不主动关力矩

## 输入输出

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

## 状态机

- `IDLE`：已启动，但不推理
- `INITIALIZING`：正在发随机初始化姿态
- `RUNNING`：正常推理控制
- `ESTOP`：急停，停止发布动作
- `FAULT`：内部错误

## 初始化

初始化中心位姿：

- `[500, 560, 120, 180, 500, 240]`

每次 `~/initialize` 时，会在每个关节上加 `±100` 范围内的随机扰动，然后裁剪到物理范围：

- `min = [0, 0, 0, 0, 0, 100]`
- `max = [1000, 1000, 1000, 1000, 1000, 700]`

## 时序策略

这版为了保证调试清楚，采用保守策略：

1. 用 message_filters 同步 RGB/depth
2. timer 到点时取最近一对同步图像
3. 立刻通过 service 读取当前舵机状态
4. 如果图像和状态时间差太大，则跳过这一拍
5. 只执行 `action_seq[0]`
6. 上一条动作的执行窗口没过，不发下一条

这样虽然保守，但观测和执行关系更清楚，便于后续做时间聚合。

## 你需要注意的地方

这个包里默认使用 `ros_robot_controller_msgs` 作为消息/服务包名，并假设：

- `ServosPosition` 里有 `duration` 和 `servos`
- `ServoPosition` 里有 `id` 和 `position`
- `GetBusServoState` request 里有 `id`
- `GetBusServoState` response 里有 `position`

如果你 Jetson 上实际字段名不完全一样，只需要在 `src/me_act_inference_node.cpp` 里改很小一段映射逻辑。

## 启动前准备

1. 先导出 baseline：

```bash
python deploy/export_torchscript_models.py \
  --act-checkpoint /path/to/best_model.pth \
  --output-dir /path/to/deploy_artifacts_baseline \
  --data-root /path/to/data_process/data \
  --smoke-test
```

2. 在 Jetson 上准备：

- `deploy_artifacts_baseline/`
- `deploy/cpp/`
- `deploy/ros2/`

## 启动

把 `deploy_dir` 改成你的实际路径后：

```bash
ros2 launch me_act_inference me_act_baseline.launch.py
```

## 调试顺序建议

1. 先启动节点，但默认不推理
2. 调用 `~/initialize`
3. 检查机械臂是否到达随机初始化姿态
4. 调用 `~/start`
5. 观察是否持续下发 `/bus_servo/set_position`
6. 出现异常时，立即调用 `~/emergency_stop`
