# me_act_inference_py

纯 Python 的 ROS2 部署包，功能口径与 `me_act_inference` 的 C++ 推理节点保持一致，但单独作为一个包存在，方便单独传输和编译。

这个包现在使用标准 `ament_python` 安装方式，不需要手动 `chmod +x`。

## 启动

baseline:

```bash
ros2 launch me_act_inference_py me_act_baseline_py.launch.py
```

memory:

```bash
ros2 launch me_act_inference_py me_act_memory_py.launch.py
```

节点名固定为 `me_act_inference_node_py`，控制 service 是：

```bash
/me_act_inference_node_py/initialize
/me_act_inference_node_py/start
/me_act_inference_node_py/stop
/me_act_inference_node_py/emergency_stop
```

不要和 `me_act_inference` 的 C++ 节点同时控制同一台机械臂，它们会发往同一个舵机命令 topic。
