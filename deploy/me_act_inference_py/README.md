# me_act_inference_py

纯 Python ROS2 部署包，功能口径与 C++ `me_act_inference` 节点一致，但不依赖 LibTorch C++ 编译环境，更适合上机快速测试。

当前实际 launch 使用的节点脚本是：

```text
me_act_inference_py/me_act_inference_node_py.py
```

备用脚本 `me_act_inference_node_py_compatible_simple.py` 没有注册到 `console_scripts`，默认不会运行。

## 启动

```bash
ros2 launch me_act_inference_py me_act_baseline_py.launch.py \
  deploy_dir:=/home/ubuntu/my_models/me_act/deploy_artifacts_baseline \
  device:=cuda
```

节点名固定为：

```text
/me_act_inference_node_py
```

控制服务：

```bash
ros2 service call /me_act_inference_node_py/initialize std_srvs/srv/Trigger {}
ros2 service call /me_act_inference_node_py/start std_srvs/srv/Trigger {}
ros2 service call /me_act_inference_node_py/stop std_srvs/srv/Trigger {}
ros2 service call /me_act_inference_node_py/emergency_stop std_srvs/srv/Trigger {}
```

不要和 C++ `me_act_inference_node` 同时控制同一台机械臂，它们都会发布到同一个舵机命令 topic。

## Temporal Aggregation

默认关闭，不改变原始 `cmd0` 控制逻辑。

开启后，当前发送命令由多个历史预测投票得到：

```text
send(t) = weighted_average(
  pred_at_t(cmd0),    weight = 1
  pred_at_t-1(cmd1),  weight = decay
  pred_at_t-2(cmd2),  weight = decay^2
  ...
)
```

`decay` 越小，越相信当前 `cmd0`；`decay=0` 等价于不用时间聚合；`decay=1` 等价于所有可用投票等权平均。

示例：

```bash
ros2 launch me_act_inference_py me_act_baseline_py.launch.py \
  deploy_dir:=/home/ubuntu/my_models/me_act/deploy_artifacts_pca16residual_fixed \
  device:=cuda \
  temporal_agg_enabled:=true \
  temporal_agg_decay:=0.7
```
