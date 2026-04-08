# Deploy

`deploy/` 只保留两类东西：

1. 模型导出脚本
2. Jetson / ROS2 运行包

## 导出 TorchScript

baseline ACT：

```powershell
python memory_enhanced_ACT/deploy/export_torchscript_models.py `
  --act-checkpoint memory_enhanced_ACT/log/exp_xxx/best_model.pth `
  --output-dir memory_enhanced_ACT/deploy_artifacts_baseline `
  --smoke-test
```

如果要同时导出 `me_block`：

```powershell
python memory_enhanced_ACT/deploy/export_torchscript_models.py `
  --act-checkpoint memory_enhanced_ACT/log/exp_xxx/best_model.pth `
  --me-block-checkpoint memory_enhanced_ACT/log/me_block/importance_xxx/best_model.pth `
  --output-dir memory_enhanced_ACT/deploy_artifacts `
  --smoke-test
```

导出产物：

- `act_inference.pt`
- `me_block_inference.pt`（可选）
- `deploy_config.yml`

## 当前统一口径

- 外部四通道图接口：`BGRA`
- `me_block` 直接吃 `BGRA`
- ACT 训练和部署都以 `BGRA` 为外部接口，再在归一化前重排颜色通道
- 关节归一化固定为物理范围，不再按数据集统计

固定关节范围：

```text
joint_min = [0, 0, 0, 0, 0, 100]
joint_max = [1000, 1000, 1000, 1000, 1000, 700]
joint_rng = [1000, 1000, 1000, 1000, 1000, 600]
```

## Jetson 上真正要放进工作区的包

直接把这个目录放进 ROS2 工作区：

- `deploy/me_act_inference/`

它本身就是一个完整 ROS2 包。

详细说明看：

- [`deploy/me_act_inference/README.md`](./me_act_inference/README.md)
