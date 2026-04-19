# Deploy

当前 deploy 只保留**单图 ACT 导出路径**，但它兼容两种训练模型：

1. **Baseline ACT**
2. **PCA 正交分解方法**

也就是说，deploy 入口是同一套，但导出的 `act_inference.pt` 会跟随 checkpoint 的训练口径自动变化。

推理路径：

```text
rgb + depth + qpos -> preprocess(BGRA) -> act_inference.pt -> action sequence
```

已移除：

- `me_block`
- `memory_image`
- 旧双图 ACT deploy 分支

## Export

```powershell
python deploy/export_torchscript_models.py `
  --act-checkpoint .\log\exp_xxx\best_model.pth `
  --output-dir .\deploy_artifacts_baseline `
  --smoke-test
```

导出文件：

- `act_inference.pt`
- `deploy_config.yml`

## deploy_config.yml

导出配置里会写出通用推理字段：

- `target_width`
- `target_height`
- `pad_left`
- `pad_top`
- `depth_clip_min`
- `depth_clip_max`
- `state_dim`
- `num_queries`
- `image_channels`
- `predict_delta_qpos`
- `delta_qpos_scale`

同时也会显式写出模型口径：

- `use_phase_pca_supervision`
- `use_phase_token`
- `phase_pca_dim`

所以你拿到 deploy 产物时，可以直接判断它是：

- baseline 导出的模型
- 还是 PCA 正交分解方法导出的模型

## TorchScript wrapper

Wrapper 实现在：

- [deploy_wrappers.py](./deploy_wrappers.py)

职责：

- 归一化原始 `qpos`
- 把 BGRA 图像转成模型输入顺序
- 调用导出的 ACT 模型
- 把动作输出解码回物理关节空间

对于 `delta qpos` 模型：

- wrapper 会再乘 `delta_qpos_scale`
- 再做时间维累加
- 最后加回当前 `qpos`

这一点对 baseline 和 PCA 方法都是统一的。
