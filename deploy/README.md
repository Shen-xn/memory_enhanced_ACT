# Deploy

这套目录把当前项目整理成一条可部署链路：

`raw BGR + raw depth + qpos -> C++ preprocess -> optional me_block -> ACT -> action seq`

## 导出 TorchScript

```powershell
python memory_enhanced_ACT/deploy/export_torchscript_models.py ^
  --act-checkpoint memory_enhanced_ACT/log/exp_xxx/best_model.pth ^
  --me-block-checkpoint memory_enhanced_ACT/log/me_block/importance_xxx/best_model.pth ^
  --output-dir memory_enhanced_ACT/deploy_artifacts ^
  --smoke-test
```

如果是 baseline 单图 ACT，不传 `--me-block-checkpoint` 就行。

导出后会生成：

- `act_inference.pt`
- `me_block_inference.pt`（可选）
- `deploy_config.yml`

## ROS2 / C++ 运行包

真正拿到 Jetson `src/` 里编译的运行代码，现在统一放在：

- `deploy/me_act_inference/`

这个目录本身就是一个完整 ROS2 包。

核心类在：

- `deploy/me_act_inference/include/act_pipeline.h`
- `deploy/me_act_inference/src/act_pipeline.cpp`

主要接口：

- `ActPipeline::Predict(bgr, depth, qpos, use_me_block)`
- `ActPipeline::PredictFromFourChannel(four_channel_bgra, qpos, use_me_block)`
- `ActPipeline::ResetMemory()`

约定：

- `bgr` 输入是 OpenCV 默认 `BGR` 图。
- `depth` 输入是原始单通道深度图，支持 `uint16/float32`。
- 预处理输出的 4 通道顺序是 `BGRA`。
- `me_block` 直接吃 `BGRA`。
- `ACT` wrapper 内部会把 `BGRA` 转成训练时对应的 `RGBA` 再归一化。

## Jetson 上怎么放

模型导出目录单独放，比如：

```bash
~/me_act_models/deploy_artifacts_baseline
```

运行包直接放进 ROS2 工作区：

```bash
~/me_act_ws/src/me_act_inference
```

然后按 `deploy/me_act_inference/README.md` 里的说明编译和启动。
