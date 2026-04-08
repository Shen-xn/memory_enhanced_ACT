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

## C++ 接口

核心类在：

- `deploy/cpp/include/act_pipeline.h`
- `deploy/cpp/src/act_pipeline.cpp`

接口：

- `ActPipeline::Predict(bgr, depth, qpos, use_me_block)`
- `ActPipeline::PredictFromFourChannel(four_channel_bgra, qpos, use_me_block)`
- `ActPipeline::ResetMemory()`

约定：

- `bgr` 输入是 OpenCV 默认 `BGR` 图。
- `depth` 输入是原始单通道深度图，支持 `uint16/float32`。
- 预处理输出的 4 通道顺序是 `BGRA`。
- `me_block` 直接吃 `BGRA`。
- `ACT` wrapper 内部会把 `BGRA` 转成训练时对应的 `RGBA` 再归一化。

## Linux / Jetson 构建

```bash
cd memory_enhanced_ACT/deploy/cpp
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DOpenCV_DIR=/path/to/opencv
cmake --build build --config Release
```

## Demo

```bash
./build/act_infer_demo \
  /path/to/deploy_artifacts \
  /path/to/rgb.jpg \
  /path/to/depth.png \
  "0.1,0.2,0.3,0.4,0.5,0.6"
```
