# Deploy

`deploy/` 负责两件事：

1. 导出 TorchScript
2. 提供 Jetson / ROS2 运行包

## 当前部署口径

- 外部四通道接口统一为 `BGRA`
- ACT 和 `me_block` 都以 `BGRA` 作为部署输入
- 关节归一化统一使用固定物理范围：

```text
joint_min = [0, 0, 0, 0, 0, 100]
joint_max = [1000, 1000, 1000, 1000, 1000, 700]
joint_rng = [1000, 1000, 1000, 1000, 1000, 600]
```

## 导出 TorchScript

### baseline ACT

```powershell
python deploy/export_torchscript_models.py `
  --act-checkpoint .\log\exp_xxx\best_model.pth `
  --output-dir .\deploy_artifacts_baseline `
  --smoke-test
```

产物：

- `act_inference.pt`
- `deploy_config.yml`

### me_act（双图 ACT + `me_block`）

```powershell
python deploy/export_torchscript_models.py `
  --act-checkpoint .\log\exp_xxx\best_model.pth `
  --me-block-checkpoint .\log\me_block\importance_xxx\best_model.pth `
  --output-dir .\deploy_artifacts_memory `
  --smoke-test
```

产物：

- `act_inference.pt`
- `me_block_inference.pt`
- `deploy_config.yml`

## 导出约束

- baseline 单图 ACT 不能带 `--me-block-checkpoint`
- 双图 ACT 必须带 `--me-block-checkpoint`
- 双图 ACT 的 checkpoint 必须是 `USE_MEMORY_IMAGE_INPUT = True` 训练出来的
- 当前部署链里，双图 ACT 的 `memory_image` 由在线 `me_block` 生成，不再额外读取离线 PNG

## 换到正式训练机后怎么做

### baseline

1. 训练 baseline ACT
2. 导出到 `deploy_artifacts_baseline`
3. 拷到 Jetson
4. 启动 `me_act_baseline.launch.py`

### me_act

1. 训练 `me_block`
2. 训练双图 ACT
3. 导出到 `deploy_artifacts_memory`
4. 拷到 Jetson
5. 启动 `me_act_memory.launch.py`

## 拷到 Jetson 的内容

模型目录单独放，例如：

```text
/home/ubuntu/my_models/me_act/deploy_artifacts_baseline
/home/ubuntu/my_models/me_act/deploy_artifacts_memory
```

ROS2 包放进工作区：

```text
deploy/me_act_inference/
```

详细启动方式看：

- [`deploy/me_act_inference/README.md`](./me_act_inference/README.md)

## 部署前检查清单

- 在导出机上先跑 `--smoke-test`
- Jetson 上 launch 文件里的 `deploy_dir` 指向真实路径
- baseline 和 memory 版不要混用目录
- 如果打开了 `enable_me_block`，导出目录里必须真的有 `me_block_inference.pt`
