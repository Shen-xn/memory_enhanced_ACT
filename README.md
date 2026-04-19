# memory_enhanced_ACT

当前仓库支持两条训练口径：

1. **Baseline ACT**
   - 不使用 phase token
   - 不使用 PCA 离线监督
   - 直接做标准 ACT 动作回归

2. **PCA 正交分解方法**
   - `phase token` 只进 encoder
   - `phase token` 不给 decoder
   - PCA 头预测 `16` 维正交坐标
   - residual 头预测全关节残差
   - 最终动作：`pca_recon + residual`

旧的 `PCA聚类方法 / alpha_tgt / prototype mixture / me_block / memory_image` 已退出训练主线。

## 一眼看懂当前开关

最关键的总开关在 [config.py](./config.py)：

```python
USE_PHASE_PCA_SUPERVISION = False   # baseline
USE_PHASE_PCA_SUPERVISION = True    # PCA正交分解方法
```

当它为：

- `False`
  - 不需要 `phase_pca16_targets.npz`
  - 不需要 PCA bank
  - 不加载 phase token / PCA head
  - loss 只用 `recon_l1 + kl`

- `True`
  - 需要 `phase_pca16_targets.npz`
  - 需要 `_phase_pca16/phase_pca16_bank.npz`
  - 启用 phase token / PCA head / residual head
  - loss 用 `recon_l1 + residual_l1 + pca_coord_mse + kl`

## 数据目录

训练数据根目录默认是：

```text
data_process/data/
```

### Baseline 所需文件

```text
task_xxx/
  four_channel/
  states_filtered.csv
```

### PCA 正交分解方法额外所需文件

```text
task_xxx/
  four_channel/
  states_filtered.csv
  phase_pca16_targets.npz

data_process/data/
  _phase_pca16/
    phase_pca16_bank.npz
```

其中：

- `four_channel/*.png` 是 OpenCV 口径 `BGRA`
- `states_filtered.csv` 至少包含：
  - `frame`
  - `j1 j2 j3 j4 j5 j10`
- `phase_pca16_targets.npz` 包含：
  - `frame_index`
  - `pca_coord_tgt`
  - `pca_recon_tgt`
  - `residual_tgt`

全局 bank `phase_pca16_bank.npz` 包含：

- `pca_mean`
- `pca_components`
- `pca_coord_mean/std`
- `residual_mean/std`
- `explained_ratio`

## 训练

先把 [config.py](./config.py) 里的：

- `DATA_ROOT`
- `USE_PHASE_PCA_SUPERVISION`
- `PHASE_BANK_PATH`（只在 PCA 方法下需要）

改到你的实际路径 / 方案，然后直接跑：

```powershell
python training.py
```

日志会写到：

```text
log/exp_YYYYMMDD_HHMMSS/
```

主要输出：

- `config.json`
- `metrics.jsonl`
- `training_curves.png`
- `ckpt_epoch_X.pth`
- `best_model.pth`

## 当前损失口径

### Baseline

- `recon_l1`
- `kl`

总损失：

```text
loss = RECON_LOSS_WEIGHT * recon_l1 + KL_WEIGHT * kl
```

### PCA 正交分解方法

- `recon_l1`
- `residual_l1`
- `pca_coord_mse`
- `kl`

总损失：

```text
loss =
    RECON_LOSS_WEIGHT * recon_l1
  + RESIDUAL_LOSS_WEIGHT * residual_l1
  + PCA_COORD_LOSS_WEIGHT * pca_coord_mse
  + KL_WEIGHT * kl
```

其中：

- `pca_coord` 和 `residual` 都用 bank 中的 `mean/std` 做归一化监督
- `recon_l1` 始终在原动作空间监督最终动作

## Deploy / 导出

当前 deploy 走**同一套单图 ACT 导出路径**，兼容两种训练模型：

- baseline checkpoint
- PCA 正交分解 checkpoint

导出命令：

```powershell
python deploy/export_torchscript_models.py `
  --act-checkpoint .\log\exp_xxx\best_model.pth `
  --output-dir .\deploy_artifacts_baseline `
  --smoke-test
```

导出目录包含：

- `act_inference.pt`
- `deploy_config.yml`

`deploy_config.yml` 里现在会显式写出：

- `use_phase_pca_supervision`
- `use_phase_token`
- `phase_pca_dim`
- `predict_delta_qpos`
- `delta_qpos_scale`

也就是说，同一个 deploy 导出入口会自动跟随 checkpoint：

- baseline 模型：导出 baseline 推理逻辑
- PCA 模型：导出 `pca_recon + residual` 推理逻辑
