# memory_enhanced_ACT

当前主线只保留一套方法：

- 单图 ACT 输入，训练图像来自 `four_channel/*.png`
- `phase token` 只进 transformer encoder
- `phase token` 不给主 decoder
- `PCA` 基础轨迹分支预测 `16` 维正交坐标
- `residual` 分支预测全关节细节残差
- 最终动作：`pca_recon + residual`

旧的 `PCA聚类方法 / alpha_tgt / prototype mixture / me_block / memory_image` 已经退出训练主线。

## 数据约定

训练目录默认是：

```text
data_process/data/
```

每个任务目录至少需要：

```text
task_xxx/
  four_channel/
  states_filtered.csv
  phase_pca16_targets.npz
```

其中：

- `four_channel/*.png` 是 OpenCV 口径的 `BGRA`
- `states_filtered.csv` 至少包含：
  - `frame`
  - `j1 j2 j3 j4 j5 j10`
- `phase_pca16_targets.npz` 包含：
  - `frame_index`
  - `pca_coord_tgt`
  - `pca_recon_tgt`
  - `residual_tgt`

全局 bank 默认放在：

```text
data_process/data/_phase_pca16/phase_pca16_bank.npz
```

它会保存：

- `pca_mean`
- `pca_components`
- `pca_coord_mean/std`
- `residual_mean/std`
- `explained_ratio`

## 训练

先把 [`config.py`](./config.py) 里的：

- `DATA_ROOT`
- `PHASE_BANK_PATH`

改到你的实际路径，然后直接跑：

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

## 当前损失

当前 ACT 主损失是：

- `recon_l1`
- `residual_l1`
- `pca_coord_mse`
- `kl`

总损失在 [`act/policy.py`](./act/policy.py) 里加权求和。

其中：

- `pca_coord` 和 `residual` 都用 bank 里的 `mean/std` 做归一化监督
- `recon_l1` 仍然在原动作空间监督最终动作

## 导出与部署

当前 deploy 仍然走 baseline 单图 ACT 路径：

```powershell
python deploy/export_torchscript_models.py `
  --act-checkpoint .\log\exp_xxx\best_model.pth `
  --output-dir .\deploy_artifacts_baseline `
  --smoke-test
```

导出目录包含：

- `act_inference.pt`
- `deploy_config.yml`
