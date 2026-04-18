# memory_enhanced_ACT

当前主线已经收敛为一条方法：

- 单图 ACT 输入：`four_channel/` 的 BGRA 四通道图
- `phase token` 只进主 transformer 的 encoder
- `phase token` 不给主 decoder
- `prototype` 分支预测全关节动作模板混合比例
- `residual` 分支预测全关节残差动作
- 最终动作：`prototype_mix + residual`

旧的 `me_block / memory_image / 简单 prototype 分类损失` 已经从工程主线移除。

## 数据约定

数据根目录默认是：

```text
data_process/data/
```

每个任务目录至少需要：

```text
task_xxx/
  four_channel/
  states_filtered.csv
  phase_proto_targets.npz
```

其中：

- `four_channel/*.png` 是 OpenCV 口径的 `BGRA`
- `states_filtered.csv` 至少包含：
  - `frame`
  - `j1 j2 j3 j4 j5 j10`
- `phase_proto_targets.npz` 是离线预计算好的监督：
  - `frame_index`
  - `alpha_tgt`
  - `prototype_tgt`
  - `residual_tgt`

## 训练前离线准备

先生成 phase prototype bank 和每个任务的 `alpha_tgt / residual_tgt`：

```powershell
python tools/build_phase_prototype_targets.py `
  --data-root F:\预处理后的新数据 `
  --bank-output .\data_process\data\_phase_proto\phase_proto_bank.npz `
  --clusters 16 `
  --pca-var-ratio 0.85 `
  --future-steps 10 `
  --target-mode delta `
  --delta-qpos-scale 10
```

生成完成后：

- bank 会写到你指定的 `--bank-output`
- 每个 `task_xxx/` 下会写出一个 `phase_proto_targets.npz`

然后把 [`config.py`](./config.py) 里的：

- `DATA_ROOT`
- `PHASE_BANK_PATH`

对到你实际路径。

## 训练

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

训练时主损失是：

- `recon_l1`
- `residual_l1`
- `prototype_mse`
- `kl`

总损失在 [`act/policy.py`](./act/policy.py) 里加权求和。

## 导出与部署

当前 deploy 只保留 baseline 单图 ACT 路径。

导出：

```powershell
python deploy/export_torchscript_models.py `
  --act-checkpoint .\log\exp_xxx\best_model.pth `
  --output-dir .\deploy_artifacts_baseline `
  --smoke-test
```

导出目录包含：

- `act_inference.pt`
- `deploy_config.yml`

部署侧仍然是：

```text
rgb + depth + qpos -> BGRA four-channel -> ACT -> action sequence
```
