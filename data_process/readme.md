### Data Processing

处理完成后的训练任务目录位于 `data_process/data/task_*`。

原始采样目录：

```text
task_xxx/
  rgb/
  depth/
  states.csv
```

训练可用目录：

```text
task_xxx/
  four_channel/
  states_filtered.csv
  phase_pca16_targets.npz
```

## 推荐入口

从项目根目录运行：

```powershell
python prepare_act_data.py
```

这条链路会：

1. 清洗 `states.csv`
2. 对齐状态、RGB、Depth
3. 生成 `depth_normalized/` 和 `four_channel/`
4. 放大夹爪关节幅值
5. 校验最终训练文件

最终校验包括：

- `states_filtered.csv` 存在并包含 `frame,j1,j2,j3,j4,j5,j10`
- `frame` 连续
- `four_channel/*.png` 与 CSV 帧号严格对应
- 每张图是 `480x640x4`
- 关节值在固定物理范围内

## PCA 正交分解监督

当前训练主线要求离线的 `PCA正交分解方法-16维` 监督文件。

每个任务里的 `phase_pca16_targets.npz` 包含：

- `frame_index`
- `pca_coord_tgt`
- `pca_recon_tgt`
- `residual_tgt`

全局 bank 在：

```text
data/_phase_pca16/phase_pca16_bank.npz
```

bank 中包含：

- `pca_mean`
- `pca_components`
- `pca_coord_mean/std`
- `residual_mean/std`

`data_loader.py` 会严格对齐：

- `states_filtered.csv`
- `four_channel/*.png`
- `phase_pca16_targets.npz`
