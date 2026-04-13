### 使用说明

数据目录放在 `data_process/data/task_*` 下。原始采集任务通常包含：

```text
task_xxx/
  rgb/
  depth/
  states.csv
```

训练 ACT 前，任务目录最终至少应包含：

```text
task_xxx/
  four_channel/
  states_filtered.csv
```

如果训练双图 ACT，还可以额外包含：

```text
task_xxx/
  memory_image_four_channel/
```

`memory_image_four_channel` 缺失时不会中断训练，dataloader 会报警并用全零 memory 图补齐。

### 推荐入口

正式训练前从项目根目录运行：

```powershell
python prepare_act_data.py
```

这是唯一推荐的数据处理入口。它会按固定顺序执行：

1. 清洗 `states.csv`，生成 `states_clean.csv`
2. 同步 `states_clean.csv`、`rgb/`、`depth/`
3. 生成 `depth_normalized/` 和 `four_channel/`
4. 将夹爪 `j10` 围绕全数据集均值做 1.2 倍动态放大，并裁剪到 `150..700`
5. 验证训练真正依赖的最终数据合同

最终验证会检查：

- `states_filtered.csv` 存在并包含 `frame,j1,j2,j3,j4,j5,j10`
- `frame` 连续为 `0..N-1`
- `four_channel/*.png` 帧号和 CSV 完全一致
- `four_channel` 图像为 `480x640x4 uint8`
- 关节值在固定物理范围内：`min=[0,100,50,50,50,150]`，`max=[1000,800,650,900,950,700]`

已有 `states_filtered.csv + four_channel/` 的生成式任务会被视为 final-only 任务，不强制要求保留原始 `rgb/depth` 中间文件，但仍会参与夹爪放大和最终验证。

夹爪放大使用 `states_filtered.pre_gripper_amp.csv` 作为备份源，避免反复运行脚本时重复叠加放大。

如果某条轨迹越出固定物理边界，预处理会打印警告并写入 `excluded_tasks.json`。训练、ME-block 数据集和遮挡任务生成都会读取这份清单并自动跳过这些任务。

### 内部实现

- `data_process_1.py`：同步轨迹和原始图像、清理坏行、平滑轨迹、过滤静止帧、重新编号。
- `data_process_2.py`：按帧号严格匹配 `rgb/` 和 `depth_normalized/`，生成 `four_channel/*.png`。
- `data_loader.py`：训练时按 `states_filtered.csv` 的 `frame` 列和 `four_channel/*.png` 文件名严格对齐，不再通过截断尾部掩盖错位问题。

这些文件只作为 `prepare_act_data.py` 的实现模块，不再单独手动运行。
