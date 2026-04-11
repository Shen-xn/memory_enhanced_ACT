### 使用说明

数据目录放在 `data_process/data/task_*` 下。原始采集数据通常包含：

```text
task_xxx/
  rgb/
  depth/
  states.csv
```

训练 ACT 前，任务目录至少应包含：

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

### 处理顺序

1. `csv_process.py`

把 `states.csv` 清洗成 `states_clean.csv`。输出列为：

```text
frame,j1,j2,j3,j4,j5,j10
```

2. `data_process_1.py`

先备份任务目录到 `task_copy/`，然后检查并同步 `states_clean.csv`、`rgb/`、`depth/`：

- CSV 缺某些帧时，删除对应的 `rgb/depth/depth_normalized` 图片
- 图片缺某些帧时，删除对应的 CSV 行
- 三方按 `frame` 列和文件名数字帧号取交集

随后脚本会生成 `depth_normalized/`，清理坏轨迹行，平滑轨迹，删除运动变化过小的帧，并把保留帧重新编号为连续的 `000000...`。最终输出 `states_filtered.csv`。

3. `data_process_2.py`

按帧号严格匹配 `rgb/` 和 `depth_normalized/`，将它们合成为 `four_channel/*.png`。如果两边帧号不一致，脚本会报错并要求先重新运行 `data_process_1.py`。

4. `data_loader.py`

训练时按 `states_filtered.csv` 的 `frame` 列和 `four_channel/*.png` 文件名严格对齐。不会再通过截断尾部来掩盖错位问题。

### 示例

```python
train_loader, val_loader = get_data_loaders(
    data_root="./data",
    future_steps=10,
    batch_size=8,
    num_workers=0,
)
```

训练/验证按任务 source group 划分。任一 split 为空时会直接报错。
