# ME Block

`me_block` 当前的定位是离线记忆图子系统，不是已经并入 ACT 主训练循环的在线 recurrent 模块。

它负责：

1. 标注重要区域
2. 训练 importance segmentation
3. 逐帧递推生成 `memory_image_four_channel`
4. 把结果留给 ACT dataloader 读取

根目录整体流程见 [`README.md`](../../../../README.md)。

## 文件职责

- [`me_block_config.py`](./me_block_config.py)：模块配置
- [`importance_dataset.py`](./importance_dataset.py)：训练数据集
- [`annotate_importance_labels.py`](./annotate_importance_labels.py)：标注器
- [`train_importance_model.py`](./train_importance_model.py)：训练入口
- [`generate_memory_images.py`](./generate_memory_images.py)：记忆图生成
- [`memory_gate_model.py`](./memory_gate_model.py)：分割模型和记忆更新逻辑

## 输入和标签

### 标注/训练输入

当前标注和 importance 训练默认从任务目录下的 `rgb/` 读取图像。

### 记忆图生成输入

生成阶段读取任务目录下的 `four_channel/*.png`，输出：

```text
task_xxx/
  memory_image_four_channel/
```

### 标签语义

- `0`：background
- `1`：target
- `2`：goal
- `3`：arm
- `255`：未标注 / ignore

推荐做法是：

- 主要精力标 `target / goal / arm`
- 对容易误检的位置补少量 `background`
- 其他大片区域留成 `255`

训练时 `255` 不参与 loss。

## 训练逻辑

当前 importance 模型是 4 类分割：

- background
- target
- goal
- arm

训练使用交叉熵，`ignore_index = 255`。

训练入口：

```powershell
python run_me_block_train_importance.py
```

常用参数：

```powershell
python run_me_block_train_importance.py --epochs 30 --batch-size 4 --gamma-min 0.6 --gamma-max 1.8 --noise-std 0.02
```

只在 train split 上启用的数据增强：

- 随机 gamma：`0.6 ~ 1.8`
- 高斯噪声：默认 `std = 0.02`

## 正式训练机上的建议流程

1. 先准备 `rgb/`、`four_channel/` 和 `states_filtered.csv`
2. 标注一批 `importance_labels`
3. 先小规模训练确认 loss / mIoU 能下降
4. 再扩充标注，重新训练正式 `me_block`
5. 训练结束后至少留好：
   - `best_model.pth`
   - `config.json`
   - `metrics.jsonl`
6. 如果后面要部署 me_act，再把这个 `best_model.pth` 和双图 ACT 一起导出

## 生成逻辑

生成入口：

```powershell
python run_me_block_generate_memory_images.py --checkpoint .\log\me_block\importance_xxx\best_model.pth
```

默认只保存：

- `memory_image_four_channel/*.png`

如果加 `--debug`，还会额外保存：

- `memory_scores/`
- `memory_binary_masks/`
- `importance_scores/`
- `write_masks/`
- `memory_image_meta.json`

如果使用 `--force` 且这次不是 `--debug`，脚本会先清掉旧的 debug 产物，避免目录内容不一致。

## 当前 gate 逻辑

当前分割输出 4 类 softmax 概率：background + target/goal/arm。

代码里实际是按三个前景类分别递推，不是先把三类加权合成一个总分。每个前景类的候选分数为：

```text
candidate_score[class] = p(class) * (1 - p(background))
```

然后每一类各自做递推：

```text
decayed_score[class] = score_decay * prev_score[class]
write_mask[class] = candidate_score[class] > decayed_score[class] + tau_up
```

`memory_state` 是覆盖式更新，不是平均融合。

最终输出 mask 不是固定阈值，而是按类分别保留 `score_state` 最高的前 `keep_top_ratio_target/goal/arm` 像素。
也就是说，当前 memory state 有三套前景记忆：target、goal、arm。

## 和 ACT 的关系

推荐用法始终是：

1. 离线生成 `memory_image_four_channel`
2. 在 ACT 主训练里打开 `USE_MEMORY_IMAGE_INPUT = True`

也就是说，ACT 只把记忆图当第二张输入图使用，不在线执行这里的 segmentation/gate。
如果某些训练样本找不到对应 `memory_image_four_channel`，ACT dataloader 会报警并用全零 memory 图补齐。

## 真机部署时的关系

当前仓库支持两种使用方式：

1. 离线生成记忆图，再训练双图 ACT
2. 部署时把训练好的 `me_block` 一起导出，在线生成 `memory_image`

如果你后面部署的是在线 me_act，那么需要：

1. 一个训练好的 `me_block` checkpoint
2. 一个训练好的双图 ACT checkpoint
3. 用这两个 checkpoint 一起导出 `deploy_artifacts_memory`
