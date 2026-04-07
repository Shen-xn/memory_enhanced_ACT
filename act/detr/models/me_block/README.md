# ME Block

`me_block` 当前的真实定位不是“已经并入 ACT 主训练的在线 recurrent memory”，而是“一套离线记忆图训练与生成子系统”。

它负责的事情是：

1. 从四通道图像中识别重要区域。
2. 用显式递推规则更新历史记忆。
3. 输出可解释的 `memory_image_four_channel`。
4. 把结果留给主训练 dataloader 读取。

如果你想看怎么运行，请先看根目录 [`README.md`](../../../../README.md)。这里主要说明模块结构、数据约定和内部实现。

## 1. 当前模块边界

当前仓库里，`me_block` 有两层含义，需要分清：

1. 离线版真实模型：`importance segmentation + memory update`
2. ACT 主训练中的兼容接口：`build_me_block()` 返回的 identity stub

也就是说，真正负责生成记忆图的是本目录下的离线模型；而 `cfg.ME_BLOCK = True` 走到 DETR 主线时，并不会在线执行这套分割和递推逻辑。

## 2. 目录内文件职责

- [`me_block_config.py`](./me_block_config.py)
  统一定义 `me_block` 的配置数据结构。
- [`importance_dataset.py`](./importance_dataset.py)
  importance segmentation 训练数据集，负责扫描任务目录、按同名文件配对图像与标签、按任务划分 train / val。
- [`memory_gate_model.py`](./memory_gate_model.py)
  核心模型实现，包括分割模型、记忆更新器和兼容 stub。
- [`annotate_importance_labels.py`](./annotate_importance_labels.py)
  OpenCV 交互式标注工具。
- [`train_importance_model.py`](./train_importance_model.py)
  importance segmentation 训练入口。
- [`generate_memory_images.py`](./generate_memory_images.py)
  用训练好的模型逐任务逐帧生成记忆图。

## 3. 配置结构

[`me_block_config.py`](./me_block_config.py) 里主要有 4 组配置：

- `ImportanceModelConfig`
  分割模型相关配置，比如 `model_name`、`input_channels`、`class_names`、`class_weights`。
- `MemoryUpdateConfig`
  显式记忆递推参数，比如 `score_decay`、`tau_up`、`tau_out`、`output_dilation_radius`。
- `ImportanceTrainingConfig`
  训练数据与优化参数，比如 `data_root`、`batch_size`、`num_epochs`、`learning_rate`。
- `MemoryGenerationConfig`
  记忆图离线导出参数，比如输出目录名和 `task_filter`。

当前默认类别是：

- `target`
- `goal`
- `arm`

默认类别权重是：

- `target`: `0.6`
- `goal`: `0.3`
- `arm`: `0.1`

这些权重会在内部归一化后用于计算 importance score。

## 4. 数据约定

### 4.1 输入图像

输入来自每个任务目录下的：

```text
task_xxx/
  four_channel/
```

要求是 4 通道 PNG：

- 前 3 通道：RGB
- 第 4 通道：depth

`importance_dataset.py` 和 `generate_memory_images.py` 都会扫描所有 `task*` 目录，并跳过名称中包含 `task_copy` 的目录。

### 4.2 标签图

监督标签放在：

```text
task_xxx/
  importance_labels/
```

标签是单通道 PNG。当前推荐的稀疏标注约定为：

- `0`：explicit background
- `1`：target
- `2`：goal
- `3`：arm
- `255`：未标注 / 忽略

也就是说，新流程下你不需要把整张图的 background 手工刷满；只标前景三类即可。对于容易误检的区域，可以补少量 `0=background` 作为负样本；其余区域直接留成 `255` 未标注即可，训练时会被忽略。

如果后面要扩类，至少要同步修改：

- `ImportanceModelConfig.class_names`
- `ImportanceModelConfig.class_weights`
- 标注工具和标签图中的类别编号

### 4.3 稀疏标注支持

训练数据不是要求“整条 task 全部标完”才能训练，而是通过文件名交集配对：

- `four_channel/000123.png`
- `importance_labels/000123.png`

只要这两个同名文件同时存在，这一帧就会进入训练样本。

当前训练是“稀疏前景监督”：

- 未标注区域 `255` 不参与 loss
- 如果有显式 `0=background` 标注，也会作为有效负样本参与训练
- loss 不再被大面积背景像素直接主导

### 4.4 生成输出

离线生成阶段会在任务目录下写出：

- `memory_image_four_channel/`
- `memory_scores/`
- `memory_binary_masks/`
- `memory_image_meta.json`

其中：

- `memory_image_four_channel` 是最终给 ACT 使用的 4 通道记忆图。
- `memory_scores` 是单通道重要性分数图。
- `memory_binary_masks` 是输出门控后的二值 mask。
- `memory_image_meta.json` 记录了 checkpoint、类别和递推超参数。

## 5. 模型结构

当前核心不是端到端长时序网络，而是两部分拼起来：

1. 一个轻量 importance segmentation 网络
2. 一个显式记忆更新器

### 5.1 ImportanceSegmentationModel

位置：

- [`memory_gate_model.py`](./memory_gate_model.py)

作用：

- 输入 `[B, 4, H, W]` 的四通道图像
- 做通道归一化
- 输出背景 + 前景类别的 segmentation logits

当前默认骨干是：

- `lraspp_mobilenet_v3_large`

代码里还做了第一层卷积改造，让网络能直接接收 4 通道输入。

### 5.2 MemoryImageUpdater

这一部分不做反向传播上的时序建模，而是用显式规则逐帧更新记忆。

核心量可以这样理解：

- `class_probs`
  前景类别概率图。
- `importance_score`
  按类别权重加权得到的单通道重要性分数图。
- `score_state`
  当前记忆分数状态。
- `memory_state`
  当前记忆内容状态。
- `write_mask`
  当前帧哪些像素值得覆盖旧记忆。
- `output_mask`
  当前哪些像素允许输出到最终记忆图。

按代码逻辑，更新过程近似是：

```text
Q_t = weighted_sum(class_probs_t)
S^-_t = score_decay * S_{t-1}
W_t = Q_t > S^-_t + tau_up
M_t = where(W_t, current_image_t, M_{t-1})
S_t = where(W_t, Q_t, S^-_t)
E_t = dilate(S_t > tau_out)
memory_image_t = M_t * E_t
```

当前默认递推参数是：

- `score_decay = 0.995`
- `tau_up = 0.05`
- `tau_out = 0.20`
- `output_dilation_radius = 0`

### 5.3 ImportanceMemoryModel

这是训练和生成阶段统一使用的外层封装：

- 训练时，实际只用它的 `segmenter` 做 segmentation supervision。
- 生成时，按任务顺序把每一帧送进去，并把上一帧的 `prev_memory`、`prev_scores` 递推给下一帧。

## 6. 训练与生成阶段真实调用关系

### 6.1 标注阶段

[`annotate_importance_labels.py`](./annotate_importance_labels.py) 会：

- 扫描可用任务
- 打开 OpenCV 标注窗口
- 支持画笔、橡皮、切换任务、跳帧、复制前一帧标签
- 在任务目录写出 `importance_labels_meta.json`

### 6.2 训练阶段

[`train_importance_model.py`](./train_importance_model.py) 会：

- 用 [`ImportanceFrameDataset`](./importance_dataset.py) 读取同名图像/标签对
- 按任务划分 train / val
- 使用交叉熵损失训练 segmentation 模型
- 计算前景类别的 mean IoU
- 保存 `latest_model.pth` 和 `best_model.pth`

这里“best”是按验证集 `miou` 选择的。

### 6.3 生成阶段

[`generate_memory_images.py`](./generate_memory_images.py) 会：

- 从 checkpoint 里恢复 `MEBlockConfig`
- 逐任务读取 `four_channel/*.png`
- 按顺序递推 `prev_memory` 和 `prev_scores`
- 生成记忆图、分数图、mask 和元数据

如果某个任务已经完整生成过，并且你没有加 `--force`，脚本会直接跳过这个任务。

## 7. 与 ACT 主线的关系

当前推荐关系是：

1. `me_block` 离线生成 `memory_image_four_channel`
2. 主训练 dataloader 直接读取这些 PNG
3. ACT 主训练里保持 `cfg.ME_BLOCK = False`

这也是为什么根目录 README 会把 `me_block` 放在“离线预处理工作流”里，而不是主训练循环里。

再强调一次：`memory_gate_model.py` 里的 `build_me_block()` 现在返回的是 `MemoryImageIdentity`，它只是一个兼容占位模块，不等价于这里的离线 importance/memory 模型。

## 8. 如果你要继续改这个模块，先看哪几处

建议阅读顺序：

1. [`me_block_config.py`](./me_block_config.py)
2. [`importance_dataset.py`](./importance_dataset.py)
3. [`memory_gate_model.py`](./memory_gate_model.py)
4. [`train_importance_model.py`](./train_importance_model.py)
5. [`generate_memory_images.py`](./generate_memory_images.py)

如果你后面要继续往“在线联合训练版”推进，最需要一起对照看的文件还有：

1. [`../detr_vae.py`](../detr_vae.py)
2. [`../../main.py`](../../main.py)
3. [`../../../policy.py`](../../../policy.py)
