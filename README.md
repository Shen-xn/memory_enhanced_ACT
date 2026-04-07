# memory_enhanced_ACT

这是一个基于 ACT / DETR 改造的模仿学习项目。仓库里现在有两条已经打通的链路：

1. 主训练链路：直接训练 ACT。
2. `me_block` 离线链路：标注重要区域 -> 训练 importance segmentation -> 生成 `memory_image_four_channel` -> 再把记忆图喂给 ACT。

当前推荐把 `me_block` 理解为“离线记忆图生成子系统”，而不是“已经接入主训练循环的在线 recurrent 模块”。

`act/detr/models/me_block/README.md` 负责说明模块内部结构；这份根目录 README 负责说明整个项目怎么跑、入口脚本是什么、数据和日志怎么组织。

## 1. 快速开始

下面这些命令都在项目根目录执行：

### 检查 CUDA

```powershell
python cuda_test/torch_cuda.py
```

主训练默认依赖 GPU；如果 `torch.cuda.is_available()` 为 `False`，主训练基本跑不起来。

### 直接跑主训练

```powershell
python training.py
```

### 走完整的离线 `me_block` 流程

```powershell
python run_me_block_label_annotator.py
python run_me_block_train_importance.py
python run_me_block_generate_memory_images.py --checkpoint .\log\me_block\importance_xxx\best_model.pth
python training.py
```

根目录这三个脚本只是薄封装，默认路径如下：

- `run_me_block_label_annotator.py` -> `data_root=./data_process/data`
- `run_me_block_train_importance.py` -> `data_root=./data_process/data`, `save_root=./log/me_block`
- `run_me_block_generate_memory_images.py` -> `data_root=./data_process/data`

## 2. 目录和职责

- `training.py`
  主训练入口，负责训练、验证、checkpoint、日志和续训。
- `config.py`
  主训练配置入口。
- `utils.py`
  主训练使用的日志、配置快照、曲线绘制、checkpoint、指标记录工具。
- `data_process/`
  数据预处理、数据加载和辅助脚本。
- `cuda_test/`
  GPU / CUDA 可用性检查。
- `act/`
  改造后的 ACT / DETR 主体代码。
- `act/detr/models/me_block/`
  `me_block` 的模型、数据集、标注、训练和记忆图生成实现。
- `log/`
  主训练实验输出和 `me_block` 训练输出。

## 3. 主训练链路

当前主训练调用关系大致是：

`training.py`
-> `config.py`
-> `data_process/data_loader.py`
-> `act/policy.py`
-> `act/detr/main.py`
-> `act/detr/models/detr_vae.py`

### 3.1 数据约定

默认数据根目录：

```text
data_process/data/
```

每个任务目录应当形如：

```text
data_process/data/task_xxx/
```

主训练目前会扫描所有 `task*` 目录，并跳过名字里包含 `task_copy` 的目录。

每个任务至少需要：

- `four_channel/*.png`
- `states_filtered.csv`

可选输入：

- `memory_image_four_channel/*.png`

如果存在 `memory_image_four_channel/`，dataloader 会把它作为额外记忆图输入；如果不存在，返回全零记忆图。

当前 `states_filtered.csv` 至少要包含下面 6 个关节列：

- `j1`
- `j2`
- `j3`
- `j4`
- `j5`
- `j10`

还有几个和代码强相关的细节值得提前知道：

- 训练/验证集是“按任务划分”，不是按帧随机划分。
- 目录名里包含 `obst` 的任务会被视为障碍轨迹，验证时会单独统计一份 `val_obst`。
- 当前 dataloader 读取记忆图时，按 `000000.png`、`000001.png` 这种 6 位编号去对齐样本，所以 `memory_image_four_channel` 最好保持和原始 `four_channel` 一致的顺序编号。

### 3.2 如何开始新实验

在 [`config.py`](./config.py) 里设置好参数后运行：

```powershell
python training.py
```

如果 `TRAIN_MODE != "resume"`，会自动创建新实验目录：

```text
log/exp_YYYYMMDD_HHMMSS/
```

### 3.3 如何断点续训

在 [`config.py`](./config.py) 里设置：

```python
TRAIN_MODE = "resume"
RESUME_CKPT_PATH = "你的 checkpoint 路径"
```

然后继续运行：

```powershell
python training.py
```

当前续训逻辑会从 checkpoint 里恢复原实验配置，并继续写回原来的实验目录，而不是新开一个目录。RNG 状态也会在 checkpoint 中保存，能尽量接近中断前的训练状态。

### 3.4 `config.py` 里最常改的内容

训练相关：

- `TRAIN_MODE`
- `RESUME_CKPT_PATH`
- `NUM_EPOCHS`
- `BATCH_SIZE`
- `NUM_WORKERS`
- `FUTURE_STEPS`
- `LR`
- `LR_BACKBONE`
- `LR_ME`
- `WEIGHT_DECAY`
- `KL_WEIGHT`
- `VAL_FREQ`
- `SAVE_FREQ`
- `SEED`

模型相关：

- `POLICY_CLASS`
- `CAMERA_NAMES`
- `ME_BLOCK`
- `DEPTH_CHANNEL`
- `BACKBONE`
- `ENC_LAYERS_ENC`
- `ENC_LAYERS`
- `DEC_LAYERS`
- `HIDDEN_DIM`
- `NHEADS`
- `STATE_DIM`

当前推荐保持：

```python
ME_BLOCK = False
```

原因不是因为项目里没有 `me_block`，而是因为当前主线的推荐方案是“先离线生成记忆图，再由 dataloader 直接读入”。  
需要特别说明的是：现在 `cfg.ME_BLOCK = True` 并不会把 `importance segmentation + memory update` 这套离线模型在线插入训练；在 DETR 路径里它目前只是一个兼容占位 stub。

### 3.5 训练输出

每次主训练实验通常会在实验目录下生成：

- `train_exp_xxx.log`
- `config.json`
- `metrics.jsonl`
- `ckpt_epoch_X.pth`
- `best_model.pth`
- `training_curves.png`

其中：

- `config.json` 是当前实验配置快照。
- `metrics.jsonl` 是逐轮追加的结构化指标，包含 `train` / `val` / `val_obst`。
- `best_model.pth` 是按普通验证集 `val.loss` 选出来的当前最优模型。

## 4. `me_block` 离线工作流

推荐流程始终是下面这 4 步：

1. 标注 `importance_labels`
2. 训练 importance segmentation 模型
3. 离线生成 `memory_image_four_channel`
4. 用生成好的记忆图再次运行 `training.py`

### 4.1 标注

运行：

```powershell
python run_me_block_label_annotator.py
```

默认会读取：

```text
./data_process/data
```

并在每个任务目录下写出：

```text
task_xxx/
  importance_labels/
  importance_labels_meta.json
```

标签约定现在是“三值监督”模式：

- `1` -> target
- `2` -> goal
- `3` -> arm
- `0` -> 显式 background
- `255` -> 未标注 / 忽略

也就是说，大多数区域你可以留成未标注，不用手工刷满整张背景；但在容易和前景混淆的区域，仍然可以补少量 `background` 负样本。

标注器是 OpenCV 交互窗口，支持逐帧涂抹、清空、复制前一帧、按任务切换等操作。当前实现支持稀疏标注，不要求一个任务的全部帧都标完；只要 `four_channel` 和 `importance_labels` 里存在同名 PNG，就能进入训练集。未标注区域在训练时会被忽略，不参与 loss。

### 4.2 训练 importance 模型

运行：

```powershell
python run_me_block_train_importance.py
```

默认参数：

- `data_root = ./data_process/data`
- `save_root = ./log/me_block`

常见覆盖方式：

```powershell
python run_me_block_train_importance.py --data-root ./data_process/data --save-root ./log/me_block --epochs 30 --batch-size 4 --num-workers 4
```

该脚本也支持 `--cpu`，但实际训练速度会明显慢很多。

输出目录通常是：

```text
log/me_block/importance_YYYYMMDD_HHMMSS/
```

里面一般包含：

- `config.json`
- `metrics.jsonl`
- `latest_model.pth`
- `best_model.pth`

### 4.3 生成记忆图

训练完成后运行：

```powershell
python run_me_block_generate_memory_images.py --checkpoint .\log\me_block\importance_xxx\best_model.pth
```

默认仍然读取：

```text
./data_process/data
```

会在每个任务目录下生成：

- `memory_image_four_channel/`
- `memory_scores/`
- `memory_binary_masks/`
- `memory_image_meta.json`

如果某个任务已经完整生成过，脚本会跳过；想强制重写可以加 `--force`。这个脚本同样支持 `--cpu`。

### 4.4 再接回 ACT

记忆图生成完成后，不需要改 `training.py` 本身。只要保证：

```python
ME_BLOCK = False
```

然后直接运行：

```powershell
python training.py
```

主训练 dataloader 会自动检查每个任务下是否存在 `memory_image_four_channel/`，存在就读取，不存在就回退到全零记忆图。

## 5. 现在这个仓库的真实边界

目前已经完成并可用的部分：

- 单相机四通道输入主训练
- checkpoint / best model / 配置快照 / 结构化指标记录
- resume 到原实验目录
- 离线 `me_block` 标注、训练、记忆图生成
- 主训练自动读取离线记忆图

目前还没有真正打通的部分：

- 在线联合训练版 `me_block`
- 把 `importance segmentation + memory update` 作为主训练中的可学习 recurrent 子模块

## 6. 建议先看哪些文件

如果你是第一次接手这个仓库，建议阅读顺序：

1. [`README.md`](./README.md)
2. [`training.py`](./training.py)
3. [`config.py`](./config.py)
4. [`data_process/data_loader.py`](./data_process/data_loader.py)
5. [`act/detr/models/me_block/README.md`](./act/detr/models/me_block/README.md)

如果你接下来主要改 `me_block`，再继续看：

1. [`act/detr/models/me_block/me_block_config.py`](./act/detr/models/me_block/me_block_config.py)
2. [`act/detr/models/me_block/importance_dataset.py`](./act/detr/models/me_block/importance_dataset.py)
3. [`act/detr/models/me_block/memory_gate_model.py`](./act/detr/models/me_block/memory_gate_model.py)
4. [`act/detr/models/me_block/train_importance_model.py`](./act/detr/models/me_block/train_importance_model.py)
5. [`act/detr/models/me_block/generate_memory_images.py`](./act/detr/models/me_block/generate_memory_images.py)
