# memory_enhanced_ACT

这是一个基于 ACT / DETR 改造的模仿学习项目。

当前主线已经打通到：

- 数据预处理
- 四通道图像加载
- ACTPolicy 训练与验证
- 日志、checkpoint、最优模型保存
- 配置快照与结构化指标落盘

目前 `me_block` 相关代码仍主要处于占位 / 预留阶段，还没有真正接入训练闭环。

## 1. 当前主训练入口

当前推荐使用的训练入口是：

```bash
python training.py
```

这条主线会走：

`training.py`
-> `config.py`
-> `data_process/data_loader.py`
-> `act/policy.py`
-> `act/detr/main.py`
-> `act/detr/models/detr_vae.py`

## 2. 项目结构

- `training.py`
  当前主训练脚本。

- `config.py`
  当前训练和模型的统一配置文件。

- `utils.py`
  日志、checkpoint、曲线图、结构化指标等通用工具。

- `data_process/`
  数据预处理、数据加载相关脚本。

- `cuda_test/torch_cuda.py`
  检查当前 PyTorch / CUDA 是否可用。

- `act/`
  基于原 ACT 改出来的策略层和模型代码。

- `CODEBASE_SUMMARY.md`
  更偏开发者视角的代码摘要索引。

## 3. 训练前准备

### 3.1 检查 CUDA

先检查当前环境是否能正常用 GPU：

```bash
python cuda_test/torch_cuda.py
```

如果这里显示 `CUDA available: False`，训练大概率会很慢，或者根本不适合跑。

### 3.2 准备数据

当前训练默认从下面这个目录读取数据：

`data_process/data/`

每个任务目录通常形如：

```text
data_process/data/task_xxx/
```

训练实际依赖的是处理后的数据，至少要有：

- `four_channel/`
- `states_filtered.csv`

如果你后面要启用记忆图像，还会额外读取：

- `memory_image_four_channel/`

当前 `data_loader` 返回的样本结构是：

- `img`
  当前帧四通道图像，形状通常是 `[4, H, W]`

- `curr`
  当前关节状态，当前默认是 6 维：`j1, j2, j3, j4, j5, j10`

- `future`
  未来 `FUTURE_STEPS` 步的动作 / 轨迹

- `m_img`
  记忆图像。如果文件不存在，会自动用全零图像替代

- `obst`
  是否为障碍样本

## 4. `training.py` 有哪些功能

`training.py` 是当前最重要的运行脚本。它主要负责下面几件事。

### 4.1 `get_device(config)`

作用：

- 根据 `config.USE_CUDA` 返回当前训练设备
- 当前主线里的设备选择都统一走这里

### 4.2 `move_tensor_batch_to_device(batch_tensors, device)`

作用：

- 把一批张量统一搬到同一个 device
- 训练和验证都会用它，避免到处写重复的 `.to(device)`

### 4.3 `prepare_run_config(config)`

作用：

- 如果是新训练：
  会创建新的实验名和实验目录

- 如果是 resume：
  会先从 checkpoint 里恢复配置
  然后继续沿用原实验目录

这一层是现在实验目录管理的关键入口。

### 4.4 `init_model_and_optimizer(config)`

作用：

- 读取 `config.py` 的训练参数和模型参数
- 构建 `ACTPolicy` 或 `CNNMLPPolicy`
- 同时创建优化器

当前默认使用的是：

```python
POLICY_CLASS = "ACTPolicy"
```

### 4.5 `train_one_epoch(...)`

作用：

- 训练一个 epoch
- 执行前向、反向传播和优化器更新
- 聚合本轮训练指标
- 把训练指标同时写到：
  - 文本日志
  - `metrics.jsonl`

输出的训练指标一般包括：

- `loss`
- `l1`
- `kl`

如果改成 `CNNMLPPolicy`，则会变成：

- `loss`
- `mse`

### 4.6 `validate(...)`

作用：

- 执行验证
- 支持分别评估：
  - 普通轨迹
  - 障碍轨迹

验证结果同样会写入：

- 文本日志
- `metrics.jsonl`

其中结构化指标的 `stage` 会区分为：

- `val`
- `val_obst`

### 4.7 `main()`

作用：

- 准备实验配置
- 创建 / 恢复实验目录
- 加载数据
- 初始化模型和优化器
- 处理 resume
- 训练与验证循环
- 保存 checkpoint
- 保存 best model
- 保存曲线图和结构化指标

## 5. 如何使用训练脚本

### 5.1 开始一个新实验

在 `config.py` 里设置好参数后直接运行：

```bash
python training.py
```

新实验会自动在 `log/exp_时间戳/` 下创建目录。

### 5.2 继续训练

在 `config.py` 里修改：

```python
self.TRAIN_MODE = "resume"
self.RESUME_CKPT_PATH = r"你的checkpoint路径"
```

然后运行：

```bash
python training.py
```

现在 resume 会继续沿用 checkpoint 里的实验目录，而不是额外开一个新的目录。

## 6. `config.py` 里最常改的参数

下面这些是最常用的配置项。

### 训练相关

- `TRAIN_MODE`
  `"train"` 或 `"resume"`

- `RESUME_CKPT_PATH`
  断点续训时使用的 checkpoint 路径

- `NUM_EPOCHS`
  总训练轮数

- `BATCH_SIZE`
  batch 大小

- `NUM_WORKERS`
  DataLoader worker 数量

- `FUTURE_STEPS`
  未来动作预测步数

- `LR`
  主学习率

- `LR_BACKBONE`
  backbone 学习率

- `LR_ME`
  `me_block` 学习率

- `VAL_FREQ`
  每多少个 epoch 做一次验证

- `SAVE_FREQ`
  每多少个 epoch 存一次 checkpoint

### 模型相关

- `POLICY_CLASS`
  当前可选：
  - `ACTPolicy`
  - `CNNMLPPolicy`

- `CAMERA_NAMES`
  相机名字列表。当前单相机默认是 `["gemini"]`

- `ME_BLOCK`
  是否启用记忆模块

- `DEPTH_CHANNEL`
  是否使用四通道输入

- `BACKBONE`
  当前默认 `resnet18`

- `ENC_LAYERS`
- `DEC_LAYERS`
- `HIDDEN_DIM`
- `NHEADS`
- `STATE_DIM`

### 输出相关

- `SAVE_PLOT`
  是否保存曲线图

- `LOG_PRINT_FREQ`
  每多少个 batch 打印一次训练日志

## 7. 实验输出目录里会有什么

每次实验通常会在：

```text
log/exp_YYYYMMDD_HHMMSS/
```

下面生成这些文件：

- `train_exp_xxx.log`
  文本训练日志

- `config.json`
  当前实验的可读配置快照

- `metrics.jsonl`
  结构化指标文件，每行一条 JSON 记录，适合后续批量分析

- `ckpt_epoch_X.pth`
  普通 checkpoint

- `best_model.pth`
  当前最优模型

- `training_curves.png`
  训练 / 验证曲线图

## 8. `metrics.jsonl` 怎么看

这是后续做实验比较最方便的文件。

每一行大概长这样：

```json
{"epoch": 1, "stage": "train", "metrics": {"loss": 1.23, "kl": 0.45}, "timestamp": "2026-04-05T03:38:41"}
```

其中：

- `epoch`
  第几轮

- `stage`
  指标阶段，当前可能是：
  - `train`
  - `val`
  - `val_obst`

- `metrics`
  对应阶段的指标字典

- `timestamp`
  落盘时间

后面如果你想做：

- 筛每次实验的 best val loss
- 比较不同配置
- 统一画汇总图

优先读这个文件会比解析纯文本日志轻松很多。

## 9. 当前已知状态

- 单相机四通道输入已经兼容
- 设备管理已经从主训练链路里去掉了硬编码 `.cuda()`
- 日志、checkpoint、best model、配置快照、结构化指标已经能一起落盘
- `me_block` 仍然没有正式接入训练主线

## 10. 建议的使用顺序

1. 先运行 `python cuda_test/torch_cuda.py`
2. 确认 `config.py` 里的训练参数
3. 确认 `data_process/data/` 下数据已经处理完毕
4. 运行 `python training.py`
5. 训练过程中重点看：
   - `train_exp_xxx.log`
   - `metrics.jsonl`
   - `best_model.pth`

## 11. 开发说明

如果你后面继续开发，推荐优先查看：

- [`README.md`](/c:/Users/DELL/Desktop/me_ACT/memory_enhanced_ACT/README.md)
- [`CODEBASE_SUMMARY.md`](/c:/Users/DELL/Desktop/me_ACT/memory_enhanced_ACT/CODEBASE_SUMMARY.md)
- [`training.py`](/c:/Users/DELL/Desktop/me_ACT/memory_enhanced_ACT/training.py)
- [`config.py`](/c:/Users/DELL/Desktop/me_ACT/memory_enhanced_ACT/config.py)

如果后面正式接 `me_block`，最重要的入口会是：

- `act/detr/models/me_block/`
- `act/detr/models/detr_vae.py`
