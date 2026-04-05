# memory_enhanced_ACT 功能摘要

这份文档用于快速查询当前项目各部分代码的职责、主流程、关键函数和重要变量，方便后续继续开发 `me_block`、数据处理和训练逻辑。

## 1. 当前项目主线

当前真正的训练链路是：

`training.py`
-> `config.py`
-> `data_process/data_loader.py`
-> `act/policy.py`
-> `act/detr/main.py`
-> `act/detr/models/detr_vae.py`
-> `act/detr/models/{backbone, transformer, position_encoding}.py`

其中：

- `act/` 是基于原 ACT 改出来的主体。
- `act/detr/` 是模型主体，保留了 DETR/ACT 的结构。
- `act/detr/models/me_block/` 是你后来新加的记忆增强模块区域，但目前仍以占位和草案为主，没有真正接进训练闭环。
- 根目录下的 `utils.py`、`config.py`、`training.py` 是你现在自己的训练入口和实验管理层。
- `data_process/` 负责把原始采集数据整理成训练可读格式。

## 2. 目录职责速查

### 根目录

- `training.py`
  训练入口。负责初始化模型、加载数据、训练、验证、记录日志、保存 checkpoint。

- `config.py`
  全局实验配置。定义训练参数、模型参数、日志目录、数据目录、是否启用 CUDA 等。

- `utils.py`
  训练辅助函数。负责日志器、训练曲线绘图、checkpoint 保存/恢复、指标聚合。

### `data_process/`

- `csv_process.py`
  把原始 `states.csv` 清洗成 `states_clean.csv`。

- `data_process_1.py`
  第一阶段预处理。
  作用包括深度图归一化、轨迹平滑、按运动幅度过滤重复/静止帧、重命名图片、生成 `states_filtered.csv`。

- `data_process_2.py`
  第二阶段预处理。
  负责将 `rgb` + `depth_normalized` 对齐并合成 `four_channel/*.png`。

- `data_loader.py`
  当前训练真正使用的数据集与 DataLoader 定义。
  输出训练所需的 5 元组：
  `img, curr, future, m_img, obst`

- `rollback.py`
  把处理过的数据从 `task_copy` 备份恢复回原始状态。

- `traj_data_analize.py`
  用于可视化轨迹和检查滤波/过滤效果。

- `readme.md`
  数据处理使用说明，并写了两个后续 TODO：
  1. 生成遮挡版本数据集。
  2. 用 `me_block` 生成 `memory_image_four_channel`。

### `cuda_test/`

- `torch_cuda.py`
  只做最基础的 CUDA 可用性测试。项目如果 `torch.cuda.is_available()` 为假，按当前设计基本无法正常训练。

### `act/`

- `policy.py`
  当前训练代码实际直接调用的策略包装层。
  把 DETR/ACT 模型包装成 `ACTPolicy` 或 `CNNMLPPolicy`，并在这里组织 loss。

- `utils.py`
  更偏向原 ACT 工具链，和你现在根目录训练主线不是同一套。后续如果只维护你当前训练入口，可以暂时视为“上游遗留辅助文件”。

- `imitate_episodes.py`、`visualize_episodes.py`
  更接近原 ACT 工作流，不是你现在 `training.py` 主线的核心入口。

### `act/detr/`

- `main.py`
  模型与优化器构建入口。把配置对象转换成模型构建参数，并按 `backbone` / `me_block` 分组设置学习率。

- `models/detr_vae.py`
  现在最核心的 ACT/DETR 模型定义文件。

- `models/backbone.py`
  ResNet backbone，并且已经改成支持 4 通道输入（RGB + Depth）。

- `models/transformer.py`
  Transformer encoder/decoder 实现。

- `models/position_encoding.py`
  位置编码。

### `act/detr/models/me_block/`

- `memory_gate_model.py`
  记忆模块主体草案。定义了 `MemoryGateBlock` 和 `TemporalMemoryNetwork`，但 `build_me_block()` 目前还是 `pass`。

- `config.py`
  `me_block` 的配置草案。

- `integration_example.py`
  一个“如何把记忆模块接进 ACT”的演示稿，偏说明性质，不是当前训练代码真正调用的实现。

- `plan.py` / `README.md`
  规划与说明性质文件。

## 3. 当前训练流程

### 3.1 `training.py`

当前主流程函数：

- `init_model_and_optimizer(config)`
  根据 `config.POLICY_CLASS` 创建：
  - `ACTPolicy`
  - `CNNMLPPolicy`
  并返回优化器。

- `train_one_epoch(model, train_loader, optimizer, epoch, config, logger)`
  单轮训练。
  对 `ACTPolicy` 会传入：
  - `qpos=currs`
  - `image=imgs`
  - `memory_image=m_imgs`
  - `actions=futures`
  - `is_pad`

- `validate(model, val_loader, config, logger, is_obst=False)`
  验证逻辑。
  会按 `obsts` 标签拆成：
  - 普通轨迹验证
  - 障碍轨迹验证

- `main()`
  负责：
  1. 构建 DataLoader
  2. 初始化模型与优化器
  3. 处理 resume
  4. 训练/验证循环
  5. 画曲线
  6. 保存 checkpoint

### 3.2 训练数据在代码中的含义

`data_loader.py` 的 `__getitem__` 返回：

- `img`
  当前时刻主观测图像，来自 `four_channel/*.png`

- `curr`
  当前关节状态，列为：
  `j1, j2, j3, j4, j5, j10`

- `future`
  后续 `future_steps` 个时刻的动作/关节轨迹

- `m_img`
  记忆图像，来自 `memory_image_four_channel/*.png`
  如果不存在则返回全零张量

- `obst`
  是否障碍样本
  当前判定逻辑很简单：任务目录名里包含 `obst` 就记为障碍样本

## 4. 关键配置与重要变量

### `config.py -> Config`

重要字段：

- `ROOT_DIR`
  项目根目录。

- `LOG_ROOT`
  训练日志根目录，默认为 `log/`。

- `EXP_NAME`
  当前实验名，默认自动带时间戳。

- `EXP_LOG_DIR`
  当前实验输出目录。

- `DATA_ROOT`
  数据根目录，默认指向 `data_process/data/`。

- `TRAIN_MODE`
  `"train"` 或 `"resume"`。

- `RESUME_CKPT_PATH`
  断点续训 checkpoint 路径。

- `NUM_EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`
  标准训练超参数。

- `FUTURE_STEPS`
  未来动作预测步数，也会同步传到模型的 `num_queries`。

- `LR`, `LR_BACKBONE`, `LR_ME`
  主体、backbone、记忆模块学习率。
  目前 `LR_ME = 0`，意味着即使 `me_block` 存在，也默认不训练它。

- `USE_CUDA`
  直接取 `torch.cuda.is_available()`。

- `POLICY_CLASS`
  当前默认是 `ACTPolicy`。

- `MODEL_PARAMS`
  模型的核心结构参数。
  其中最关键的有：
  - `camera_names`
  - `me_block`
  - `depth_channel`
  - `backbone`
  - `hidden_dim`
  - `nheads`
  - `num_queries`
  - `state_dim`

## 5. 数据处理链路

项目当前假定原始数据大致在：

`data_process/data/task_*`

每个任务目录原始输入至少包含：

- `rgb/`
- `depth/`
- `states.csv`

处理后会逐步得到：

1. `csv_process.py`
   `states.csv` -> `states_clean.csv`

2. `data_process_1.py`
   - 生成 `depth_normalized/`
   - 过滤轨迹
   - 对齐并清理图片
   - 输出 `states_filtered.csv`

3. `data_process_2.py`
   - 合成 `four_channel/`

4. 未来预期但尚未完成
   - 生成 `memory_image_four_channel/`

### `data_loader.py` 的关键约定

- 只读取目录名形如 `task*` 的任务目录。
- 会跳过包含 `task_copy` 的目录。
- 如果图片数和 CSV 行数不匹配，并且 `strict_alignment=True`，该任务会被整体跳过。
- 训练/验证切分是“按任务目录切分”，不是按帧随机切分。
- 关节归一化采用 min-max，统计来自训练集，再共享给验证集。

## 6. 模型相关速查

### `act/policy.py`

#### `ACTPolicy`

作用：

- 调用 `build_me_ACT_model_and_optimizer`
- 在训练时执行前向传播
- 组织 loss：
  - `l1`
  - `kl`
  - `loss = l1 + kl * kl_weight`

训练输入：

- `qpos`
- `image`
- `memory_image`
- `actions`
- `is_pad`

推理输出：

- `a_hat`
- `new_memory_image`

#### `CNNMLPPolicy`

备用策略，不是当前默认主线。
loss 使用 `mse`。

### `act/detr/main.py`

关键函数：

- `build_me_ACT_model_and_optimizer(args_override)`
  当前 `ACTPolicy` 真正调用的模型构建入口。

- `build_CNNMLP_model_and_optimizer(args_override)`
  `CNNMLPPolicy` 对应构建入口。

特殊点：

- 如果 `lr_backbone <= 0`，会冻结 backbone 参数。
- 如果 `lr_me <= 0`，会冻结 `me_block` 参数。
- 优化器参数分组已经把 `me_block` 单独拆出来了，说明结构上已经为后续接入记忆模块留好了口。

### `act/detr/models/detr_vae.py`

这是当前最重要的模型文件。

#### `DETRVAE`

核心职责：

- 用 encoder 从动作序列和当前状态里编码 latent `z`
- 用 backbone 提取图像特征
- 用 transformer 结合视觉、关节状态、latent 进行动作预测
- 输出：
  - `a_hat`
  - `is_pad_hat`
  - `[mu, logvar]`
  - `new_memory_image`

关键成员：

- `self.action_head`
  预测动作

- `self.is_pad_head`
  预测 padding

- `self.query_embed`
  查询向量，数量等于 `num_queries`

- `self.encoder_action_proj`
  将动作序列投影到 encoder 输入空间

- `self.encoder_joint_proj`
  将当前关节状态投影到 encoder 输入空间

- `self.latent_proj`
  输出 `mu` 和 `logvar`

- `self.latent_out_proj`
  将采样的 latent 投回 transformer 使用的 hidden space

#### 关于 `memory_image`

代码中已经留了接口：

- 如果 `self.me_block is not None and memory_image is not None`
  会执行：
  `new_memory_image = self.me_block(memory_image, image)`

但当前这条分支还没有真正可用，因为 `build_me_block()` 尚未实现。

### `act/detr/models/backbone.py`

当前已做的自定义改动：

- 支持 `depth_channel=True`
- 会把 ResNet 第一层卷积从 3 通道改成 4 通道
- 第 4 通道权重用 RGB 通道均值初始化

这意味着你的 `four_channel` 数据设计已经和 backbone 接上了。

### `act/detr/models/transformer.py`

职责：

- encoder 处理视觉 token + 额外输入
- decoder 使用 query 预测未来动作序列

在 4 维图像输入分支里，代码会把：

- `latent_input`
- `proprio_input`

拼到视觉 token 前面一起送进 encoder。

## 7. `me_block` 当前状态

### 已有内容

- `MemoryGateBlock`
  有较完整的草案实现：
  - update gate
  - reset gate
  - candidate memory
  - cross attention
  - memory buffer

- `TemporalMemoryNetwork`
  封装多层/多时刻记忆处理。

- `config.py`
  已定义了 `MemoryGateConfig`、`MemoryTrainingConfig` 和若干预设配置。

- `integration_example.py`
  已经表达了“把记忆模块插在视觉编码之后、Transformer 之前”的想法。

### 还没真正完成的关键点

- `build_me_block()` 目前是 `pass`
- 当前训练主线没有真实可运行的 `me_block` 构建逻辑
- `memory_image_four_channel` 生成链路也还没完成
- `LR_ME` 默认是 0，说明就算补上模块，也还需要同时调整训练配置

## 8. 日志、模型保存与实验产物

### `utils.py`

关键函数：

- `setup_logger(log_dir, exp_name)`
  同时输出控制台和文件日志。

- `plot_training_curves(train_metrics, val_metrics, val_obst_metrics, save_path)`
  画训练、普通验证、障碍验证曲线。

- `save_checkpoint(...)`
  保存：
  - epoch
  - 模型参数
  - 优化器状态
  - config
  - metrics
  - best_loss

- `load_checkpoint(...)`
  断点恢复。

- `compute_metrics(loss_dict)`
  把 loss dict 中 tensor 转成标量。

- `aggregate_metrics(metrics_list)`
  对整轮 batch 指标求均值。

### `log/`

按当前设计，每次实验都会新建一个以时间戳命名的实验目录，里面至少可能有：

- `train_*.log`
- `ckpt_epoch_*.pth`
- `best_model.pth`
- `training_curves.png`

## 9. 当前代码中的“重要约定”

后续开发时最好默认记住这些约定：

- 当前训练默认使用 6 维关节：
  `j1, j2, j3, j4, j5, j10`

- 当前视觉输入默认是 4 通道：
  `RGB + Depth`

- 当前障碍样本识别依赖目录名包含 `obst`

- 当前 `memory_image` 缺失时，数据加载器会直接返回全零图像

- 当前 train/val 切分是按任务切，不按帧切

- 当前项目对 GPU 依赖强，`USE_CUDA=False` 时不一定能顺畅跑完整训练

## 10. 后续开发建议入口

如果后面我们继续开发，最值得优先看的入口是：

1. `config.py`
   先确认实验配置和模型开关。

2. `data_process/data_loader.py`
   先确认训练到底读进来了什么张量。

3. `training.py`
   先确认当前 loss 和训练流程是怎么串的。

4. `act/policy.py`
   先确认策略层给模型喂了哪些参数。

5. `act/detr/models/detr_vae.py`
   真正要改模型结构时，从这里下手最直接。

6. `act/detr/models/me_block/memory_gate_model.py`
   真正开始做记忆模块实现时，这里是主战场。

## 11. 一句话总结

这个项目目前已经具备：

- 数据预处理链路
- 四通道视觉输入
- ACT/DETR 主训练流程
- 日志与 checkpoint 管理

但“记忆增强”部分仍停留在接口预留 + 模块草案阶段，真正可运行的关键缺口主要是：

- `build_me_block()`
- `memory_image_four_channel` 的生成流程
- `me_block` 与当前训练闭环的正式联调
