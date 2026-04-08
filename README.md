# memory_enhanced_ACT

这个仓库现在按两条主线来理解最清楚：

1. ACT 主训练
   - `USE_MEMORY_IMAGE_INPUT = False`：baseline，单张四通道图输入
   - `USE_MEMORY_IMAGE_INPUT = True`：双图 ACT，输入 `image + memory_image`
2. `me_block` 离线 pipeline
   - 标注重要区域
   - 训练 importance segmentation
   - 生成 `memory_image_four_channel`
   - 再把记忆图喂给双图 ACT

当前不做在线联合训练。`me_block` 是离线子系统，ACT 只负责吃单图或双图输入。

## 快速开始

检查 CUDA：

```powershell
python cuda_test/torch_cuda.py
```

训练 baseline ACT：

```powershell
python training.py
```

跑完整离线记忆图流程：

```powershell
python run_me_block_label_annotator.py
python run_me_block_train_importance.py
python run_me_block_generate_memory_images.py --checkpoint .\log\me_block\importance_xxx\best_model.pth
python training.py
```

## 目录

- `training.py`：ACT 训练入口
- `config.py`：主训练配置
- `data_process/`：数据预处理和 dataloader
- `act/`：ACT / DETR 主体
- `act/detr/models/me_block/`：`me_block` 离线实现
- `deploy/`：TorchScript 导出和 Jetson / ROS2 部署代码
- `log/`：训练日志和模型

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
```

如果训练双图 ACT，还会额外读取：

```text
task_xxx/
  memory_image_four_channel/
```

`states_filtered.csv` 至少包含这 6 列：

- `j1`
- `j2`
- `j3`
- `j4`
- `j5`
- `j10`

训练/验证按任务目录划分，不是按帧随机划分。

## 当前统一口径

### 图像通道

- 四通道图在磁盘和部署侧都按 OpenCV 语义处理，也就是 `BGRA`
- ACT 训练 dataloader 现在也统一按 `BGRA` 读图
- ACT 在归一化前会把颜色通道重排成 `RGB` 语义后再套 ImageNet mean/std
- `me_block` 训练、生成和部署都继续按 OpenCV 口径处理 `BGR/BGRA`

换句话说：训练、导出、部署现在都以 `BGRA` 作为外部接口，不再混用 PIL 的 `RGBA` 读图口径。

### 关节归一化

ACT 训练、导出、部署统一使用固定物理范围，不再按数据集统计：

```text
joint_min = [0, 0, 0, 0, 0, 100]
joint_max = [1000, 1000, 1000, 1000, 1000, 700]
joint_rng = [1000, 1000, 1000, 1000, 1000, 600]
```

## ACT 训练

主训练开关在 [`config.py`](./config.py)：

- `USE_MEMORY_IMAGE_INPUT = False`：baseline 单图 ACT
- `USE_MEMORY_IMAGE_INPUT = True`：双图 ACT

运行：

```powershell
python training.py
```

新实验默认写到：

```text
log/exp_YYYYMMDD_HHMMSS/
```

常见输出：

- `config.json`
- `metrics.jsonl`
- `ckpt_epoch_X.pth`
- `best_model.pth`
- `training_curves.png`

断点续训：

```python
TRAIN_MODE = "resume"
RESUME_CKPT_PATH = "你的 checkpoint 路径"
```

然后再运行 `python training.py`。

## `me_block` 离线流程

详细说明看 [`act/detr/models/me_block/README.md`](./act/detr/models/me_block/README.md)。

这里记住结论就够：

1. `run_me_block_label_annotator.py`
2. `run_me_block_train_importance.py`
3. `run_me_block_generate_memory_images.py --checkpoint ...`
4. 打开 `USE_MEMORY_IMAGE_INPUT = True` 再训练 ACT

## 部署

部署说明看：

- [`deploy/README.md`](./deploy/README.md)
- [`deploy/me_act_inference/README.md`](./deploy/me_act_inference/README.md)

当前已经打通的是 baseline ACT 部署链：

```text
rgb + depth + qpos -> preprocess(BGRA) -> ACT -> action sequence
```

ROS2 节点支持：

- 初始化
- 开始
- 停止
- 急停

当前不做时间聚合，先执行 `action_seq[0]`。
