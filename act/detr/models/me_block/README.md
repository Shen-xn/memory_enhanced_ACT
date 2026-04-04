# 记忆增强模块 (Memory Enhancement Block)

为ACT模型添加记忆能力的模块，用于处理遮挡场景下的机器人操作任务。

## 文件结构

```
me_block/
├── memory_gate_model.py    # 核心记忆门控模型
├── config.py               # 配置类
├── integration_example.py  # ACT集成示例
├── plan.py                 # 对比学习预训练计划
└── README.md               # 本文件
```

## 核心模型

### MemoryGateBlock
基于GRU和注意力机制的门控记忆融合模块：

**主要组件**：
1. **更新门控** (Update Gate)：控制记忆更新程度
2. **重置门控** (Reset Gate)：控制记忆重置程度
3. **跨模态注意力**：当前观测与历史记忆的交互
4. **记忆缓冲区**：存储历史信息的可学习表示

**输入输出**：
- 输入：当前特征 `[batch, input_dim]`
- 输出：融合特征 `[batch, input_dim]` + 更新记忆 + 注意力权重

### TemporalMemoryNetwork
多层时序记忆网络，处理序列数据：
- 多层MemoryGateBlock堆叠
- 支持序列输入输出
- 保持时间一致性

## 配置

使用 `MemoryGateConfig` 类进行配置：

```python
from config import MemoryGateConfig, ACT_INTEGRATION_CONFIG

# 使用默认配置
config = MemoryGateConfig()

# 使用ACT集成配置
config = ACT_INTEGRATION_CONFIG

# 自定义配置
config = MemoryGateConfig(
    input_dim=512,
    hidden_dim=256,
    memory_size=10,
    num_heads=8,
    dropout=0.1
)
```

## 集成到ACT

### 基本集成方式

```python
from memory_gate_model import TemporalMemoryNetwork
from integration_example import ACTWithMemory

# 1. 创建视觉编码器
visual_encoder = YourVisualEncoder()

# 2. 创建带记忆的ACT模型
model = ACTWithMemory(
    visual_encoder=visual_encoder,
    transformer_dim=512,
    action_dim=14,
    chunk_size=100
)

# 3. 使用模型
actions_pred, memory_state = model(images, qpos)
```

### 集成位置
记忆模块插入在：
```
图像输入 → 视觉编码器 → [记忆增强模块] → Transformer → 动作输出
```

## 训练建议

### 1. 预训练记忆模块
使用对比学习预训练记忆模块（参考 `plan.py`）：
- 正样本：时间连续的帧
- 负样本：随机其他帧
- 目标：学习时间一致性表示

### 2. 端到端训练
将记忆模块与ACT一起训练：
- 使用遮挡数据集
- 联合优化记忆和动作预测
- 监控记忆注意力可视化

### 3. 消融实验
比较：
1. 无记忆的基线ACT
2. 有记忆的ACT
3. 不同记忆配置（大小、层数等）

## 使用示例

### 基本使用
```python
import torch
from memory_gate_model import MemoryGateBlock

# 创建模型
model = MemoryGateBlock(
    input_dim=512,
    hidden_dim=256,
    memory_size=10,
    num_heads=8
)

# 处理序列
batch_size = 4
seq_len = 5
features = torch.randn(batch_size, seq_len, 512)

memories = None
for t in range(seq_len):
    current_feat = features[:, t, :]
    fused_feat, memories, attn_weights = model(current_feat, memories)
    # 使用 fused_feat 进行后续处理
```

### 集成到训练循环
```python
def train_step(model, batch):
    images = batch['images']
    qpos = batch['qpos']
    actions_gt = batch['actions']
    
    # 如果是episode开始，重置记忆
    if batch.get('is_first', False):
        model.reset_memory(images.size(0))
    
    # 前向传播
    actions_pred, memory_state = model(images, qpos, actions_gt)
    
    # 计算损失
    loss = compute_loss(actions_pred, actions_gt)
    
    return loss, memory_state
```

## 参数说明

### MemoryGateBlock 参数
- `input_dim`: 输入特征维度（匹配视觉编码器输出）
- `hidden_dim`: 记忆隐藏状态维度
- `memory_size`: 记忆缓冲区大小（存储多少"记忆片段"）
- `num_heads`: 注意力头数
- `dropout`: Dropout率

### 典型配置
| 场景 | input_dim | hidden_dim | memory_size | num_heads |
|------|-----------|------------|-------------|-----------|
| 小型模型 | 256 | 128 | 5 | 4 |
| 中型模型 | 512 | 256 | 10 | 8 |
| 大型模型 | 1024 | 512 | 20 | 16 |

## 可视化建议

### 记忆注意力可视化
```python
# 获取注意力权重
_, _, attn_weights = memory_block(current_feat, prev_memory)

# 可视化
plt.imshow(attn_weights.cpu().numpy(), cmap='hot')
plt.title('Memory Attention Weights')
plt.xlabel('Memory Slot')
plt.ylabel('Batch Item')
plt.colorbar()
```

### 记忆内容可视化
使用PCA/t-SNE将记忆向量降维，观察记忆空间的聚类情况。

## 调试技巧

1. **记忆不更新**：检查更新门的值是否接近0
2. **过拟合**：增加dropout，使用更小的记忆大小
3. **训练不稳定**：降低学习率，使用梯度裁剪
4. **内存不足**：减小batch_size或记忆大小

## 后续改进方向

1. **分层记忆**：短期记忆 + 长期记忆
2. **稀疏注意力**：减少计算复杂度
3. **可解释性**：添加记忆内容解释模块
4. **多模态记忆**：结合视觉、触觉、语言信息

## 注意事项

1. 记忆模块会增加模型参数量和计算量
2. 需要足够的遮挡数据训练
3. 记忆初始化对性能有影响
4. 注意序列长度与记忆大小的平衡

---

*创建者：昊宇的记忆增强ACT项目*
*创建时间：2026-03-31*
*用途：机器人操作任务的遮挡处理*