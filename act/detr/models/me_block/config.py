"""
记忆增强模块配置
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryGateConfig:
    """记忆门控配置"""
    
    # 基础配置
    enabled: bool = True
    input_dim: int = 512           # 输入特征维度（与视觉编码器输出匹配）
    hidden_dim: int = 256          # 记忆隐藏状态维度
    memory_size: int = 10          # 记忆缓冲区大小
    num_heads: int = 8             # 注意力头数
    dropout: float = 0.1           # Dropout率
    
    # 时序网络配置
    num_layers: int = 2            # 记忆层数
    use_temporal: bool = True      # 是否使用时序记忆网络
    
    # 训练配置
    pretrain_epochs: int = 50      # 预训练轮数
    learning_rate: float = 1e-3    # 学习率
    weight_decay: float = 1e-4     # 权重衰减
    
    # 融合策略
    fusion_method: str = "gate"    # 融合方法：gate, attention, residual
    residual_weight: float = 0.5   # 残差连接权重（如果使用residual）
    
    # 初始化配置
    init_method: str = "normal"    # 初始化方法：normal, xavier, kaiming
    init_std: float = 0.02         # 初始化标准差
    
    def __post_init__(self):
        """配置验证"""
        assert self.fusion_method in ["gate", "attention", "residual"], \
            f"不支持的融合方法: {self.fusion_method}"
        assert self.init_method in ["normal", "xavier", "kaiming"], \
            f"不支持的初始化方法: {self.init_method}"
        
        # 确保隐藏维度能被注意力头数整除
        assert self.hidden_dim % self.num_heads == 0, \
            f"隐藏维度({self.hidden_dim})必须能被注意力头数({self.num_heads})整除"


@dataclass
class MemoryTrainingConfig:
    """记忆训练配置"""
    
    # 数据集配置
    dataset_path: Optional[str] = None      # 数据集路径
    sequence_length: int = 100              # 序列长度
    batch_size: int = 32                    # 批次大小
    
    # 训练配置
    num_epochs: int = 100                   # 训练轮数
    warmup_steps: int = 1000                # 预热步数
    gradient_clip: float = 1.0              # 梯度裁剪
    
    # 损失配置
    use_contrastive_loss: bool = True       # 是否使用对比损失
    contrastive_weight: float = 0.1         # 对比损失权重
    reconstruction_weight: float = 1.0      # 重建损失权重
    
    # 日志配置
    log_interval: int = 10                  # 日志间隔
    save_interval: int = 1000               # 保存间隔
    eval_interval: int = 500                # 评估间隔
    
    # 设备配置
    device: str = "cuda"                    # 设备
    num_workers: int = 4                    # 数据加载工作线程数


# 预定义配置
DEFAULT_CONFIG = MemoryGateConfig()
SMALL_CONFIG = MemoryGateConfig(
    hidden_dim=128,
    memory_size=5,
    num_heads=4,
    num_layers=1
)

LARGE_CONFIG = MemoryGateConfig(
    hidden_dim=512,
    memory_size=20,
    num_heads=16,
    num_layers=3,
    dropout=0.2
)

# ACT集成配置
ACT_INTEGRATION_CONFIG = MemoryGateConfig(
    input_dim=512,      # 匹配ACT的视觉编码器输出
    hidden_dim=256,
    memory_size=10,
    num_heads=8,
    fusion_method="gate"
)


def load_config(config_path: Optional[str] = None) -> MemoryGateConfig:
    """
    从文件加载配置（如果提供），否则返回默认配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置对象
    """
    if config_path is None:
        return DEFAULT_CONFIG
    
    # 这里可以添加从JSON/YAML文件加载配置的逻辑
    # 目前返回默认配置
    return DEFAULT_CONFIG


def save_config(config: MemoryGateConfig, config_path: str):
    """
    保存配置到文件
    
    Args:
        config: 配置对象
        config_path: 保存路径
    """
    import json
    
    config_dict = {
        k: v for k, v in config.__dict__.items() 
        if not k.startswith('_')
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


if __name__ == "__main__":
    # 测试配置
    config = DEFAULT_CONFIG
    print("默认配置:")
    print(config)
    
    # 测试ACT集成配置
    act_config = ACT_INTEGRATION_CONFIG
    print("\nACT集成配置:")
    print(act_config)