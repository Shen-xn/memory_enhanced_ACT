"""
记忆模块与ACT模型集成示例
昊宇 - 这是一个占位实现，展示如何将记忆模块集成到ACT中
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

# 假设这是ACT的视觉编码器（简化版）
class ACTVisualEncoder(nn.Module):
    """ACT视觉编码器（简化示例）"""
    
    def __init__(self, output_dim: int = 512):
        super().__init__()
        # 这里应该是ResNet等视觉编码器
        # 简化实现：直接投影
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(128, output_dim)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # images: [batch, num_cameras, 3, H, W]
        batch_size, num_cameras = images.shape[:2]
        
        # 处理每个相机图像
        features = []
        for cam_idx in range(num_cameras):
            img = images[:, cam_idx]  # [batch, 3, H, W]
            feat = self.conv(img)     # [batch, 128, 1, 1]
            feat = feat.view(batch_size, -1)  # [batch, 128]
            feat = self.proj(feat)    # [batch, output_dim]
            features.append(feat)
        
        # 合并多相机特征（简单平均）
        combined = torch.stack(features, dim=1).mean(dim=1)  # [batch, output_dim]
        return combined


class ACTWithMemory(nn.Module):
    """
    带记忆增强的ACT模型
    将记忆模块集成到ACT的视觉编码器之后，transformer之前
    """
    
    def __init__(
        self,
        # ACT原始参数
        visual_encoder: nn.Module,
        transformer_dim: int = 512,
        action_dim: int = 14,
        chunk_size: int = 100,
        
        # 记忆模块参数
        memory_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        # 原始ACT组件
        self.visual_encoder = visual_encoder
        self.transformer_dim = transformer_dim
        
        # 位置编码（简化）
        self.pos_embedding = nn.Parameter(
            torch.randn(1, chunk_size, transformer_dim) * 0.01
        )
        
        # Transformer编码器（简化）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )
        
        # 动作预测头
        self.action_head = nn.Linear(transformer_dim, action_dim)
        
        # 记忆模块
        from act.detr.models.me_block.memory_gate_model import TemporalMemoryNetwork
        from act.detr.models.me_block.config import ACT_INTEGRATION_CONFIG
        
        # 使用配置或默认配置
        if memory_config is not None:
            from act.detr.models.me_block.config import MemoryGateConfig
            config = MemoryGateConfig(**memory_config)
        else:
            config = ACT_INTEGRATION_CONFIG
        
        # 确保输入维度匹配
        config.input_dim = transformer_dim
        
        self.memory_network = TemporalMemoryNetwork(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            memory_size=config.memory_size,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # 记忆状态
        self.memory_state = None
        
        # 是否启用记忆
        self.use_memory = config.enabled
    
    def reset_memory(self, batch_size: int):
        """重置记忆状态"""
        self.memory_state = None
        # 重置记忆网络的内部状态
        for layer in self.memory_network.layers:
            layer.reset_memory(batch_size)
    
    def forward(
        self,
        images: torch.Tensor,          # [batch, num_cameras, 3, H, W]
        qpos: torch.Tensor,            # [batch, joint_dim]
        actions: Optional[torch.Tensor] = None,  # [batch, chunk_size, action_dim]
        is_pad: Optional[torch.Tensor] = None    # [batch, chunk_size]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        ACT前向传播（带记忆增强）
        
        Returns:
            actions_pred: 预测的动作 [batch, chunk_size, action_dim]
            memory_state: 记忆状态（用于可视化或分析）
        """
        batch_size = images.shape[0]
        
        # 1. 视觉编码
        visual_features = self.visual_encoder(images)  # [batch, transformer_dim]
        
        # 2. 记忆增强（如果启用）
        if self.use_memory:
            # 将当前特征与历史记忆融合
            # 注意：这里简化处理，实际应该处理序列
            
            # 扩展特征为序列（假设chunk_size=1的序列）
            visual_seq = visual_features.unsqueeze(1)  # [batch, 1, transformer_dim]
            
            # 通过记忆网络
            enhanced_seq, memory_state = self.memory_network(
                visual_seq, 
                self.memory_state
            )
            
            # 更新记忆状态
            self.memory_state = memory_state
            
            # 提取增强后的特征
            enhanced_features = enhanced_seq[:, -1, :]  # [batch, transformer_dim]
        else:
            enhanced_features = visual_features
        
        # 3. 与关节位置特征融合（简化）
        # 实际ACT中这里会有更复杂的融合
        joint_features = qpos  # 简化：直接使用关节位置
        
        # 合并视觉和关节特征
        combined_features = torch.cat([
            enhanced_features,
            joint_features
        ], dim=-1)
        
        # 投影到transformer维度
        combined_proj = nn.Linear(
            combined_features.shape[-1], 
            self.transformer_dim
        ).to(combined_features.device)(combined_features)
        
        # 4. 准备transformer输入
        # 扩展为序列（假设预测chunk_size个动作）
        seq_features = combined_proj.unsqueeze(1).repeat(1, self.pos_embedding.size(1), 1)
        seq_features = seq_features + self.pos_embedding
        
        # 5. Transformer编码
        transformer_output = self.transformer_encoder(seq_features)
        
        # 6. 动作预测
        actions_pred = self.action_head(transformer_output)
        
        return actions_pred, self.memory_state
    
    def select_action(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        推理时选择动作（类似ACT的select_action）
        
        Args:
            obs: 观测字典，包含images和qpos
            deterministic: 是否确定性选择
            
        Returns:
            action: 选择的动作
        """
        images = obs.get('images')
        qpos = obs.get('qpos')
        
        if images is None or qpos is None:
            raise ValueError("观测必须包含images和qpos")
        
        # 前向传播
        actions_pred, _ = self.forward(images, qpos)
        
        # 取第一个时间步的动作（ACT通常预测chunk，但执行第一个）
        action = actions_pred[:, 0, :]
        
        return action


# 训练循环示例
def train_act_with_memory(
    model: ACTWithMemory,
    dataloader,
    num_epochs: int = 100,
    device: str = "cuda"
):
    """训练带记忆的ACT模型（示例）"""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # 获取数据
            images = batch['images'].to(device)
            qpos = batch['qpos'].to(device)
            actions_gt = batch['actions'].to(device)
            
            # 重置记忆（每个episode开始）
            if batch.get('is_first', False):
                model.reset_memory(images.size(0))
            
            # 前向传播
            actions_pred, _ = model(images, qpos, actions_gt)
            
            # 计算损失
            loss = criterion(actions_pred, actions_gt)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} completed, Avg Loss: {epoch_loss/len(dataloader):.4f}")


# 使用示例
if __name__ == "__main__":
    print("ACT with Memory Integration Example")
    print("=" * 50)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建组件
    visual_encoder = ACTVisualEncoder(output_dim=512)
    
    # 创建带记忆的ACT模型
    model = ACTWithMemory(
        visual_encoder=visual_encoder,
        transformer_dim=512,
        action_dim=14,      # 7关节 * 2（位置+速度）或类似
        chunk_size=100
    )
    
    # 模型统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  记忆模块参数量: {sum(p.numel() for p in model.memory_network.parameters()):,}")
    
    # 测试前向传播
    batch_size = 2
    num_cameras = 2
    H, W = 480, 640
    
    # 创建测试数据
    test_images = torch.randn(batch_size, num_cameras, 3, H, W)
    test_qpos = torch.randn(batch_size, 7)  # 7个关节
    
    print(f"\n测试输入:")
    print(f"  Images: {test_images.shape}")
    print(f"  Qpos: {test_qpos.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        actions_pred, memory_state = model(test_images, test_qpos)
    
    print(f"\n测试输出:")
    print(f"  Actions pred: {actions_pred.shape}")
    if memory_state is not None:
        print(f"  Memory state: {memory_state.shape}")
    
    print("\n集成示例完成！")
    print("\n下一步:")
    print("1. 将这个记忆模块集成到你的ACT训练代码中")
    print("2. 调整配置以匹配你的实际模型维度")
    print("3. 在遮挡数据集上训练和测试")
    print("4. 根据结果调整记忆模块架构")