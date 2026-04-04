"""
记忆门控融合模型 - 用于ACT模型的记忆增强模块
昊宇的记忆增强ACT项目 - 占位实现，后续可修改和训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def build_me_block():
    pass


class MemoryGateBlock(nn.Module):
    """
    门控记忆融合模块
    基于GRU和注意力机制，融合当前观测与历史记忆
    """
    
    def __init__(
        self,
        input_dim: int = 512,          # 输入特征维度（通常来自视觉编码器）
        hidden_dim: int = 256,         # 记忆隐藏状态维度
        memory_size: int = 10,         # 记忆缓冲区大小
        num_heads: int = 8,            # 注意力头数
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.num_heads = num_heads
        
        # 记忆更新门控（GRU风格）
        self.update_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.reset_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.candidate_transform = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 跨模态注意力（当前观测与记忆的交互）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim * 2, input_dim)
        
        # 记忆缓冲区初始化
        self.memory_buffer = None
        
        # 可学习的记忆初始化
        self.memory_init = nn.Parameter(torch.randn(1, memory_size, hidden_dim) * 0.01)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def reset_memory(self, batch_size: int):
        """重置记忆缓冲区"""
        self.memory_buffer = self.memory_init.repeat(batch_size, 1, 1)
        return self.memory_buffer
    
    def update_memory(
        self,
        current_feat: torch.Tensor,      # [batch, input_dim]
        prev_memory: Optional[torch.Tensor] = None  # [batch, memory_size, hidden_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新记忆缓冲区
        
        Args:
            current_feat: 当前时间步的特征
            prev_memory: 之前的记忆缓冲区（如果为None则使用内部缓冲区）
            
        Returns:
            updated_memory: 更新后的记忆 [batch, memory_size, hidden_dim]
            attention_weights: 注意力权重 [batch, memory_size]
        """
        batch_size = current_feat.size(0)
        
        # 如果没有提供之前的记忆，使用内部缓冲区或初始化
        if prev_memory is None:
            if self.memory_buffer is None:
                memory = self.reset_memory(batch_size)
            else:
                memory = self.memory_buffer
        else:
            memory = prev_memory
        
        # 确保记忆形状正确
        if memory.size(0) != batch_size:
            memory = memory[:batch_size]
        
        # 扩展当前特征用于与记忆交互
        current_expanded = current_feat.unsqueeze(1)  # [batch, 1, input_dim]
        
        # 计算记忆与当前特征的相似度（用于注意力）
        memory_flat = memory.view(batch_size * self.memory_size, self.hidden_dim)
        current_repeated = current_feat.unsqueeze(1).repeat(1, self.memory_size, 1)
        current_repeated = current_repeated.view(batch_size * self.memory_size, self.input_dim)
        
        # 拼接特征用于门控计算
        combined = torch.cat([current_repeated, memory_flat], dim=-1)
        
        # 计算门控信号
        update_gate = self.update_gate(combined)  # 更新门
        reset_gate = self.reset_gate(combined)    # 重置门
        
        # 计算候选记忆
        candidate_input = torch.cat([current_repeated, reset_gate * memory_flat], dim=-1)
        candidate_memory = self.candidate_transform(candidate_input)
        
        # 更新记忆：旧记忆 + 新信息
        updated_memory_flat = (1 - update_gate) * memory_flat + update_gate * candidate_memory
        updated_memory = updated_memory_flat.view(batch_size, self.memory_size, self.hidden_dim)
        
        # 跨模态注意力：当前特征与记忆的交互
        attn_output, attn_weights = self.cross_attention(
            query=current_expanded,
            key=updated_memory,
            value=updated_memory
        )
        
        # 更新内部记忆缓冲区
        self.memory_buffer = updated_memory.detach()
        
        return updated_memory, attn_weights.squeeze(1)
    
    def fuse_memory(
        self,
        current_feat: torch.Tensor,      # [batch, input_dim]
        memory: torch.Tensor,            # [batch, memory_size, hidden_dim]
        attention_weights: torch.Tensor  # [batch, memory_size]
    ) -> torch.Tensor:
        """
        融合当前特征与加权记忆
        
        Args:
            current_feat: 当前特征
            memory: 记忆缓冲区
            attention_weights: 注意力权重
            
        Returns:
            fused_feature: 融合后的特征 [batch, input_dim]
        """
        batch_size = current_feat.size(0)
        
        # 使用注意力权重加权记忆
        weighted_memory = torch.sum(
            memory * attention_weights.unsqueeze(-1),
            dim=1
        )  # [batch, hidden_dim]
        
        # 拼接当前特征与加权记忆
        combined = torch.cat([current_feat, weighted_memory], dim=-1)
        
        # 投影回原始特征维度
        fused = self.output_proj(combined)
        fused = self.norm2(fused)
        fused = self.dropout(fused)
        
        return fused
    
    def forward(
        self,
        current_feat: torch.Tensor,
        prev_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：更新记忆并融合特征
        
        Args:
            current_feat: 当前时间步的特征 [batch, input_dim]
            prev_memory: 之前的记忆（可选）
            
        Returns:
            fused_feature: 融合后的特征 [batch, input_dim]
            updated_memory: 更新后的记忆 [batch, memory_size, hidden_dim]
            attention_weights: 注意力权重 [batch, memory_size]
        """
        # 更新记忆
        updated_memory, attention_weights = self.update_memory(current_feat, prev_memory)
        
        # 融合特征
        fused_feature = self.fuse_memory(current_feat, updated_memory, attention_weights)
        
        return fused_feature, updated_memory, attention_weights


class TemporalMemoryNetwork(nn.Module):
    """
    时序记忆网络 - 封装多个时间步的记忆处理
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 多层记忆块（类似RNN的层）
        self.layers = nn.ModuleList([
            MemoryGateBlock(
                input_dim=input_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                **kwargs
            )
            for i in range(num_layers)
        ])
        
        # 层间投影（如果需要改变维度）
        self.inter_layer_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            if i < num_layers - 1 else nn.Identity()
            for i in range(num_layers)
        ])
    
    def forward(
        self,
        features: torch.Tensor,  # [batch, seq_len, input_dim]
        initial_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理整个序列
        
        Args:
            features: 输入特征序列
            initial_memory: 初始记忆（可选）
            
        Returns:
            output_features: 处理后的特征序列 [batch, seq_len, input_dim]
            final_memory: 最终记忆状态 [batch, num_layers, memory_size, hidden_dim]
        """
        batch_size, seq_len, _ = features.shape
        
        # 初始化记忆
        if initial_memory is None:
            memories = [None] * self.num_layers
        else:
            memories = list(torch.unbind(initial_memory, dim=1))
        
        # 存储每层的输出
        layer_outputs = []
        all_memories = []
        
        # 逐时间步处理
        for t in range(seq_len):
            current_feat = features[:, t, :]
            layer_input = current_feat
            
            # 逐层处理
            layer_memories = []
            for layer_idx, (layer, proj) in enumerate(zip(self.layers, self.inter_layer_proj)):
                # 处理当前层
                layer_output, updated_memory, _ = layer(layer_input, memories[layer_idx])
                
                # 更新记忆
                memories[layer_idx] = updated_memory
                layer_memories.append(updated_memory)
                
                # 准备下一层输入
                layer_input = proj(layer_output)
            
            # 收集输出
            layer_outputs.append(layer_output)
            all_memories.append(torch.stack(layer_memories, dim=1))
        
        # 堆叠输出
        output_features = torch.stack(layer_outputs, dim=1)  # [batch, seq_len, input_dim]
        final_memory = torch.stack(all_memories[-1], dim=1)  # [batch, num_layers, memory_size, hidden_dim]
        
        return output_features, final_memory


# 简单的测试代码
if __name__ == "__main__":
    print("测试记忆门控融合模型...")
    
    # 创建模型
    model = MemoryGateBlock(
        input_dim=512,
        hidden_dim=256,
        memory_size=10,
        num_heads=8
    )
    
    # 测试输入
    batch_size = 4
    seq_len = 5
    
    # 随机输入特征
    features = torch.randn(batch_size, seq_len, 512)
    
    print(f"输入形状: {features.shape}")
    
    # 逐时间步处理
    memories = None
    outputs = []
    
    for t in range(seq_len):
        current_feat = features[:, t, :]
        fused_feat, memories, attn_weights = model(current_feat, memories)
        outputs.append(fused_feat)
        
        print(f"时间步 {t}:")
        print(f"  输入特征: {current_feat.shape}")
        print(f"  融合特征: {fused_feat.shape}")
        print(f"  记忆形状: {memories.shape}")
        print(f"  注意力权重: {attn_weights.shape}")
        print()
    
    outputs = torch.stack(outputs, dim=1)
    print(f"最终输出形状: {outputs.shape}")
    print("测试完成！")