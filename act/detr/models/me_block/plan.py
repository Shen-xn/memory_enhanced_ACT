from torch import nn
import torch.functional as F
import random
import torch

class ContrastivePretraining:
    """使用对比学习预训练门控网络"""
    
    def __init__(self, gate_net):
        self.gate_net = gate_net
        
        # 投影头（用于对比学习）
        self.projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def create_contrastive_pairs(self, video_sequence):
        """创建正负样本对"""
        pairs = []
        
        for i in range(len(video_sequence) - 1):
            anchor = video_sequence[i]
            
            # 正样本：下一帧（时间连续）
            positive = video_sequence[i + 1]
            
            # 负样本：随机其他帧
            neg_idx = random.choice([j for j in range(len(video_sequence)) if j != i and j != i+1])
            negative = video_sequence[neg_idx]
            
            # 模拟遮挡：为anchor添加随机遮挡
            occluded_anchor = self._add_occlusion(anchor)
            
            pairs.append({
                'anchor': occluded_anchor,
                'positive': positive,
                'negative': negative,
                'memory': video_sequence[i-1] if i > 0 else anchor
            })
        
        return pairs
    
    def contrastive_loss(self, anchor_feat, positive_feat, negative_feat, temperature=0.1):
        """InfoNCE损失"""
        # 计算相似度
        pos_sim = F.cosine_similarity(anchor_feat, positive_feat, dim=-1)
        neg_sim = F.cosine_similarity(anchor_feat, negative_feat, dim=-1)
        
        # InfoNCE损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1) / temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def pretrain_gate_net(self, video_dataset):
        """对比学习预训练"""
        optimizer = torch.optim.Adam(self.gate_net.parameters(), lr=1e-3)
        
        for epoch in range(50):
            epoch_loss = 0
            for pair in self.create_contrastive_pairs(video_dataset):
                # 提取特征
                anchor_input = torch.cat([pair['anchor'], pair['memory']], dim=1)
                positive_input = torch.cat([pair['positive'], pair['anchor']], dim=1)
                negative_input = torch.cat([pair['negative'], pair['anchor']], dim=1)
                
                anchor_gate = self.gate_net(anchor_input)
                positive_gate = self.gate_net(positive_input)
                negative_gate = self.gate_net(negative_input)
                
                # 融合特征
                anchor_fused = anchor_gate * pair['anchor'] + (1 - anchor_gate) * pair['memory']
                positive_fused = positive_gate * pair['positive'] + (1 - positive_gate) * pair['anchor']
                negative_fused = negative_gate * pair['negative'] + (1 - negative_gate) * pair['anchor']
                
                # 投影到对比空间
                anchor_proj = self.projector(anchor_fused.mean(dim=[2, 3]))
                positive_proj = self.projector(positive_fused.mean(dim=[2, 3]))
                negative_proj = self.projector(negative_fused.mean(dim=[2, 3]))
                
                # 对比损失
                loss = self.contrastive_loss(anchor_proj, positive_proj, negative_proj)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch}, Contrastive Loss: {epoch_loss:.4f}")