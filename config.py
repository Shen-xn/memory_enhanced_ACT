import os
from datetime import datetime
import torch

# ===================== 基础配置 =====================
class Config:
    def __init__(self):
        # 项目根目录
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        # 实验名称（自动生成，包含时间戳）
        self.EXP_NAME = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # 日志/模型保存根目录
        self.LOG_ROOT = os.path.join(self.ROOT_DIR, "log")
        # 当前实验日志目录
        self.EXP_LOG_DIR = os.path.join(self.LOG_ROOT, self.EXP_NAME)
        # 数据根目录（指向data_process下的data）
        self.DATA_ROOT = os.path.join(self.ROOT_DIR, "data_process", "data")
        
        # ===================== 训练配置 =====================
        # 训练模式：train(从头训练)/resume(断点续训)
        self.TRAIN_MODE = "train"
        # 断点续训时加载的模型路径
        self.RESUME_CKPT_PATH = ""
        # 总训练轮数
        self.NUM_EPOCHS = 20
        # 批次大小
        self.BATCH_SIZE = 8
        # 数据加载线程数
        self.NUM_WORKERS = 8
        # 未来动作预测步数（对应data_loader的future_steps）
        self.FUTURE_STEPS = 10
        # 学习率
        self.LR = 1e-4
        # Backbone学习率
        self.LR_BACKBONE = 1e-5
        # 记忆增强模块学习率
        self.LR_ME = 0
        # 权重衰减
        self.WEIGHT_DECAY = 1e-4
        # KL散度权重（仅ACTPolicy生效）
        self.KL_WEIGHT = 1.0
        # 验证频率（每多少轮验证一次）
        self.VAL_FREQ = 1
        # 模型保存频率（每多少轮保存一次ckpt）
        self.SAVE_FREQ = 5
        # 是否使用GPU
        self.USE_CUDA = torch.cuda.is_available()
        # 随机种子
        self.SEED = 42
        
        # ===================== 模型配置 =====================
        # 策略类型：ACTPolicy/CNNMLPPolicy
        self.POLICY_CLASS = "ACTPolicy"
        # DETR模型参数（对应main.py中的参数）
        self.MODEL_PARAMS = {
            "camera_names": ["gemini"],        
            "me_block": False,               # 是否使用记忆增强模块
            "depth_channel": True,          # 是否使用4通道输入（RGB+Depth）
            "backbone": "resnet18",         # 骨干网络类型
            "position_embedding": "sine",   # 位置embedding计算方式
            "dilation": False,               # If true, we replace stride with dilation in the last convolutional block (DC5)
            "pre_norm": True,               # use LN before or after Multiheadatten and FNN
            "enc_layers_enc": 4,                # Transformer编码层数量
            "enc_layers": 4,                # Transformer编码层数量
            "dec_layers": 6,                # Transformer解码层数量
            "dropout": 0.1,
            "dim_feedforward": 2048,        # Transformer前馈层维度
            "hidden_dim": 512,              # Transformer嵌入维度
            "nheads": 8,                    # 注意力头数
            "num_queries": self.FUTURE_STEPS,             # 查询槽数量
            "state_dim": 6,                 # 关节状态维度（j1-j5+j10）
        }
        
        # ===================== 可视化配置 =====================
        # 是否保存训练曲线
        self.SAVE_PLOT = True
        # 是否实时打印训练日志
        self.PRINT_LOG = True
        # 日志打印频率（每多少个batch打印一次）
        self.LOG_PRINT_FREQ = 10

    def update_from_ckpt(self, ckpt_config):
        """从断点续训的ckpt中更新配置"""
        for k, v in ckpt_config.items():
            if hasattr(self, k):
                setattr(self, k, v)
        # 重置实验目录（避免覆盖原有日志）
        self.EXP_NAME = f"{self.EXP_NAME}_resume_{datetime.now().strftime('%H%M%S')}"
        self.EXP_LOG_DIR = os.path.join(self.LOG_ROOT, self.EXP_NAME)

# 初始化配置实例
cfg = Config()

# 确保日志目录存在
os.makedirs(cfg.EXP_LOG_DIR, exist_ok=True)
