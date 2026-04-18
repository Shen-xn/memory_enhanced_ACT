import os
from datetime import datetime

import torch


class Config:
    """Single source of truth for ACT training configuration.

    Top-level fields are kept explicit so checkpoint configs remain readable.
    `refresh_model_params()` mirrors the subset needed by the DETR/ACT builder.
    """

    def __init__(self):
        # ===================== 基础路径 =====================
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.LOG_ROOT = os.path.join(self.ROOT_DIR, "log")
        self.EXP_NAME = ""
        self.EXP_LOG_DIR = ""
        self.DATA_ROOT = os.path.join(self.ROOT_DIR, "data_process", "data")

        # ===================== 训练配置 =====================
        self.TRAIN_MODE = ""
        self.RESUME_CKPT_PATH = ""

        self.NUM_EPOCHS = 40
        self.BATCH_SIZE = 8
        self.NUM_WORKERS = 8
        self.FUTURE_STEPS = 10
        self.PREDICT_DELTA_QPOS = True
        self.DELTA_QPOS_SCALE = 10.0

        self.LR = 1e-5
        self.LR_BACKBONE = 1e-6
        self.WEIGHT_DECAY = 1e-4
        self.KL_WEIGHT = 5
        self.ACTION_L1_WEIGHT = 0.1

        self.VAL_FREQ = 1
        self.SAVE_FREQ = 5
        self.LOG_PRINT_FREQ = 200
        self.SAVE_PLOT = True
        self.PRINT_LOG = True

        self.SEED = 1
        self.USE_CUDA = torch.cuda.is_available()

        # ===================== 策略 / 模型选择 =====================
        self.POLICY_CLASS = "ACTPolicy"

        # ===================== 模型参数（显式顶层定义，方便引用） =====================
        self.CAMERA_NAMES = ["gemini"]
        # Current modes:
        # - IMAGE_CHANNELS=3 + USE_MEMORY_IMAGE_INPUT=False: RGB baseline
        # - IMAGE_CHANNELS=4 + USE_MEMORY_IMAGE_INPUT=False: RGBD baseline
        # - IMAGE_CHANNELS=4 + USE_MEMORY_IMAGE_INPUT=True: RGBD + memory image
        self.USE_MEMORY_IMAGE_INPUT = False
        self.IMAGE_CHANNELS = 4
        self.DEPTH_CHANNEL = True
        self.BACKBONE = "resnet18"
        self.POSITION_EMBEDDING = "sine"
        self.DILATION = False
        self.PRE_NORM = True

        self.ENC_LAYERS_ENC = 5
        self.ENC_LAYERS = 4
        self.DEC_LAYERS = 6
        self.DROPOUT = 0.1
        self.DIM_FEEDFORWARD = 2048
        self.HIDDEN_DIM = 512
        self.NHEADS = 8
        self.NUM_QUERIES = self.FUTURE_STEPS
        self.STATE_DIM = 6

        # ===================== 兼容 act/detr/main.py 参数 =====================
        # 这些参数当前训练主线未必都会直接用到，但其他模型代码里有引用或预留。
        self.MASKS = False
        self.LR_DROP = 200
        self.CLIP_MAX_NORM = 0.1
        self.CHUNK_SIZE = self.FUTURE_STEPS
        self.TEMPORAL_AGG = False
        self.EVAL = False
        self.ONSCREEN_RENDER = False
        self.CKPT_DIR = self.EXP_LOG_DIR
        self.TASK_NAME = "custom_dataset"

        self.MODEL_PARAMS = {}
        self.refresh_model_params()

    def start_new_experiment(self):
        """Create a fresh experiment name/path for a new training run."""
        self.EXP_NAME = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.EXP_LOG_DIR = os.path.join(self.LOG_ROOT, self.EXP_NAME)
        self.refresh_model_params()

    def refresh_model_params(self):
        """同步顶层模型配置到传给 ACT/DETR 的参数字典。"""
        self.NUM_QUERIES = self.FUTURE_STEPS
        self.CHUNK_SIZE = self.FUTURE_STEPS
        self.CKPT_DIR = self.EXP_LOG_DIR or ""
        self.IMAGE_CHANNELS = int(getattr(self, "IMAGE_CHANNELS", 4))
        if self.IMAGE_CHANNELS not in (3, 4):
            raise ValueError(f"IMAGE_CHANNELS must be 3 or 4, got {self.IMAGE_CHANNELS}")
        if self.USE_MEMORY_IMAGE_INPUT and self.IMAGE_CHANNELS != 4:
            raise ValueError("USE_MEMORY_IMAGE_INPUT=True currently requires IMAGE_CHANNELS=4.")
        self.DEPTH_CHANNEL = self.IMAGE_CHANNELS == 4

        self.MODEL_PARAMS = {
            "camera_names": self.CAMERA_NAMES,
            "use_memory_image_input": self.USE_MEMORY_IMAGE_INPUT,
            "image_channels": self.IMAGE_CHANNELS,
            "depth_channel": self.DEPTH_CHANNEL,
            "backbone": self.BACKBONE,
            "position_embedding": self.POSITION_EMBEDDING,
            "dilation": self.DILATION,
            "pre_norm": self.PRE_NORM,
            "enc_layers_enc": self.ENC_LAYERS_ENC,
            "enc_layers": self.ENC_LAYERS,
            "dec_layers": self.DEC_LAYERS,
            "dropout": self.DROPOUT,
            "dim_feedforward": self.DIM_FEEDFORWARD,
            "hidden_dim": self.HIDDEN_DIM,
            "nheads": self.NHEADS,
            "num_queries": self.NUM_QUERIES,
            "state_dim": self.STATE_DIM,
            "predict_delta_qpos": self.PREDICT_DELTA_QPOS,
            "delta_qpos_scale": self.DELTA_QPOS_SCALE,
            # 兼容备用/扩展路径
            "masks": self.MASKS,
            "lr_drop": self.LR_DROP,
            "clip_max_norm": self.CLIP_MAX_NORM,
            "chunk_size": self.CHUNK_SIZE,
            "temporal_agg": self.TEMPORAL_AGG,
            "eval": self.EVAL,
            "onscreen_render": self.ONSCREEN_RENDER,
            "ckpt_dir": self.CKPT_DIR,
            "policy_class": self.POLICY_CLASS,
            "task_name": self.TASK_NAME,
            "seed": self.SEED,
            "num_epochs": self.NUM_EPOCHS,
            "kl_weight": self.KL_WEIGHT,
            "action_l1_weight": self.ACTION_L1_WEIGHT,
            "batch_size": self.BATCH_SIZE,
            "epochs": self.NUM_EPOCHS,
        }
        return self.MODEL_PARAMS

    def update_from_ckpt(self, ckpt_config):
        """从 checkpoint 配置恢复可识别字段。"""
        for k, v in ckpt_config.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Older checkpoints may only know DEPTH_CHANNEL. Some intermediate
        # exports may only carry the mirrored MODEL_PARAMS value.
        model_params = ckpt_config.get("MODEL_PARAMS") or {}
        if "IMAGE_CHANNELS" not in ckpt_config:
            if "image_channels" in model_params:
                self.IMAGE_CHANNELS = int(model_params["image_channels"])
            elif "DEPTH_CHANNEL" in ckpt_config:
                self.IMAGE_CHANNELS = 4 if bool(ckpt_config["DEPTH_CHANNEL"]) else 3
            elif "depth_channel" in model_params:
                self.IMAGE_CHANNELS = 4 if bool(model_params["depth_channel"]) else 3
        self.refresh_model_params()


cfg = Config()
