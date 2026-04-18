import os
from datetime import datetime

import torch


class Config:
    """Single source of truth for training and export configuration."""

    def __init__(self):
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.LOG_ROOT = os.path.join(self.ROOT_DIR, "log")
        self.DATA_ROOT = os.path.join(self.ROOT_DIR, "data_process", "data")
        self.EXP_NAME = ""
        self.EXP_LOG_DIR = ""

        self.TRAIN_MODE = ""
        self.RESUME_CKPT_PATH = ""

        self.NUM_EPOCHS = 40
        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 8
        self.FUTURE_STEPS = 10
        self.PREDICT_DELTA_QPOS = True
        self.DELTA_QPOS_SCALE = 10.0

        self.LR = 1e-5
        self.LR_BACKBONE = 1e-6
        self.WEIGHT_DECAY = 1e-4
        self.KL_WEIGHT = 1.0

        self.PROTOTYPE_LOSS_WEIGHT = 0.1
        self.RESIDUAL_LOSS_WEIGHT = 1.0
        self.RECON_LOSS_WEIGHT = 1.0

        self.VAL_FREQ = 1
        self.SAVE_FREQ = 5
        self.LOG_PRINT_FREQ = 200
        self.SAVE_PLOT = True
        self.PRINT_LOG = True

        self.SEED = 1
        self.USE_CUDA = torch.cuda.is_available()
        self.POLICY_CLASS = "ACTPolicy"

        self.CAMERA_NAMES = ["gemini"]
        self.IMAGE_CHANNELS = 4
        self.DEPTH_CHANNEL = True
        self.BACKBONE = "resnet18"
        self.POSITION_EMBEDDING = "sine"
        self.DILATION = False
        self.PRE_NORM = True

        self.ENC_LAYERS_ENC = 3
        self.ENC_LAYERS = 5
        self.DEC_LAYERS = 5
        self.DROPOUT = 0.1
        self.DIM_FEEDFORWARD = 2048
        self.HIDDEN_DIM = 512
        self.NHEADS = 8
        self.NUM_QUERIES = self.FUTURE_STEPS
        self.STATE_DIM = 6

        self.USE_PHASE_TOKEN = True
        self.PHASE_TARGETS_FILENAME = "phase_proto_targets.npz"
        self.PHASE_BANK_PATH = os.path.join(
            self.DATA_ROOT, "_phase_proto", "phase_proto_bank.npz"
        )
        self.PHASE_NUM_PROTOTYPES = 16
        self.PHASE_PCA_DIM = 0
        self.PHASE_PCA_VAR_RATIO = 0.85

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
        if not self.EXP_NAME:
            self.EXP_NAME = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.EXP_LOG_DIR = os.path.join(self.LOG_ROOT, self.EXP_NAME)
        self.refresh_model_params()

    def refresh_model_params(self):
        self.NUM_QUERIES = self.FUTURE_STEPS
        self.CHUNK_SIZE = self.FUTURE_STEPS
        self.CKPT_DIR = self.EXP_LOG_DIR or ""
        self.IMAGE_CHANNELS = int(self.IMAGE_CHANNELS)
        if self.IMAGE_CHANNELS not in (3, 4):
            raise ValueError(f"IMAGE_CHANNELS must be 3 or 4, got {self.IMAGE_CHANNELS}")
        self.DEPTH_CHANNEL = self.IMAGE_CHANNELS == 4

        self.MODEL_PARAMS = {
            "camera_names": self.CAMERA_NAMES,
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
            "use_phase_token": self.USE_PHASE_TOKEN,
            "phase_bank_path": self.PHASE_BANK_PATH,
            "phase_num_prototypes": self.PHASE_NUM_PROTOTYPES,
            "phase_pca_dim": self.PHASE_PCA_DIM,
            "phase_pca_var_ratio": self.PHASE_PCA_VAR_RATIO,
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
            "prototype_loss_weight": self.PROTOTYPE_LOSS_WEIGHT,
            "residual_loss_weight": self.RESIDUAL_LOSS_WEIGHT,
            "recon_loss_weight": self.RECON_LOSS_WEIGHT,
            "batch_size": self.BATCH_SIZE,
            "epochs": self.NUM_EPOCHS,
        }
        return self.MODEL_PARAMS

    def update_from_ckpt(self, ckpt_config):
        for key, value in ckpt_config.items():
            if hasattr(self, key):
                setattr(self, key, value)

        model_params = ckpt_config.get("MODEL_PARAMS") or {}
        if "IMAGE_CHANNELS" not in ckpt_config and "image_channels" in model_params:
            self.IMAGE_CHANNELS = int(model_params["image_channels"])
        if "PHASE_NUM_PROTOTYPES" not in ckpt_config and "phase_num_prototypes" in model_params:
            self.PHASE_NUM_PROTOTYPES = int(model_params["phase_num_prototypes"])
        if "PHASE_PCA_DIM" not in ckpt_config and "phase_pca_dim" in model_params:
            self.PHASE_PCA_DIM = int(model_params["phase_pca_dim"])
        self.refresh_model_params()


cfg = Config()
