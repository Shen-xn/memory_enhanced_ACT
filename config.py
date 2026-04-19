import os
from datetime import datetime

import torch


class Config:
    """Single source of truth for training and export configuration."""

    def __init__(self):
        self._init_paths()
        self._init_run_control()
        self._init_data_and_targets()
        self._init_optimization()
        self._init_logging()
        self._init_visual_model()
        self._init_phase_pca_supervision()
        self._init_misc_model_flags()

        self.MODEL_PARAMS = {}
        self.refresh_model_params()

    def _init_paths(self):
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.LOG_ROOT = os.path.join(self.ROOT_DIR, "log")
        self.DATA_ROOT = os.path.join(self.ROOT_DIR, "data_process", "data")
        self.EXP_NAME = ""
        self.EXP_LOG_DIR = ""
        self.CKPT_DIR = ""

    def _init_run_control(self):
        self.TRAIN_MODE = ""
        self.RESUME_CKPT_PATH = ""
        self.SEED = 1
        self.USE_CUDA = torch.cuda.is_available()
        self.POLICY_CLASS = "ACTPolicy"
        self.TASK_NAME = "custom_dataset"

    def _init_data_and_targets(self):
        self.FUTURE_STEPS = 10
        self.STATE_DIM = 6

        # Current training target is step-wise delta action, divided by DELTA_QPOS_SCALE.
        self.PREDICT_DELTA_QPOS = True
        self.DELTA_QPOS_SCALE = 10.0

        # Current offline supervision path:
        #   phase_pca16_targets.npz + _phase_pca16/phase_pca16_bank.npz
        self.PHASE_TARGETS_FILENAME = "phase_pca16_targets.npz"
        self.PHASE_BANK_PATH = os.path.join(
            self.DATA_ROOT,
            "_phase_pca16",
            "phase_pca16_bank.npz",
        )

    def _init_optimization(self):
        self.NUM_EPOCHS = 40
        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 8

        self.LR = 1e-5
        self.LR_BACKBONE = 1e-6
        self.WEIGHT_DECAY = 1e-4
        self.KL_WEIGHT = 1.0

        # Total loss =
        #   RECON_LOSS_WEIGHT * recon_l1
        # + RESIDUAL_LOSS_WEIGHT * residual_l1
        # + PCA_COORD_LOSS_WEIGHT * pca_coord_mse
        # + KL_WEIGHT * kl
        self.RECON_LOSS_WEIGHT = 1.0
        self.RESIDUAL_LOSS_WEIGHT = 1.0
        self.PCA_COORD_LOSS_WEIGHT = 0.1

    def _init_logging(self):
        self.VAL_FREQ = 1
        self.SAVE_FREQ = 5
        self.LOG_PRINT_FREQ = 200
        self.SAVE_PLOT = True
        self.PRINT_LOG = True

    def _init_visual_model(self):
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
        self.CHUNK_SIZE = self.FUTURE_STEPS

    def _init_phase_pca_supervision(self):
        # Main method:
        # - phase token enters encoder only
        # - PCA head predicts 16D orthogonal coordinates
        # - residual head predicts all-joint residual actions
        self.USE_PHASE_TOKEN = True
        self.PHASE_PCA_DIM = 16

        # PCA head is intentionally deeper than the residual head.
        self.PCA_HEAD_HIDDEN_DIM = 1024
        self.PCA_HEAD_DEPTH = 3

    def _init_misc_model_flags(self):
        self.MASKS = False
        self.LR_DROP = 200
        self.CLIP_MAX_NORM = 0.1
        self.TEMPORAL_AGG = False
        self.EVAL = False
        self.ONSCREEN_RENDER = False

    def start_new_experiment(self):
        if not self.EXP_NAME:
            self.EXP_NAME = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.EXP_LOG_DIR = os.path.join(self.LOG_ROOT, self.EXP_NAME)
        self.refresh_model_params()

    def refresh_model_params(self):
        self.NUM_QUERIES = int(self.FUTURE_STEPS)
        self.CHUNK_SIZE = int(self.FUTURE_STEPS)
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
            "phase_pca_dim": self.PHASE_PCA_DIM,
            "pca_head_hidden_dim": self.PCA_HEAD_HIDDEN_DIM,
            "pca_head_depth": self.PCA_HEAD_DEPTH,
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
            "pca_coord_loss_weight": self.PCA_COORD_LOSS_WEIGHT,
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
        if "PHASE_PCA_DIM" not in ckpt_config and "phase_pca_dim" in model_params:
            self.PHASE_PCA_DIM = int(model_params["phase_pca_dim"])
        if "PCA_HEAD_HIDDEN_DIM" not in ckpt_config and "pca_head_hidden_dim" in model_params:
            self.PCA_HEAD_HIDDEN_DIM = int(model_params["pca_head_hidden_dim"])
        if "PCA_HEAD_DEPTH" not in ckpt_config and "pca_head_depth" in model_params:
            self.PCA_HEAD_DEPTH = int(model_params["pca_head_depth"])
        if "PCA_COORD_LOSS_WEIGHT" not in ckpt_config:
            if "pca_coord_loss_weight" in model_params:
                self.PCA_COORD_LOSS_WEIGHT = float(model_params["pca_coord_loss_weight"])
            elif "prototype_loss_weight" in model_params:
                # Backward-compatible read for older checkpoints.
                self.PCA_COORD_LOSS_WEIGHT = float(model_params["prototype_loss_weight"])

        self.refresh_model_params()


cfg = Config()
