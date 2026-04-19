# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR-VAE backbone adapted for phase-token PCA-residual learning.

Key design:
- a phase token enters the visual transformer encoder only;
- the decoder never cross-attends to the phase token;
- the PCA branch predicts low-dimensional orthogonal coordinates;
- the residual branch predicts all-joint residual actions;
- final action = pca_recon + residual.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from act.phase_pca import PhasePCABank
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        image_channels,
        use_phase_token,
        phase_bank_path,
        predict_delta_qpos=False,
        delta_qpos_scale=10.0,
        phase_pca_dim=16,
        pca_head_hidden_dim=1024,
        pca_head_depth=3,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.image_channels = image_channels
        self.predict_delta_qpos = bool(predict_delta_qpos)
        self.delta_qpos_scale = float(delta_qpos_scale)
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model

        self.use_phase_token = bool(use_phase_token)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.residual_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)

        if not phase_bank_path:
            raise ValueError("phase_bank_path must be provided for phase-token prototype training.")
        self.phase_bank = PhasePCABank.from_npz(phase_bank_path)
        if phase_pca_dim and phase_pca_dim != self.phase_bank.pca_dim:
            raise ValueError(
                f"phase_pca_dim={phase_pca_dim} does not match bank={self.phase_bank.pca_dim}"
            )
        if self.phase_bank.action_dim != num_queries * state_dim:
            raise ValueError(
                f"phase bank action_dim={self.phase_bank.action_dim} does not match {num_queries}x{state_dim}"
            )

        self.register_buffer("pca_components", self.phase_bank.pca_components.clone())
        self.register_buffer("pca_mean", self.phase_bank.pca_mean.clone())
        self.register_buffer("pca_coord_mean", self.phase_bank.pca_coord_mean.clone())
        self.register_buffer("pca_coord_std", self.phase_bank.pca_coord_std.clone())
        self.register_buffer("residual_mean", self.phase_bank.residual_mean.clone())
        self.register_buffer("residual_std", self.phase_bank.residual_std.clone())

        pca_layers = []
        in_dim = hidden_dim
        hidden_dim_pca = int(pca_head_hidden_dim)
        depth = max(int(pca_head_depth), 1)
        for _ in range(depth - 1):
            pca_layers.extend([nn.Linear(in_dim, hidden_dim_pca), nn.ReLU(inplace=True)])
            in_dim = hidden_dim_pca
        pca_layers.append(nn.Linear(in_dim, self.phase_bank.pca_dim))
        self.pca_head = nn.Sequential(*pca_layers)
        self.phase_token_embed = nn.Embedding(1, hidden_dim)
        self.phase_pos_embed = nn.Embedding(1, hidden_dim)

        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(state_dim, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer("pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def _build_phase_token_inputs(self, batch_size, device):
        phase_token = self.phase_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1).to(device)
        phase_pos = self.phase_pos_embed.weight.unsqueeze(1).repeat(1, batch_size, 1).to(device)
        return phase_token, phase_pos

    def normalize_pca_coords(self, coords):
        return (coords - self.pca_coord_mean) / self.pca_coord_std

    def denormalize_pca_coords(self, coords_norm):
        return coords_norm * self.pca_coord_std + self.pca_coord_mean

    def normalize_residual(self, residual):
        return (residual - self.residual_mean) / self.residual_std

    def denormalize_residual(self, residual_norm):
        return residual_norm * self.residual_std + self.residual_mean

    def forward(self, qpos, image, env_state=None, actions=None, is_pad=None):
        is_training = actions is not None
        batch_size, _ = qpos.shape

        if is_training:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1).permute(1, 0, 2)
            cls_joint_is_pad = qpos.new_zeros((batch_size, 2), dtype=torch.bool)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)
            pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = qpos.new_zeros((batch_size, self.latent_dim))
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            all_cam_features = []
            all_cam_pos = []
            for cam_id, _ in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)

            proprio_input = self.input_proj_robot_state(qpos)
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)

            phase_token, phase_pos = self._build_phase_token_inputs(batch_size, src.device)
            hs, phase_memory = self.transformer(
                src,
                None,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
                encoder_prefix_tokens=phase_token if self.use_phase_token else None,
                encoder_prefix_pos_embed=phase_pos if self.use_phase_token else None,
                return_prefix_memory=self.use_phase_token,
            )
            if hs.dim() == 4:
                hs = hs[-1]
        else:
            qpos_proj = self.input_proj_robot_state(qpos)
            env_proj = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos_proj, env_proj], axis=1)
            hs = self.transformer(transformer_input, None, None, self.query_embed.weight, self.pos.weight)[0]
            if hs.dim() == 4:
                hs = hs[-1]
            phase_memory = hs.mean(dim=1, keepdim=True).transpose(0, 1)

        residual_norm = self.residual_head(hs)
        phase_feat = phase_memory[0]
        pca_coord_norm = self.pca_head(phase_feat)
        pca_coord = self.denormalize_pca_coords(pca_coord_norm)
        pca_recon_flat = torch.matmul(pca_coord, self.pca_components.T) + self.pca_mean.unsqueeze(0)
        pca_recon_action = pca_recon_flat.view(batch_size, self.num_queries, -1)
        residual = self.denormalize_residual(residual_norm)
        action_hat = pca_recon_action + residual
        is_pad_hat = self.is_pad_head(hs)

        aux = {
            "pca_coord_norm": pca_coord_norm,
            "pca_coord": pca_coord,
            "pca_recon_action": pca_recon_action,
            "residual_norm": residual_norm,
            "residual_action": residual,
        }
        return action_hat, is_pad_hat, [mu, logvar], aux


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim)
        if backbones is None:
            raise NotImplementedError
        self.backbones = nn.ModuleList(backbones)
        backbone_down_projs = []
        for backbone in backbones:
            down_proj = nn.Sequential(
                nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                nn.Conv2d(128, 64, kernel_size=5),
                nn.Conv2d(64, 32, kernel_size=5),
            )
            backbone_down_projs.append(down_proj)
        self.backbone_down_projs = nn.ModuleList(backbone_down_projs)
        mlp_in_dim = 768 * len(backbones) + 14
        self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)

    def forward(self, qpos, image, env_state, actions=None):
        batch_size, _ = qpos.shape
        all_cam_features = []
        for cam_id, _ in enumerate(self.camera_names):
            features, _ = self.backbones[cam_id](image[:, cam_id])
            features = features[0]
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        flattened_features = [cam_feature.reshape([batch_size, -1]) for cam_feature in all_cam_features]
        flattened_features = torch.cat(flattened_features, axis=1)
        features = torch.cat([flattened_features, qpos], axis=1)
        return self.mlp(features)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*mods)


def build_encoder(args):
    encoder_layer = TransformerEncoderLayer(
        args.hidden_dim,
        args.nheads,
        args.dim_feedforward,
        args.dropout,
        "relu",
        args.pre_norm,
    )
    encoder_norm = nn.LayerNorm(args.hidden_dim) if args.pre_norm else None
    return TransformerEncoder(encoder_layer, args.enc_layers_enc, encoder_norm)


def build(args):
    backbones = [build_backbone(args)]
    transformer = build_transformer(args)
    encoder = build_encoder(args)
    model = DETRVAE(
        backbones=backbones,
        transformer=transformer,
        encoder=encoder,
        state_dim=args.state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        image_channels=args.image_channels,
        use_phase_token=args.use_phase_token,
        phase_bank_path=args.phase_bank_path,
        predict_delta_qpos=getattr(args, "predict_delta_qpos", False),
        delta_qpos_scale=getattr(args, "delta_qpos_scale", 10.0),
        phase_pca_dim=getattr(args, "phase_pca_dim", 16),
        pca_head_hidden_dim=getattr(args, "pca_head_hidden_dim", 1024),
        pca_head_depth=getattr(args, "pca_head_depth", 3),
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))
    return model


def build_cnnmlp(args):
    state_dim = 14
    backbones = [build_backbone(args) for _ in args.camera_names]
    model = CNNMLP(backbones, state_dim=state_dim, camera_names=args.camera_names)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))
    return model
