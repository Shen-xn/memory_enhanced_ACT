"""Policy wrappers around the ACT/DETR model builders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F

from act.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer


class ACTPolicy(nn.Module):
    """Main CVAE ACT policy with phase-token PCA-residual supervision."""

    def __init__(self, args_override, device=None):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override, device=device)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = float(args_override["kl_weight"])
        self.use_phase_pca_supervision = bool(args_override.get("use_phase_pca_supervision", True))
        self.pca_coord_loss_weight = float(args_override.get("pca_coord_loss_weight", 0.1))
        self.residual_loss_weight = float(args_override.get("residual_loss_weight", 1.0))
        self.recon_loss_weight = float(args_override.get("recon_loss_weight", 1.0))
        print(f"KL Weight {self.kl_weight}")

    def _normalize_tensor(self, image):
        channel_count = image.shape[-3]
        if channel_count == 4:
            if image.dim() == 4:
                image = image[:, [2, 1, 0, 3]]
            elif image.dim() == 5:
                image = image[:, :, [2, 1, 0, 3]]
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406, 0.5],
                std=[0.229, 0.224, 0.225, 0.5],
            )
        elif channel_count == 3:
            if image.dim() == 4:
                image = image[:, [2, 1, 0]]
            elif image.dim() == 5:
                image = image[:, :, [2, 1, 0]]
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        else:
            raise ValueError(f"Unsupported channel count for normalization: {channel_count}")
        return normalize(image)

    def _prepare_image_input(self, image):
        expected_cameras = len(getattr(self.model, "camera_names", []))
        if image.dim() == 4:
            if expected_cameras not in (0, 1):
                raise ValueError(
                    f"Expected {expected_cameras} camera views, but got single-image tensor {tuple(image.shape)}."
                )
            return image.unsqueeze(1)
        if image.dim() != 5:
            raise ValueError(f"Expected [B,C,H,W] or [B,N,C,H,W], got {tuple(image.shape)}.")
        if expected_cameras and image.size(1) != expected_cameras:
            raise ValueError(f"Expected {expected_cameras} camera views, got {tuple(image.shape)}.")
        return image

    def __call__(
        self,
        qpos,
        image,
        pca_coord_targets=None,
        residual_targets=None,
        actions=None,
        is_pad=None,
    ):
        env_state = None
        image = self._normalize_tensor(image)
        image = self._prepare_image_input(image)

        if actions is not None:
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]
            action_hat, is_pad_hat, (mu, logvar), aux = self.model(
                qpos, image, env_state, actions=actions, is_pad=is_pad
            )
            total_kld, _, _ = kl_divergence(mu, logvar)
            valid_mask = (~is_pad).unsqueeze(-1)

            recon_l1 = (F.l1_loss(actions, action_hat, reduction="none") * valid_mask).mean()
            if self.use_phase_pca_supervision:
                pca_coord_targets_norm = self.model.normalize_pca_coords(pca_coord_targets)
                residual_targets_norm = self.model.normalize_residual(residual_targets)

                residual_l1 = (
                    F.l1_loss(aux["residual_norm"], residual_targets_norm, reduction="none") * valid_mask
                ).mean()
                pca_coord_mse = F.mse_loss(aux["pca_coord_norm"], pca_coord_targets_norm)

                loss_dict = {
                    "recon_l1": recon_l1,
                    "residual_l1": residual_l1,
                    "pca_coord_mse": pca_coord_mse,
                    "kl": total_kld[0],
                }
                loss_dict["loss"] = (
                    loss_dict["recon_l1"] * self.recon_loss_weight
                    + loss_dict["residual_l1"] * self.residual_loss_weight
                    + loss_dict["pca_coord_mse"] * self.pca_coord_loss_weight
                    + loss_dict["kl"] * self.kl_weight
                )
            else:
                loss_dict = {
                    "recon_l1": recon_l1,
                    "kl": total_kld[0],
                }
                loss_dict["loss"] = loss_dict["recon_l1"] * self.recon_loss_weight + loss_dict["kl"] * self.kl_weight
            return loss_dict

        action_hat, _, (_, _), _ = self.model(qpos, image, env_state)
        return action_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override, device=None):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override, device=device)
        self.model = model
        self.optimizer = optimizer

    def _normalize_tensor(self, image):
        channel_count = image.shape[-3]
        if channel_count == 4:
            if image.dim() == 4:
                image = image[:, [2, 1, 0, 3]]
            elif image.dim() == 5:
                image = image[:, :, [2, 1, 0, 3]]
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406, 0.5],
                std=[0.229, 0.224, 0.225, 0.5],
            )
        elif channel_count == 3:
            if image.dim() == 4:
                image = image[:, [2, 1, 0]]
            elif image.dim() == 5:
                image = image[:, :, [2, 1, 0]]
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        else:
            raise ValueError(f"Unsupported channel count for normalization: {channel_count}")
        return normalize(image)

    def _prepare_image_input(self, image):
        expected_cameras = len(getattr(self.model, "camera_names", []))
        if image.dim() == 4:
            if expected_cameras not in (0, 1):
                raise ValueError(
                    f"Expected {expected_cameras} camera views, but got single-image tensor {tuple(image.shape)}."
                )
            return image.unsqueeze(1)
        if image.dim() != 5:
            raise ValueError(f"Expected [B,C,H,W] or [B,N,C,H,W], got {tuple(image.shape)}.")
        if expected_cameras and image.size(1) != expected_cameras:
            raise ValueError(f"Expected {expected_cameras} camera views, got {tuple(image.shape)}.")
        return image

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        image = self._normalize_tensor(image)
        image = self._prepare_image_input(image)
        if actions is not None:
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            return {"mse": mse, "loss": mse}
        return self.model(qpos, image, env_state)

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld
