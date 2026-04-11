"""Policy wrappers around the ACT/DETR model builders.

The wrappers normalize external image tensors before calling the model and
compute training losses. Deployment wrappers must mirror the same image and
qpos normalization rules.
"""

import torch.nn as nn
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms

from act.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
# import IPython
# e = IPython.embed

class ACTPolicy(nn.Module):
    """Main CVAE ACT policy used by this project."""

    def __init__(self, args_override, device=None):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override, device=device)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.use_memory_image_input = bool(args_override.get("use_memory_image_input", False))
        print(f'KL Weight {self.kl_weight}')

    def _normalize_tensor(self, image):
        """Convert OpenCV BGR/BGRA tensors to RGB/RGBD and apply ImageNet stats."""
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
        """
        Accept either:
        - [B, C, H, W] for a single camera
        - [B, N, C, H, W] for multi-camera input
        and normalize to the 5D format expected by the DETR backbone path.
        """
        expected_cameras = len(getattr(self.model, "camera_names", []))

        if image.dim() == 4:
            if expected_cameras not in (0, 1):
                raise ValueError(
                    f"Expected {expected_cameras} camera views, but got a single-image tensor "
                    f"with shape {tuple(image.shape)}."
                )
            return image.unsqueeze(1)

        if image.dim() != 5:
            raise ValueError(
                f"Expected image shape [B, C, H, W] or [B, N, C, H, W], got {tuple(image.shape)}."
            )

        if expected_cameras and image.size(1) != expected_cameras:
            raise ValueError(
                f"Expected {expected_cameras} camera views, but got tensor shape {tuple(image.shape)}."
            )

        return image

    def __call__(self, qpos, image, memory_image=None, actions=None, is_pad=None):
        """Return a loss dict during training or action chunks during inference."""
        env_state = None
        image = self._normalize_tensor(image)
        image = self._prepare_image_input(image)
        
        if self.use_memory_image_input and memory_image is not None:
            memory_image = self._normalize_tensor(memory_image)
        else:
            memory_image = None
        
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, memory_image, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(
                qpos,
                image,
                env_state,
                memory_image=memory_image,
            )
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    """Legacy policy kept only for old ACT compatibility paths."""

    def __init__(self, args_override, device=None):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override, device=device)
        self.model = model # decoder
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
                    f"Expected {expected_cameras} camera views, but got a single-image tensor "
                    f"with shape {tuple(image.shape)}."
                )
            return image.unsqueeze(1)

        if image.dim() != 5:
            raise ValueError(
                f"Expected image shape [B, C, H, W] or [B, N, C, H, W], got {tuple(image.shape)}."
            )

        if expected_cameras and image.size(1) != expected_cameras:
            raise ValueError(
                f"Expected {expected_cameras} camera views, but got tensor shape {tuple(image.shape)}."
            )

        return image

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        image = self._normalize_tensor(image)
        image = self._prepare_image_input(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

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
