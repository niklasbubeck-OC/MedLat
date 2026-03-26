import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from einops import rearrange
from generative.networks.nets import AutoencoderKL as AEKL
from accelerate import cpu_offload, cpu_offload_with_hook
import gc
from src.accelerator import AccelerateParent
from src.utils import instantiate_from_config, init_from_ckpt
from generative.networks.nets import AutoencoderKL as AEKL
import scipy.stats as stats


class MaskGenerator(nn.Module):
    def __init__(self, mask_ratio_min=0.05, mask_ratio_max=0.95, mask_ratio_mu=0.5, mask_ratio_std=0.25):
        super().__init__()
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)

    def forward(self, x):
        dims = x.shape
        b, c = dims[0], dims[1]
        spatial_dims = dims[2:]
        
        mask_ratio = self.mask_ratio_generator.rvs(size=1).item()
        if mask_ratio < self.mask_ratio_min:
            mask_ratio = self.mask_ratio_min
        elif mask_ratio > self.mask_ratio_max:
            mask_ratio = self.mask_ratio_max
        
        mask_orig = torch.rand((b, 1, *spatial_dims)) > mask_ratio
        mask = mask_orig.expand(b, c, *spatial_dims)
        return mask


class MDTv2(nn.Module):
    def __init__(
        self,
        generator: Optional[nn.Module],
        first_stage: Optional[nn.Module],
        scale_factor: float = 0.2385,
        ckpt_path: Optional[str] = None,
        mask_generator: Optional[nn.Module] = None
    ):
        super().__init__()
        self.generator = generator
        self.first_stage = first_stage
        self.scale_factor = scale_factor
        self.mask_generator = mask_generator
        
        if ckpt_path is not None: 
            init_from_ckpt(self, ckpt_path)
        
        self.lock_n_load()

    def lock_n_load(self):
        locks = [self.first_stage]
        for lock in locks:
            lock.eval()
            for p in lock.parameters():
                p.requires_grad = False

    def vae_encode(self, image: torch.Tensor, sample: bool=True) -> torch.Tensor:
        # return image
        # with self.vae_hook.on_gpu():
        temp = self.first_stage.encode(image).sample()
        return temp * self.scale_factor
        
    def vae_decode(self, z: torch.Tensor) -> torch.Tensor:
        # return z
        # with self.vae_hook.on_gpu():
        decoded = self.first_stage.decode(z / self.scale_factor)
        return decoded


    def forward(self, x, *args, **kwargs):
        if self.mask_generator is not None:
            mask = self.mask_generator(x).to(x.device)
            x = x * mask
            # kwargs['mask'] = mask
        loss = self.generator.forward(x, *args, **kwargs)
        return loss
