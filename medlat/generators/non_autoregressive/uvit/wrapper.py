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

class UViT(nn.Module):
    def __init__(
        self,
        generator: Optional[Dict[str, Any]],
        first_stage: Optional[Dict[str, Any]],
        scale_factor: float = 0.2385,
        ckpt_path: Optional[str] = None
    ):
        super().__init__()
        self.generator = generator
        self.first_stage = first_stage
        self.scale_factor = scale_factor

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


    def forward(self, *args, **kwargs):
        loss = self.generator.forward(*args, **kwargs)
        return loss
