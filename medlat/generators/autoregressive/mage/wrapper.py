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


class Mage(nn.Module):
    def __init__(
        self,
        generator: Optional[Dict[str, Any]],
        first_stage: Optional[Dict[str, Any]],
        ckpt_path: Optional[str] = None
    ):
        super().__init__()
        self.generator = generator
        self.first_stage = first_stage

        if ckpt_path is not None: 
            init_from_ckpt(self, ckpt_path)
        
        self.lock_n_load()

    def lock_n_load(self):
        locks = [self.first_stage]
        for lock in locks:
            lock.eval()
            for p in lock.parameters():
                p.requires_grad = False

    def vae_encode(self, x):
        with torch.no_grad():
            quant, _ , (_, _, min_encoding_indices) = self.first_stage.encode(x)
            input_tokens = min_encoding_indices 

            input_tokens = input_tokens.reshape(x.shape[0], -1)
        return input_tokens

    def vae_decode(self, x, shape):
        z_channels = self.first_stage.quantize.embedding.weight.shape[1]
        reconstructed_images = self.first_stage.decode_code(x, out_shape=shape)
        
        return reconstructed_images

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        return self.generator.sample(*args, **kwargs)

    def forward(self, x, **kwargs):
        b = x.shape[0]
        ids = self.vae_encode(x)
        loss = self.generator.forward(ids, **kwargs)
        return loss
