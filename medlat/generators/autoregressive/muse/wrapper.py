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
from .ldm_vq import VQModel


class MaskGIT(nn.Module):
    def __init__(
        self,
        maskgit_cfg: Optional[Dict[str, Any]],
        vae_cfg: Optional[Dict[str, Any]],
        ckpt_path: Optional[str] = None
    ):
        super().__init__()
        self.maskgit = instantiate_from_config(maskgit_cfg)
        self.vae = instantiate_from_config(vae_cfg)

        if ckpt_path is not None: 
            init_from_ckpt(self, ckpt_path)
        
        self.lock_n_load()

    def lock_n_load(self):
        locks = [self.vae]
        for lock in locks:
            lock.eval()
            for p in lock.parameters():
                p.requires_grad = False

    def vae_encode(self, x):
        with torch.no_grad():
            if isinstance(self.vae, VQModel):
                quant, _ , (_, _, min_encoding_indices) = self.vae.encode(x)
                input_tokens = min_encoding_indices 
            elif isinstance(self.vae, ConvVQModel):
                _, encoder_dict = self.vae.encode(x)
                input_tokens = encoder_dict["min_encoding_indices"]
            else:
                raise NotImplementedError(f"Encoding not implemented for {self.vae.__class__.__name__}")

            input_tokens = input_tokens.reshape(x.shape[0], -1)
        return input_tokens

    def vae_decode(self, x, shape):
        if isinstance(self.vae, VQModel):
            z_channels = self.vae.quantize.embedding.weight.shape[1]
            reconstructed_images = self.vae.decode_code(x, out_shape=shape)
        elif isinstance(self.vae, ConvVQModel):
            reconstructed_images = self.vae.decode_tokens(x)
        else:
            raise NotImplementedError(f"Decoding not implemented for {self.vae.__class__.__name__}")
        return reconstructed_images

    @torch.no_grad()
    def sample(self,*args, **kwargs):
        return self.maskgit.generate(*args, **kwargs)

    def forward(self, x, **kwargs):
        b = x.shape[0]
        ids = self.vae_encode(x)
        loss = self.maskgit.forward(ids, **kwargs)
        return loss
