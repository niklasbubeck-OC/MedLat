import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from einops import rearrange
import gc
from src.accelerator import AccelerateParent
from src.utils import instantiate_from_config
from src.utils import init_from_ckpt

class RAR(nn.Module):
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

    def sample(self, *args, **kwargs):
        return self.generator.sample(*args, **kwargs)

    def train(self, mode: bool = True):
        # set wrapper + generator in train/eval
        super().train(mode)
        if self.generator is not None:
            self.generator.train(mode)

        # always keep first_stage frozen in eval mode
        if self.first_stage is not None:
            self.first_stage.eval()
            for p in self.first_stage.parameters():
                p.requires_grad = False

        return self

    # def vae_encode(self, image: torch.Tensor, sample: bool=True) -> torch.Tensor:
    #     # return image
    #     # with self.vae_hook.on_gpu():
    #     temp = self.first_stage.encode(image).sample()
    #     return temp * self.scale_factor
        
    # def vae_decode(self, z: torch.Tensor) -> torch.Tensor:
    #     # return z
    #     # with self.vae_hook.on_gpu():
    #     decoded = self.first_stage.decode(z / self.scale_factor)
    #     return decoded

    def vae_encode(self, image: torch.Tensor, sample: bool=True) -> torch.Tensor:
        temp = self.first_stage.tokenize(image)
        return temp

    def vae_decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.first_stage.detokenize(z)
        return decoded

    # def vae_encode(self, x):
    #     with torch.no_grad():
    #         quant, _ , (_, _, min_encoding_indices) = self.first_stage.encode(x)
    #         input_tokens = min_encoding_indices 
        
    #         input_tokens = input_tokens.reshape(x.shape[0], -1)
    #     return input_tokens

    # def vae_decode(self, x, shape):
    #     z_channels = self.first_stage.quantize.embedding.weight.shape[1]
    #     reconstructed_images = self.first_stage.decode_code(x, out_shape=shape)
    
    #     return reconstructed_images


    def forward(self, *args, **kwargs):
        loss = self.generator.forward(*args, **kwargs)
        return loss
