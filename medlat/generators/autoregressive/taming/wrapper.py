import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from einops import rearrange
from generative.networks.nets import AutoencoderKL as AEKL
from accelerate import cpu_offload, cpu_offload_with_hook
import gc
from src.accelerator import AccelerateParent
from src.utils import instantiate_from_config, init_from_ckpt
from generative.networks.nets import AutoencoderKL as AEKL

class CoordStage(nn.Module):
    def __init__(self, n_embed, down_factor):
        self.n_embed = n_embed
        self.down_factor = down_factor

    def eval(self):
        return self

    def encode(self, c):
        """fake vqmodel interface"""
        assert 0.0 <= c.min() and c.max() <= 1.0
        b,ch,h,w = c.shape
        assert ch == 1

        c = torch.nn.functional.interpolate(c, scale_factor=1/self.down_factor,
                                            mode="area")
        c = c.clamp(0.0, 1.0)
        c = self.n_embed*c
        c_quant = c.round()
        c_ind = c_quant.to(dtype=torch.long)

        info = None, None, c_ind
        return c_quant, None, info

    def decode(self, c):
        c = c/self.n_embed
        c = torch.nn.functional.interpolate(c, scale_factor=self.down_factor,
                                            mode="nearest")
        return c

class SOSProvider(nn.Module):
    # for unconditional training
    def __init__(self, sos_token, quantize_interface=True):
        super().__init__()
        self.sos_token = sos_token
        self.quantize_interface = quantize_interface

    def encode(self, x):
        # get batch size from data and replicate sos_token
        c = torch.ones(x.shape[0], 1)*self.sos_token
        c = c.long().to(x.device)
        if self.quantize_interface:
            return c, None, [None, None, c]
        return c

class Labelator(nn.Module):
    """Net2Net Interface for Class-Conditional Model"""
    def __init__(self, n_classes, quantize_interface=True):
        super().__init__()
        self.n_classes = n_classes
        self.quantize_interface = quantize_interface

    def encode(self, c):
        c = c[:,None]
        if self.quantize_interface:
            return c, None, [None, None, c.long()]
        return c

class Taming(nn.Module):
    def __init__(
        self,
        generator: Optional[Dict[str, Any]],
        first_stage: Optional[Dict[str, Any]],
        cond_stage: Optional[Dict[str, Any]],
        # permuter_config=None,
        downsample_cond_size=-1,
        pkeep=1.0,
        sos_token=0,
        unconditional=False,
        ckpt_path: Optional[str] = None
    ):
        super().__init__()
        self.sos_token = sos_token
        self.pkeep = pkeep
        self.downsample_cond_size=-1
        self.unconditional = unconditional
        
        self.generator = generator
        self.first_stage = first_stage
        self.cond_stage = cond_stage
        
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep


        if ckpt_path is not None: 
            init_from_ckpt(self, ckpt_path)
        
        self.lock_n_load()

    def lock_n_load(self):
        locks = [self.first_stage]
        for lock in locks:
            lock.eval()
            for p in lock.parameters():
                p.requires_grad = False

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(config)
            model = model.eval()
            cond_stage_model = model
        return cond_stage_model


    
    @torch.no_grad()
    def vae_encode(self, x):
        quant_z, _, info = self.first_stage.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        # indices = self.permuter(indices)
        return quant_z, indices
    
    @torch.no_grad()
    def vae_decode(self, index, zshape):
        # index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage.decode(quant_z)
        return x
    
    @torch.no_grad()
    def cond_encode(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    # @torch.no_grad()
    # def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
    #            callback=lambda k: None):
    #     x = torch.cat((c,x),dim=1)
    #     block_size = self.generator.get_block_size()
    #     assert not self.generator.training
    #     if self.pkeep <= 0.0:
    #         # one pass suffices since input is pure noise anyway
    #         assert len(x.shape)==2
    #         noise_shape = (x.shape[0], steps-1)
    #         #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
    #         noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
    #         x = torch.cat((x,noise),dim=1)
    #         logits, _ = self.generator(x)
    #         # take all logits for now and scale by temp
    #         logits = logits / temperature
    #         # optionally crop probabilities to only the top k options
    #         if top_k is not None:
    #             logits = self.top_k_logits(logits, top_k)
    #         # apply softmax to convert to probabilities
    #         probs = F.softmax(logits, dim=-1)
    #         # sample from the distribution or take the most likely
    #         if sample:
    #             shape = probs.shape
    #             probs = probs.reshape(shape[0]*shape[1],shape[2])
    #             ix = torch.multinomial(probs, num_samples=1)
    #             probs = probs.reshape(shape[0],shape[1],shape[2])
    #             ix = ix.reshape(shape[0],shape[1])
    #         else:
    #             _, ix = torch.topk(probs, k=1, dim=-1)
    #         # cut off conditioning
    #         x = ix[:, c.shape[1]-1:]
    #     else:
    #         for k in range(steps):
    #             callback(k)
    #             assert x.size(1) <= block_size # make sure model can see conditioning
    #             x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
    #             logits, _ = self.generator(x_cond)
    #             # pluck the logits at the final step and scale by temperature
    #             logits = logits[:, -1, :] / temperature
    #             # optionally crop probabilities to only the top k options
    #             if top_k is not None:
    #                 logits = self.top_k_logits(logits, top_k)
    #             # apply softmax to convert to probabilities
    #             probs = F.softmax(logits, dim=-1)
    #             # sample from the distribution or take the most likely
    #             if sample:
    #                 ix = torch.multinomial(probs, num_samples=1)
    #             else:
    #                 _, ix = torch.topk(probs, k=1, dim=-1)
    #             # append to the sequence and continue
    #             x = torch.cat((x, ix), dim=1)
    #         # cut off conditioning
    #         x = x[:, c.shape[1]:]
    #     return x

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None, MASK_TOKEN_ID=17385,
               callback=lambda k: None):
        block_size = self.generator.get_block_size()  # or set manually
        x = torch.cat((c,x),dim=1) # add conditioning to the input
        x = x.clone()  # This is your input with MASK_TOKEN_IDs
        steps = (x == MASK_TOKEN_ID).sum(dim=1).max().item()  # max number of masked tokens in batch
        print(steps)
        for step in range(steps):
            for b in range(x.shape[0]):  # loop over batch
                # Find first masked token in sequence
                mask_pos = (x[b] == MASK_TOKEN_ID).nonzero(as_tuple=False)
                if mask_pos.numel() == 0:
                    continue  # nothing left to inpaint in this sample
                i = mask_pos[0].item()

                # Determine conditioning window (e.g., full or clipped by block_size)
                x_cond = x[b:b+1, :i]  # tokens before position `i`
                if x_cond.shape[1] > block_size:
                    x_cond = x_cond[:, -block_size:]

                # Run transformer forward
                logits, _ = self.generator(x_cond)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)

                probs = F.softmax(logits, dim=-1)
                if sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    _, next_token = torch.topk(probs, k=1, dim=-1)

                # Replace the masked token at position i
                x[b, i] = next_token.item()

            # Optional callback
            # callback(step)
        print(x.shape)
        x = x[:, c.shape[1]:]
        return x

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    def forward(self, x, **kwargs):
        b,c,h,w = x.shape
        # coord = (torch.arange(h * w).reshape(1, 1, h, w).float() / (h * w)).repeat(b, 1, 1, 1)
        _, target = self.vae_encode(x)
        _, cds = self.cond_encode(x)
        target = target.to(x.device)
        cds = cds.to(x.device)
        ids = torch.cat([cds,target], dim=1)
        logits, loss = self.generator.forward(ids[:,:-1], **kwargs)
        logits = logits[:, cds.shape[1]-1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss
