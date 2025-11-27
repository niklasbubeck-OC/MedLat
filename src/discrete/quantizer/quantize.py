import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange, reduce
from functools import partial
from typing import List, Union, Optional, Tuple, Sequence, Text, Mapping
import random
from .modules import *
from torch.amp import autocast
from src.registry import register_model


__all__ = ["VectorQuantizer", "GumbelQuantize", "VectorQuantizer2", "ViTVectorQuantizer", "LookupFreeQuantizer", "MultiScaleResidualQuantizer"]

_REGISTRY_PREFIX = "discrete.quantizer."

@register_model(f"{_REGISTRY_PREFIX}vector_quantizer",
code_url="https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py",
paper_url="https://arxiv.org/abs/1711.00937",)
class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE/VQ-GAN quantizer

    Args:
        n_e: Number of embeddings
        e_dim: Dimension of embedding
        beta: Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        rotation_trick: Whether to apply rotation trick
        use_ema: Whether to use EMA updates for embeddings
        ema_decay: EMA decay rate
        ema_eps: Epsilon value for numerical stability
    """

    # NOTE: this class contains a bug regarding beta; see VectorQuantizer2 for
    # a fix and use legacy=False to apply that fix. VectorQuantizer2 can be
    # used wherever VectorQuantizer has been used before and is additionally
    # more efficient.
    def __init__(self, n_e, e_dim, beta, rotation_trick: bool = False, 
                 use_ema: bool = False, ema_decay: float = 0.99, ema_eps: float = 1e-5):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.rotation_trick = rotation_trick
        self.use_ema = use_ema
        
        if use_ema:
            self.embedding = EmbeddingEMA(self.n_e, self.e_dim, ema_decay, ema_eps)
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # Ensure quantization is performed using f32
    @autocast('cuda',enabled=False)
    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        # Perform EMA update if enabled
        if self.use_ema:
            z_flattened = z_flattened  # Already defined above
            self.embedding.perform_ema_update(min_encodings, z_flattened, self.n_e)
        
        # compute loss for embedding
        commitment_loss = torch.mean((z_q.detach()-z)**2)
        codebook_loss = self.beta * torch.mean((z_q - z.detach()) ** 2)
        loss = commitment_loss + codebook_loss

        if self.rotation_trick:
            # apply rotation trick -> https://arxiv.org/abs/2410.06424
            z_q = rotate_to(z, z_q)
        else:     
            # preserve gradients -> STE
            z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


@register_model(f"{_REGISTRY_PREFIX}gumbel_quantizer",
code_url="https://github.com/karpathy/deep-vector-quantization/blob/main/model.py",
paper_url="https://arxiv.org/abs/1611.01144",)
class GumbelQuantize(nn.Module):
    """
    Gumbel Softmax trick quantizer

    Args:
        num_hiddens: Number of hidden dimensions
        embedding_dim: Dimension of embedding
        n_embed: Number of embeddings
        straight_through: Whether to use straight through estimator
    """
    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1

        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)
        # Ensure quantization is performed using f32
    @autocast('cuda',enabled=False)
    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b*h*w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q


@register_model(f"{_REGISTRY_PREFIX}vector_quantizer2",
code_url="https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py",
paper_url="https://arxiv.org/abs/1711.00937",)
class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    Supports optional EMA updates via use_ema parameter.

    Args:
        n_e: Number of embeddings
        e_dim: Dimension of embedding
        beta: Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        remap: Whether to remap indices
        unknown_index: Index to use for unknown values
        sane_index_shape: Whether to keep index shape sane
        legacy: Whether to use legacy mode
        dims: Number of dimensions
        rotation_trick: Whether to apply rotation trick
        use_norm: Whether to use normalization
        use_ema: Whether to use EMA updates for embeddings
        ema_decay: EMA decay rate
        ema_eps: Epsilon value for numerical stability
    """
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, dims=2, rotation_trick: bool = False, 
                 use_norm=False, use_ema=False, ema_decay=0.99, ema_eps=1e-5):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x
        self.beta = beta
        self.legacy = legacy
        self.dims = dims  # 2 for 2D, 3 for 3D
        self.rotation_trick = rotation_trick
        self.use_ema = use_ema
        
        if use_ema:
            self.embedding = EmbeddingEMA(self.n_e, self.e_dim, ema_decay, ema_eps)
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1

        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

        # Ensure quantization is performed using f32
    @autocast('cuda', enabled=False)
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        z=z.float()
        if self.dims == 2:
            # 2D case: (batch, channel, height, width) -> (batch, height, width, channel)
            z = rearrange(z, 'b c h w -> b h w c').contiguous()
        else:
            # 3D case: (batch, channel, depth, height, width) -> (batch, depth, height, width, channel)
            z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
            
        z_flattened = z.view(-1, self.e_dim)

        z_flattened = self.norm(z_flattened)
        embedding = self.norm(self.embedding.weight)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_q, z = self.norm(z_q), self.norm(z)
        perplexity = None
        min_encodings = None

        # Perform EMA update if enabled
        if self.use_ema:
            encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
            self.embedding.perform_ema_update(encodings, z_flattened, self.n_e)
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        if self.rotation_trick:
            # apply rotation trick -> https://arxiv.org/abs/2410.06424
            z_q = rotate_to(z, z_q)
        else:     
            # preserve gradients -> STE
            z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        if self.dims == 2:
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        else:
            z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            if self.dims == 2:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3])
            else:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel) for 2D or (batch, depth, height, width, channel) for 3D
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            if self.dims == 2:
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        z_q = self.norm(z_q)
        return z_q


@register_model(f"{_REGISTRY_PREFIX}msrq_vector_quantizer2",
code_url="https://github.com/FoundationVision/VAR/blob/main/models/quant.py",
paper_url="https://arxiv.org/pdf/2404.02905",)
class MultiScaleResidualQuantizer(nn.Module):
    """
    Multi-Scale Residual Quantizer 
    As presented in VAR: Visual Autoregressive Models
    https://arxiv.org/pdf/2404.02905

    Args:
        n_e: Number of embeddings
        e_dim: Dimension of embedding
        using_znorm: Whether to use z-normalization
        beta: Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        use_ema: Whether to use EMA updates for embeddings
        ema_decay: EMA decay rate
        ema_eps: Epsilon value for numerical stability
        default_qresi_counts: Number of quantizers to use
        v_patch_nums: List of patch sizes
        quant_resi: Quantization residual ratio
        share_quant_resi: Number of quantizers to share
    """
    def __init__(
        self, n_e: int, e_dim: int, using_znorm: bool = True, beta: float = 0.25, use_ema: bool = False, ema_decay: float = 0.99, ema_eps: float = 1e-5,
        default_qresi_counts: int = 0, v_patch_nums: Tuple[int] = None, quant_resi: float = 0.5, share_quant_resi: int = 4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.using_znorm = using_znorm
        self.use_ema = use_ema
        self.v_patch_nums = v_patch_nums
        
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.n_e), fill_value=0.0))
        self.record_hit = 0
        
        self.beta = beta
        if use_ema:
            self.embedding = EmbeddingEMA(self.n_e, self.e_dim, decay=ema_decay, eps=ema_eps)
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
        
        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1   # progressive training: not supported yet, prog_si always -1
    
    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.n_e, abs(eini) / self.n_e)
    
    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw):
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        with torch.amp.autocast('cuda', enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.n_embeddings, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)
            encoding_indices_list = []
            for si, pn in enumerate(self.v_patch_nums):
                if self.using_znorm:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                    idx_N = torch.argmin(d_no_grad, dim=1)
                
                hit_V = idx_N.bincount(minlength=self.n_embeddings).float()
                encoding_indices_list.append(idx_N)
                
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)  # This will be identity if no quant_resi was provided
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw
                
                if self.training:
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            
            mean_vq_loss *= 1. / SN
            if self.rotation_trick:
                f_hat = rotate_to(f_hat, f_BChw)
            else:
                f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
        
        # Calculate perplexity
        encodings = F.one_hot(encoding_indices_list[-1], self.n_embeddings).type(f_BChw.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Return in the same format as other quantizers
        return f_hat, mean_vq_loss, (perplexity, encodings, encoding_indices_list[-1])
    # ===================== `forward` is only used in VAE training =====================
    
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.e_dim, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.e_dim, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    def f_to_idxBl_or_fhat(self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]    # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws): # from small to large
            if 0 <= self.prog_si < si: break    # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, n_e)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph*pw))
        
        return f_hat_or_idx_Bl
    
    def idxBl_to_msrq_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """Convert indices to MSRQ input"""
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.embed_dim
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN-1):
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            # Handle both Identity and Phi cases
            if isinstance(self.quant_resi, nn.Identity):
                f_hat.add_(h_BChw)
            else:
                f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            pn_next = self.v_patch_nums[si+1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None

    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Get next autoregressive input"""
        HW = self.v_patch_nums[-1]
        if si != SN-1:
            # Handle both Identity and Phi cases
            if isinstance(self.quant_resi, nn.Identity):
                f_hat.add_(F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))
            else:
                f_hat.add_(self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic')))
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area')
        else:
            # Handle both Identity and Phi cases
            if isinstance(self.quant_resi, nn.Identity):
                f_hat.add_(h_BChw)
            else:
                f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            return f_hat, f_hat


### taken from https://github.com/hieuGoku/vitvq-gan/blob/c38704d9ce9d2d86d57d0938a60fc81eddfba2dd/enhancing/modules/stage1/quantizers.py
# Is basically the same as the ususal ones but rearranges a sequence like B, N, D, C instead of B H W C
@register_model(f"{_REGISTRY_PREFIX}vit_vector_quantizer",
code_url="https://github.com/hieuGoku/vitvq-gan/blob/c38704d9ce9d2d86d57d0938a60fc81eddfba2dd/enhancing/modules/stage1/quantizers.py",
paper_url="https://arxiv.org/abs/2404.02905",)
class ViTVectorQuantizer(nn.Module):
    def __init__(self, n_e: int, e_dim: int, beta: float = 0.25, use_norm: bool = True,
                 use_residual: bool = False, num_quantizers: Optional[int] = None, rotation_trick: bool = False,
                 use_ema: bool = False, ema_decay: float = 0.99, ema_eps: float = 1e-5, **kwargs) -> None:
        super().__init__()
        self.beta = beta
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x
        self.use_residual = use_residual
        self.num_quantizers = num_quantizers
        self.use_norm = use_norm
        self.e_dim = e_dim
        self.n_e = n_e
        self.rotation_trick = rotation_trick
        self.use_ema = use_ema
        
        if use_ema:
            self.embedding = EmbeddingEMA(n_e, e_dim, ema_decay, ema_eps)
        else:
            self.embedding = nn.Embedding(n_e, e_dim)
            self.embedding.weight.data.normal_()

    def quantize(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z_reshaped_norm = self.norm(z.view(-1, self.e_dim))
        embedding_norm = self.norm(self.embedding.weight)
        
        d = torch.sum(z_reshaped_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_norm ** 2, dim=1) - 2 * \
            torch.einsum('b d, n d -> b n', z_reshaped_norm, embedding_norm)

        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encoding_indices = encoding_indices.view(*z.shape[:-1])
        
        z_q = self.embedding(encoding_indices).view(z.shape)
        z_qnorm, z_norm = self.norm(z_q), self.norm(z)
        
        # Perform EMA update if enabled
        if self.use_ema:
            z_flattened = z.view(-1, self.e_dim)
            encoding_indices_flat = encoding_indices.view(-1)
            encodings = F.one_hot(encoding_indices_flat, self.n_e).type(z.dtype)
            self.embedding.perform_ema_update(encodings, z_flattened, self.n_e)
        
        # compute loss for embedding
        loss = self.beta * torch.mean((z_qnorm.detach() - z_norm)**2) +  \
               torch.mean((z_qnorm - z_norm.detach())**2)

        return z_qnorm, loss, (None, None, encoding_indices)

        # Ensure quantization is performed using f32
    @autocast('cuda',enabled=False)
    def forward(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z=z.float()
        if not self.use_residual:
            z_q, loss, (_, _, encoding_indices) = self.quantize(z)
        else:
            z_q = torch.zeros_like(z)
            residual = z.detach().clone()

            losses = []
            encoding_indices = []

            for _ in range(self.num_quantizers):
                z_qi, loss, indices = self.quantize(residual.clone())
                residual.sub_(z_qi)
                z_q.add_(z_qi)

                encoding_indices.append(indices)
                losses.append(loss)

            losses, encoding_indices = map(partial(torch.stack, dim = -1), (losses, encoding_indices))
            loss = losses.mean()

        # apply rotation trick
        if self.rotation_trick:
            z_q = rotate_to(z, z_q)
        # Straight-through estimator
        else: 
            z_q = z + (z_q - z).detach()

        return z_q, loss, (None, None, encoding_indices)

@register_model(f"{_REGISTRY_PREFIX}lookup_free_quantizer",)
class LookupFreeQuantizer(torch.nn.Module):
    def __init__(
        self,
        token_bits: int = 10,
        commitment_cost: float = 0.25,
        entropy_loss_weight: float = 0.1,
        entropy_loss_temperature: float = 0.01,
        entropy_gamma: float = 1.0,
        dims: int = 2
    ):
        """ 
        Args:
            token_bits -> int: The number of bits per token.
            commitment_cost -> float: The commitment cost.
            entropy_loss_weight -> float: The weight of the entropy loss.
            entropy_loss_temperature -> float: The temperature for the entropy loss.
            entropy_gamma -> float: The gamma for the entropy loss.
            dims -> int: The number of dimensions of the input.
        """
        super().__init__()
        self.token_size = token_bits
        self.codebook_size = 2 ** token_bits

        self.commitment_cost = commitment_cost
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma
        self.is_3d = dims == 3

        bits_to_indices = torch.pow(2.0, torch.arange(0, self.token_size, dtype=torch.float32))
        self.register_buffer('bits_to_indices', bits_to_indices.int())

        all_codes = torch.arange(self.codebook_size)
        bits = ((all_codes[..., None].int() & self.bits_to_indices) != 0).float()
        self.register_buffer('codebook', bits * 2.0 - 1.0)

        # Ensure quantization is performed using f32
    @autocast('cuda',enabled=False)
    def forward(self, z: torch.Tensor):
        z=z.float()
        # Rearrange dimensions based on specified dimensionality
        if self.is_3d:
            # 3D input: [b, c, d, h, w] -> [b, d, h, w, c]
            z = rearrange(z, 'b c d h w -> b d h w c')
        else:
            # 2D input: [b, c, h, w] -> [b, h, w, c]
            z = rearrange(z, 'b c h w -> b h w c')

        ones = torch.ones_like(z)
        sign_mask = (z > 0.0)
        z_quantized = torch.where(sign_mask, ones, -ones)

        min_encoding_indices = self.convert_bits_to_indices(z_quantized)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        entropy_loss = torch.zeros((), device=z.device)
        per_sample_entropy = torch.zeros((), device=z.device)
        avg_entropy = torch.zeros((), device=z.device)

        # Use entropy loss on the codebook
        if self.entropy_loss_weight != 0.0 and self.training:
            d = -2 * torch.einsum('... c, n c -> ... n', z, self.codebook)

            per_sample_entropy, avg_entropy = entropy_loss_fn(-1*d, self.entropy_loss_temperature, self.entropy_gamma)
            entropy_loss = self.entropy_loss_weight * (per_sample_entropy - avg_entropy)

        loss = commitment_loss + entropy_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        if self.is_3d:
            # 3D output: [b, d, h, w, c] -> [b, c, d, h, w]
            z_quantized = rearrange(z_quantized, 'b d h w c -> b c d h w')
        else:
            # 2D output: [b, h, w, c] -> [b, c, h, w]
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w')

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            entropy_loss=entropy_loss,
            per_sample_entropy=per_sample_entropy,
            avg_entropy=avg_entropy,
            min_encoding_indices=min_encoding_indices
        )

        # return z_quantized, result_dict # Old return
        return z_quantized, loss, (per_sample_entropy, None, min_encoding_indices) # We don't have one_hot encodings here

    def get_codebook_entry(self, indices: torch.Tensor, shape=None) -> torch.Tensor:
        indices = indices.long()
        bits = ((indices[..., None].int() & self.bits_to_indices) != 0).float()
        tokens = bits * 2.0 - 1.0  # scale to -1..1
        return tokens

    def convert_bits_to_indices(self, tokens: torch.Tensor) -> torch.Tensor:
        sign_mask = (tokens > 0.0)
        return reduce(sign_mask.int() * self.bits_to_indices, '... c -> ...', 'sum')

    def convert_indices_to_bits(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.long()
        return self.get_codebook_entry(indices)
    
