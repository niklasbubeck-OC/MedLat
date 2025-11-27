import logging
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from src.models.utils.in_and_out import DynamicPatchEmbed, DynamicToPixel, PatchEmbed, ToPixel
from src.models.utils.pos_embed import get_rope_tensor_2d, get_rope_tensor_3d, apply_rotary_emb, get_sincos_pos_embed

from src.modules import DiagonalGaussianDistribution
from src.utils import init_from_ckpt
from ...registry import register_model

logger = logging.getLogger("MedTok")
from transformers import AutoModel


SIZE_DICT = {
    "small": {"width": 512, "layers": 8, "heads": 8},
    "base": {"width": 768, "layers": 12, "heads": 12},
    "large": {"width": 1024, "layers": 24, "heads": 16},
    "xl": {"width": 1152, "layers": 28, "heads": 16},
    "huge": {"width": 1280, "layers": 32, "heads": 16},
}

# ================================
# Utility Functions
# ================================

def _to_tensor(x):
    return x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x)


# ================================
# Neural Network Components
# ================================


class SwiGLUFFN(nn.Module):
    """Swish-Gated Linear Unit Feed-Forward Network."""

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Attention(nn.Module):
    """multi-head attention with optional rotary position embedding."""

    def __init__(self, dim: int, num_heads: int = 8, use_rope: bool = True) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim % num_heads !=0, got {dim} and {num_heads}"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rope: Tensor = None, grid: tuple = None) -> Tensor:
        bsz, n_ctx, ch = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads).unbind(0)
        
        if self.use_rope and rope is not None:
            q, k = apply_rotary_emb(q, rope), apply_rotary_emb(k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(bsz, n_ctx, ch))


class Block(nn.Module):
    """transformer block with attention and feed-forward layers."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_layer=partial(nn.RMSNorm, eps=1e-6),
        use_rope: bool = True,
        attention_type: str = "vanilla",
    ) -> None:
        super().__init__()
        self.norm1, self.norm2 = norm_layer(dim), norm_layer(dim)
        if attention_type == "vanilla":
            self.attn = Attention(dim, num_heads, use_rope=use_rope)
        else:
            raise ValueError(f"Invalid attention type: {attention_type}")
        self.mlp = SwiGLUFFN(dim, int(2 / 3 * dim * mlp_ratio))

    def forward(self, x: Tensor, rope: Tensor = None, grid: tuple = None) -> Tensor:
        x = x + self.attn(self.norm1(x), rope=rope, grid=grid)
        x = x + self.mlp(self.norm2(x))
        return x


# ================================
# Encoder and Decoder
# ================================




class Decoder(nn.Module):
    """vision Transformer decoder with mask tokens for image reconstruction."""

    def __init__(
        self,
        img_size: int | tuple[int, ...] = 256,
        patch_size: int | tuple[int, ...] = 16,
        width: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        token_channels: int = 16,
        input_channels: int = 3,
        dimension: int = 2,
        use_rope: bool = True,
        attention_type: str = "vanilla",
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dimension = dimension
        self.width = width
        self.depth = depth
        self.num_heads = num_heads
        self.token_channels = token_channels
        self.use_rope = use_rope
        # Handle different input formats
        if isinstance(img_size, int):
            if dimension == 2:
                self.img_size = (img_size, img_size)
            elif dimension == 3:
                self.img_size = (img_size, img_size, img_size)
        else:
            self.img_size = img_size
            
        if isinstance(patch_size, int):
            if dimension == 2:
                self.patch_size = (patch_size, patch_size)
            elif dimension == 3:
                self.patch_size = (patch_size, patch_size, patch_size)
        else:
            self.patch_size = patch_size
        
        # Calculate grid sizes
        if dimension == 2:
            self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
            self.seq_len = self.grid_size[0] * self.grid_size[1]
            self.output_channels = 3
        elif dimension == 3:
            self.grid_size = (self.img_size[0] // self.patch_size[0], 
                            self.img_size[1] // self.patch_size[1], 
                            self.img_size[2] // self.patch_size[2])
            self.seq_len = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
            self.output_channels = 1  # For 3D, typically single channel
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

        num_layers, num_heads, width = depth, num_heads, width

        # mask token only; no absolute positional embeddings
        scale = width**-0.5
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, width))

        # decoder layers
        self.decoder_embed = nn.Linear(self.token_channels, width)
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.ModuleList(
            [Block(dim=width, num_heads=num_heads, norm_layer=norm_layer, use_rope=use_rope, attention_type=attention_type) for _ in range(num_layers)]
        )
        self.ln_post = norm_layer(width)

        # output layers
        # To pixel head (grid-dynamic)
        self.to_pixel = DynamicToPixel(to_pixel="conv", img_size=self.img_size, in_channels=input_channels,
                               in_dim=width, patch_size=self.patch_size)
        # no precomputed rope; compute per forward
        self.head_dim = self.transformer[0].attn.head_dim
        # rotary position embedding factory
        if use_rope:
            if dimension == 2:
                self.get_rope_tensor = get_rope_tensor_2d
            elif dimension == 3:
                self.get_rope_tensor = get_rope_tensor_3d
            else:
                self.get_rope_tensor = None
        else:
            self.decoder_pos_embed_learned = get_sincos_pos_embed(embed_dim=self.width, grid_size=self.grid_size, dims=dimension)
            self.decoder_pos_embed_learned = nn.Parameter(torch.tensor(self.decoder_pos_embed_learned), requires_grad=False)
            self.get_rope_tensor = None

    def forward(self, z_latents: Tensor, ids_restore: Tensor | None = None, target_img_size: tuple[int, ...] | None = None, buffer:int =0) -> Tensor:
        """forward pass through decoder."""
        z = self.decoder_embed(z_latents)
        bsz, seq_len, _ = z.shape

        if ids_restore is not None:
            num_mask_tokens = ids_restore.shape[1] + 1 - seq_len
            mask_tokens = self.mask_token.repeat(bsz, num_mask_tokens, 1)
            z_ = torch.cat([z, mask_tokens], dim=1)
            expanded_ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, z_.shape[-1])
            z = torch.gather(z_, dim=1, index=expanded_ids_restore)

        z = self.ln_pre(z)
        if not self.use_rope:
            z[:,buffer:] = z[:,buffer:] + self.decoder_pos_embed_learned

        # compute runtime rope from target grid
        rope = None
        grid = None
        if self.get_rope_tensor is not None:
            assert target_img_size is not None, "Decoder requires target_img_size to compute RoPE dynamically."
            if self.dimension == 2:
                H, W = target_img_size
                ph, pw = self.patch_size
                gh, gw = H // ph, W // pw
                rope = self.get_rope_tensor(self.head_dim, gh, gw).unsqueeze(0).to(z.device)
                grid = (gh, gw)
            else:
                D, H, W = target_img_size
                pd, ph, pw = self.patch_size
                gd, gh, gw = D // pd, H // ph, W // pw
                rope = self.get_rope_tensor(self.head_dim, gd, gh, gw).unsqueeze(0).to(z.device)
                grid = (gd, gh, gw)
            rope = rope.expand(bsz, -1, -1)
        for block in self.transformer:
            z = block(z, rope, grid=grid)
            print(f"z shape: {z.shape}")
        z = self.ln_post(z)
        print(f"lnPost: {z.shape}")
        z = self.to_pixel(z[:,buffer:], img_size=target_img_size)
        print(f"toPixel: {z.shape}")
        return z


# ================================
# Main DeTok Model
# ================================


@register_model("token.dinotok.base")
class DinoTok(nn.Module): 
    """
    l-DeTok: latent denoising makes good visual tokenizers.
    Supports both 2D and 3D inputs with arbitrary dimensions.
    """
    def __init__(
        self,
        image_size: int | tuple[int, ...] = 256,
        patch_size: int | tuple[int, ...] = 16,
        input_channels: int = 3,
        encoder: str = "dinov3-vits16-pretrain-lvd1689m",
        dec_width: int = 768,
        dec_depth: int = 12,
        dec_num_heads: int = 12,
        token_channels: int = 16,
        mask_ratio: float = 0.75,
        gamma: float = 3.0,
        use_additive_noise: bool = False,
        dimension: int = 2,
        # normalization parameters used for generative model training
        mean=0.0,
        std=1.0,
        scale_factor: float = 1.0,
        ckpt_path: str = None,
        use_rope: bool = True,
        attention_type: str = "vanilla",
    ) -> None:
        super().__init__()

        print(f"token channels: {token_channels}, gamma: {gamma}")

        self.encoder = AutoModel.from_pretrained(encoder).eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        
        self.decoder = Decoder(
            img_size=image_size,
            patch_size=patch_size,
            input_channels=input_channels,
            width=dec_width,
            depth=dec_depth,
            num_heads=dec_num_heads,
            token_channels=token_channels,
            dimension=dimension,
            use_rope=use_rope,
            attention_type=attention_type,
        )

        # model configuration
        self.dimension = dimension
        self.image_size = image_size
        self.patch_size = patch_size
        
        # Calculate canonical sizes (used as defaults; runtime grid is dynamic)
        if isinstance(image_size, int):
            if dimension == 2:
                self.img_size = (image_size, image_size)
            elif dimension == 3:
                self.img_size = (image_size, image_size, image_size)
        else:
            self.img_size = image_size
            
        if isinstance(patch_size, int):
            if dimension == 2:
                self.patch_size_tuple = (patch_size, patch_size)
            elif dimension == 3:
                self.patch_size_tuple = (patch_size, patch_size, patch_size)
        else:
            self.patch_size_tuple = patch_size
            
        if dimension == 2:
            self.grid_size = (self.img_size[0] // self.patch_size_tuple[0], self.img_size[1] // self.patch_size_tuple[1])
        elif dimension == 3:
            self.grid_size = (self.img_size[0] // self.patch_size_tuple[0], 
                            self.img_size[1] // self.patch_size_tuple[1], 
                            self.img_size[2] // self.patch_size_tuple[2])
            
        self.use_additive_noise = use_additive_noise
        self.gamma = gamma

        self.scale_factor = scale_factor

        # initialize weights
        self.apply(self._init_weights)

        # setup to-posteriors function
        self.to_posteriors = partial(DiagonalGaussianDistribution, channel_dim=-1)

        # setup normalization parameters
        if isinstance(mean, np.ndarray) or isinstance(mean, list):
            mean = np.array(mean).reshape(1, -1, 1, 1)
            std = np.array(std).reshape(1, -1, 1, 1)
        self.register_buffer("mean", torch.tensor(mean), persistent=False)
        self.register_buffer("std", torch.tensor(std), persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok] params: {params_M:.2f}M, {dimension}D, size: {self.img_size}, patch: {self.patch_size_tuple}")
        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)

    def _init_weights(self, module: nn.Module) -> None:
        """initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def freeze_everything_but_decoder(self) -> None:
        """freeze all parameters except the decoder, used for decoder fine-tuning"""
        for param in self.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = True

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok] trainable params: {params_M:.2f}M (after freezing all but decoder)")

    def reset_stats(self, mean: Tensor | np.ndarray | float, std: Tensor | np.ndarray | float) -> None:
        if isinstance(mean, float) and isinstance(std, float) or (mean.ndim == 0 and std.ndim == 0):
            # a single digit global mean and global std
            self.register_buffer("mean", _to_tensor(mean), persistent=False)
            self.register_buffer("std", _to_tensor(std), persistent=False)
        else:
            n_chans = mean.shape[-1]
            self.register_buffer("mean", _to_tensor(mean).reshape(1, 1, n_chans), persistent=False)
            self.register_buffer("std", _to_tensor(std).reshape(1, 1, n_chans), persistent=False)
        logger.info(f"Resetting mean and std ({mean.shape=}, {std.shape=})")
        logger.info(f"Mean: {self.mean}")
        logger.info(f"Std: {self.std}")

    def denormalize_z(self, z: Tensor) -> Tensor:
        """denormalize latent tokens."""
        return z * self.std.to(z) / self.scale_factor + self.mean.to(z)

    def normalize_z(self, z: Tensor) -> Tensor:
        """normalize latent tokens."""
        return (z - self.mean.to(z)) * self.scale_factor / self.std.to(z)

    def encode_into_posteriors(self, x: Tensor):
        """encode image into posterior distributions."""
        z = self.encoder(x, mask_ratio=0.0)[0]
        return self.to_posteriors(z)

    def encode(self, x: Tensor, sampling: bool = False, mask_ratio: float = -1, noise_level: float = -1.0):
        """encode image into latent tokens."""
        z_latents = self.encoder(x).last_hidden_state
        z_latents = z_latents[:,5:] # get rid of class and reg tokens


        if self.training and self.gamma > 0.0:
            device = z_latents.device
            bsz, n_tokens, chans = z_latents.shape
            if noise_level > 0.0:
                noise_level_tensor = torch.full((bsz, 1, 1), noise_level, device=device)
            else:
                noise_level_tensor = torch.rand(bsz, 1, 1, device=device)
            noise_level_tensor = noise_level_tensor.expand(-1, n_tokens, chans)
            noise = torch.randn(bsz, n_tokens, chans, device=device) * self.gamma
            if self.use_additive_noise:
                z_latents = z_latents + noise_level_tensor * noise
            else:
                z_latents = (1 - noise_level_tensor) * z_latents + noise_level_tensor * noise

        return z_latents

    def forward(self, x: Tensor, return_posteriors: bool = False):
        """forward pass through the entire model."""
        z_latents = self.encode(x, sampling=self.training)
        # target image size from input x
        target_img_size = x.shape[-2:] if self.dimension == 2 else x.shape[-3:]
        decoded = self.decoder(z_latents, ids_restore=None, target_img_size=target_img_size, buffer=0)
        return decoded, None

    def tokenize(self, x: Tensor, sampling: bool = False) -> Tensor:
        """tokenize input image and normalize the latent tokens."""
        z = self.encode(x, sampling=sampling, mask_ratio=0.0)
        z = self.normalize_z(z)
        if self.dimension == 2:
            H, W = x.shape[-2], x.shape[-1]
            ph, pw = self.patch_size_tuple
            gh, gw = H // ph, W // pw
            z = rearrange(z, "b (h w) c -> b c h w", h=gh, w=gw)
        elif self.dimension == 3:
            D, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
            pd, ph, pw = self.patch_size_tuple
            gd, gh, gw = D // pd, H // ph, W // pw
            z = rearrange(z, "b (d h w) c -> b c d h w", d=gd, h=gh, w=gw)
        return z

    def detokenize(self, z: Tensor) -> Tensor:
        """detokenize latent representation back to image."""
        if self.dimension == 2:
            H, W = z.shape[-2], z.shape[-1]
            target_img_size = (H * self.patch_size_tuple[0], W * self.patch_size_tuple[1])
            z = rearrange(z, "b c h w -> b (h w) c")
        elif self.dimension == 3:
            D, H, W = z.shape[-3], z.shape[-2], z.shape[-1]
            target_img_size = (D * self.patch_size_tuple[0], H * self.patch_size_tuple[1], W * self.patch_size_tuple[2])
            z = rearrange(z, "b c d h w -> b (d h w) c")
        z = self.denormalize_z(z)
        decoded_images = self.decoder(z, ids_restore=None, target_img_size=target_img_size)
        return decoded_images

    def sample_from_moments(self, moments: Tensor) -> Tensor:
        """sample from latent moments."""
        z = DiagonalGaussianDistribution(moments, channel_dim=-1).sample()
        z = self.normalize_z(z)
        if self.dimension == 2:
            z = rearrange(z, "b (h w) c -> b c h w", h=self.grid_size[0], w=self.grid_size[1])
        elif self.dimension == 3:
            z = rearrange(z, "b (d h w) c -> b c d h w", d=self.grid_size[0], h=self.grid_size[1], w=self.grid_size[2])
        return z

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        """reconstruct input image."""
        return self.detokenize(self.tokenize(x))
