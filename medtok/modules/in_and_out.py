from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _to_tuple(value: int | Tuple[int, ...], dims: int) -> Tuple[int, ...]:
    if isinstance(value, tuple):
        if len(value) != dims:
            raise ValueError(f"Expected {dims} values, got {len(value)}")
        return value
    return (int(value),) * dims


def _infer_dims(img_size: int | Tuple[int, ...]) -> int:
    return 3 if isinstance(img_size, tuple) and len(img_size) == 3 else 2


class PatchEmbed(nn.Module):
    """
    Lightweight convolutional patch embedding that supports 2D and 3D inputs.
    """

    def __init__(
        self,
        *,
        img_size: int | Tuple[int, ...],
        patch_size: int | Tuple[int, ...],
        in_chans: int,
        embed_dim: int,
        bias: bool = True,
        to_embed: str = "conv",
    ) -> None:
        super().__init__()
        self.dims = _infer_dims(img_size)
        self.img_size = _to_tuple(img_size, self.dims)
        kernel = _to_tuple(patch_size, self.dims)
        stride = kernel
        self.grid_size = tuple(img // patch for img, patch in zip(self.img_size, kernel))
        self.num_patches = int(torch.tensor(self.grid_size).prod().item())
        self.embed_dim = embed_dim
        self.patch_size = kernel
        if to_embed == "conv":
            if self.dims == 2:
                self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel, stride=stride, bias=bias)
            else:
                self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=stride, bias=bias)
        elif to_embed == "linear":
            self.proj = nn.Linear(in_chans * self.num_patches, embed_dim)
        else:
            raise ValueError(f"Unsupported embedding type '{to_embed}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DynamicPatchEmbed(PatchEmbed):
    """
    Alias of ``PatchEmbed`` kept for compatibility. Input shape can vary between calls.
    """

    pass


class ToPixel(nn.Module):
    """
    Inverse of PatchEmbed using a transposed convolution.
    """

    def __init__(
        self,
        *,
        img_size: int | Tuple[int, ...],
        patch_size: int | Tuple[int, ...],
        out_channels: int,
        in_dim: int,
        to_pixel: str = "conv",
    ) -> None:
        super().__init__()
        self.dims = _infer_dims(img_size)
        self.img_size = _to_tuple(img_size, self.dims)
        kernel = _to_tuple(patch_size, self.dims)
        self.patch_size = kernel
        self.patch_area = int(torch.tensor(self.patch_size).prod().item())
        stride = kernel
        self.grid_size = tuple(img // patch for img, patch in zip(self.img_size, kernel))
        self.num_patches = int(torch.tensor(self.grid_size).prod().item())
        self.out_channels = out_channels
        self.in_dim = in_dim
        if to_pixel == "conv":
            if self.dims == 2:
                self.proj = nn.ConvTranspose2d(in_dim, out_channels, kernel_size=kernel, stride=stride)
            else:
                self.proj = nn.ConvTranspose3d(in_dim, out_channels, kernel_size=kernel, stride=stride)
        elif to_pixel == "linear":
            # Map each patch token to its patch-area pixels, then unpatchify
            self.proj = nn.Linear(in_dim, out_channels * self.patch_area)
        elif to_pixel == "identity":
            self.proj = nn.Identity()
        else: 
            raise ValueError(f"Unsupported pixel type '{to_pixel}'")

    def forward(self, tokens: torch.Tensor, img_size: Tuple[int, ...] | None = None) -> torch.Tensor:
        bsz, seq_len, channels = tokens.shape
        target_size = self.img_size if img_size is None else _to_tuple(img_size, self.dims)
        grid = tuple(dim // patch for dim, patch in zip(target_size, self.patch_size))
        
        if isinstance(self.proj, nn.Linear):
            # For Linear: map tokens to patch pixels then fold (unpatchify)
            if self.dims == 2:
                gh, gw = grid
                ph, pw = self.patch_size
                x = self.proj(tokens)  # (bsz, seq_len, out_channels * patch_area)
                x = x.view(bsz, gh, gw, self.out_channels, ph, pw)
                x = x.permute(0, 3, 1, 4, 2, 5).reshape(bsz, self.out_channels, gh * ph, gw * pw)
            elif self.dims == 3:
                gd, gh, gw = grid
                pd, ph, pw = self.patch_size
                x = self.proj(tokens)
                x = x.view(bsz, gd, gh, gw, self.out_channels, pd, ph, pw)
                x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(bsz, self.out_channels, gd * pd, gh * ph, gw * pw)
            else:
                raise ValueError("Unsupported dims for linear to-pixel projection.")
        elif isinstance(self.proj, nn.Identity):
            # This is used in alignment modules where we need token format (B, L, D)
            return tokens
        else:
            # For Conv: reshape to spatial first, then apply conv
            x = tokens.transpose(1, 2).reshape(bsz, channels, *grid)
            x = self.proj(x)
        return x


class DynamicToPixel(ToPixel):
    """
    Variant that requires ``img_size`` to be provided at runtime.
    """

    def forward(self, tokens: torch.Tensor, img_size: Tuple[int, ...] | None = None) -> torch.Tensor:
        if img_size is None:
            raise ValueError("DynamicToPixel.forward expects 'img_size'.")
        return super().forward(tokens, img_size=img_size)

