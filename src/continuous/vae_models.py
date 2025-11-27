import torch
import torch.nn as nn
import numpy as np
from ..utils import init_from_ckpt
from ..modules import Encoder, Decoder, DiagonalGaussianDistribution, get_conv_layer
from ..registry import register_model, get_model

__all__ = [
    "AutoencoderKL",
    "AutoencoderKL_f4",
    "AutoencoderKL_f8",
    "AutoencoderKL_f16",
    "AutoencoderKL_f32",
]

_REGISTRY_PREFIX = "continuous.autoencoder."


class AutoencoderKL(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 embed_dim: int = None,
                 ckpt_path: str = None):
        super().__init__()
        self.dims = getattr(encoder, "dims", 2)
        conv_layer = get_conv_layer(self.dims)

        self.encoder = encoder
        self.decoder = decoder
        self.encoder_z_channels = getattr(encoder, "z_channels", None)
        if self.encoder_z_channels is None:
            raise ValueError(f"Encoder {encoder.__class__.__name__} must define z_channels.")

        if embed_dim is None:
            embed_dim = self.encoder_z_channels

        self.quant_conv = conv_layer(2 * self.encoder_z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = conv_layer(embed_dim, self.encoder_z_channels, 1)
        self.embed_dim = embed_dim

        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)


    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def encode_sliding(self, x, roi_size=(64, 64, 64), sw_batch_size=1, overlap=0.0):
        z = sliding_window_inference(
            inputs=x,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=self.encode,
            mode="gaussian",
            overlap=overlap,
        )
        return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def decode_sliding(self, z, roi_size=(8, 8, 8), sw_batch_size=1, overlap=0.0):
        dec = sliding_window_inference(
            inputs=z,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=self.decode,
            mode="gaussian",
            overlap=overlap,
        )
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior


@register_model(f"{_REGISTRY_PREFIX}kl-f4d3", 
code_url="https://github.com/CompVis/latent-diffusion/blob/main/models/first_stage_models/kl-f4/config.yaml", 
paper_url="https://arxiv.org/pdf/2112.10752",)
def AutoencoderKL_f4(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=3,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 2, 4],
    num_res_blocks=2,
    attn_resolutions=[],
    dropout=0.0,
    **kwargs):
    """
    AutoencoderKL with compression factor 4
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
    """
    encoder = Encoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    decoder = Decoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    return AutoencoderKL(encoder=encoder, decoder=decoder, **kwargs)

@register_model(f"{_REGISTRY_PREFIX}kl-f8d4", 
code_url="https://github.com/CompVis/latent-diffusion/blob/main/models/first_stage_models/kl-f8/config.yaml", 
paper_url="https://arxiv.org/pdf/2112.10752",)
def AutoencoderKL_f8(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=4,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_resolutions=[],
    dropout=0.0,
    **kwargs):
    """
    AutoencoderKL with compression factor 8.
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
        kwargs: Can include 'ddconfig' dict or any ddconfig key directly
    """
    encoder = Encoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    decoder = Decoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    return AutoencoderKL(encoder=encoder, decoder=decoder, **kwargs)

@register_model(f"{_REGISTRY_PREFIX}kl-f16d8", 
code_url="https://github.com/CompVis/latent-diffusion/blob/main/configs/autoencoder/autoencoder_kl_16x16x16.yaml", 
paper_url="https://arxiv.org/pdf/2112.10752",
description="ATTENTION: There are two different official configurations with z=8 and z=16 depending on the repo, we use z=8 here.")
def AutoencoderKL_f16(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=8,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 1, 2, 2, 4],
    num_res_blocks=2,
    attn_resolutions=[16],
    dropout=0.0,
    **kwargs):
    """
    AutoencoderKL with compression factor 16
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
    """
    encoder = Encoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    decoder = Decoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    return AutoencoderKL(encoder=encoder, decoder=decoder, **kwargs)

@register_model(f"{_REGISTRY_PREFIX}kl-f32d64", 
code_url="https://github.com/CompVis/latent-diffusion/blob/main/models/first_stage_models/kl-f32/config.yaml",
paper_url="https://arxiv.org/pdf/2112.10752",)
def AutoencoderKL_f32(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=64,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 1, 2, 2, 4, 4],
    num_res_blocks=2,
    attn_resolutions=[16, 8],
    dropout=0.0,
    **kwargs):
    """
    AutoencoderKL with compression factor 32
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
    """
    encoder = Encoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    decoder = Decoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    return AutoencoderKL(encoder=encoder, decoder=decoder, **kwargs)

