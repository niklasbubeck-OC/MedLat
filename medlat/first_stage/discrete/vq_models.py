# Adopted from LDM's KL-VAE: https://github.com/CompVis/latent-diffusion
import torch
from torch import nn
from medlat.utils import init_from_ckpt
from typing import Optional, Sequence, Union, List, Any, Dict, Tuple
from medlat.first_stage.discrete.modules.ldm_modules import get_conv_layer
from medlat.modules.alignments import AlignmentModule
from einops import rearrange
from medlat.base import DiscreteFirstStage
__all__ = ["VQModel", "VQModelTransformer"]


class VQModel(DiscreteFirstStage):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        alignment: AlignmentModule = None,
        ckpt_path=None,
        # Additional parameters
        quant_conv_ks=1,   ## in var its 3, but VQVAE / VQGAN uses 1
        pre_post_layer="conv",
    ):
        super().__init__()
        self._embed_dim = quantizer.e_dim
        self._n_embed = quantizer.n_e
        self.dims = getattr(encoder, "dims", 2)
        self.encoder = encoder
        self.decoder = decoder
        self.alignment = alignment
        self.z_channels = getattr(encoder, "z_channels", None)
        self.quantizer = quantizer

        if self.z_channels is None:
            raise ValueError(f"Encoder {encoder.__class__.__name__} must define z_channels.")

        self._vae_stride = getattr(encoder, "vae_stride", None)
        if pre_post_layer == "conv":
            conv_layer = get_conv_layer(self.dims)
            self.quant_conv = conv_layer(self.z_channels, self.embed_dim, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
            self.post_quant_conv = conv_layer(self.embed_dim, self.z_channels, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        elif pre_post_layer == "none":
            self.quant_conv = nn.Identity()
            self.post_quant_conv = nn.Identity()
        else:
            raise ValueError(f"Invalid pre_post_layer: {pre_post_layer}")


        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)

    @property
    def vae_stride(self):
        return self._vae_stride

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def n_embed(self):
        return self._n_embed

    # def _check_msrq_features(self, method_name):
    #     if not self.quantizer.has_msrq_features:
    #         raise NotImplementedError(
    #             f"Method {method_name} requires MSRQ features. Please initialize VQModel_Combined with "
    #             "v_patch_nums, quant_resi, share_quant_resi, and using_znorm parameters."
    #         )

    def lock_parameters(self):
        """Lock the parameters of the model to prevent them from being updated during training."""
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        h = self.encoder(x)
        # Allow encoders that return (features, aux)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info

    def quantize(self, h):
        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info
    
    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h, None, None

    def decode_from_prequant(self, h):
        quant, emb_loss, info = self.quantizer(h)
        return self.decode(quant)
        
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, out_shape=None):
        quant_b = self.quantizer.get_codebook_entry(code_b, shape=out_shape)
        # Move channel dimension (which is last) to the second
        if quant_b.dim() == 4:
            # (B, H, W, C) -> (B, C, H, W)
            quant_b = quant_b.permute(0, 3, 1, 2).contiguous()
        elif quant_b.dim() == 5:
            # (B, D, H, W, C) -> (B, C, D, H, W)
            quant_b = quant_b.permute(0, 4, 1, 2, 3).contiguous()
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, ind = self.encode(input)
        dec = self.decode(quant)

        if self.alignment is not None:
            alignment_loss, _ = self.alignment(quant, input)
            diff = diff + alignment_loss
            
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff


class VQModelTransformer(DiscreteFirstStage):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        alignment: AlignmentModule = None,
        ckpt_path=None,
        pre_post_layer="linear",
    ):
        super().__init__()
        self._embed_dim = quantizer.e_dim
        self._n_embed = quantizer.n_e
        self.dims = getattr(encoder, "dims", 2)
        self.encoder = encoder
        self.decoder = decoder
        self.alignment = alignment
        self.z_channels = getattr(encoder, "z_channels", None)
        self.quantizer = quantizer

        if self.z_channels is None:
            raise ValueError(f"Encoder {encoder.__class__.__name__} must define z_channels.")

        self._vae_stride = getattr(encoder, "vae_stride", None)
        if pre_post_layer == "linear":
            self.quant_conv = nn.Linear(self.z_channels, self._embed_dim)
            self.post_quant_conv = nn.Linear(self._embed_dim, self.z_channels)
        elif pre_post_layer == "none":
            self.quant_conv = nn.Identity()
            self.post_quant_conv = nn.Identity()
        else:
            raise ValueError(f"Invalid pre_post_layer: {pre_post_layer}")

        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)

    @property
    def vae_stride(self):
        return self._vae_stride

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def n_embed(self):
        return self._n_embed

    def lock_parameters(self):
        """Lock the parameters of the model to prevent them from being updated during training."""
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        h, aux = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info, aux

    def quantize(self, h):
        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h, aux = self.encoder(x)
        h = self.quant_conv(h)
        return h, None, None, aux

    def decode_from_prequant(self, h, aux=None):
        quant, emb_loss, info = self.quantizer(h)
        return self.decode(quant, aux=aux)
        
    def decode(self, quant, aux=None):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, ids_restore=aux["ids_restore"] if aux is not None else None)
        return dec

    def decode_code(self, code_b, out_shape=None, aux=None):
        quant_b = self.quantizer.get_codebook_entry(code_b, shape=out_shape)
        # Move channel dimension (which is last) to the second
        if quant_b.dim() == 4:
            # (B, H, W, C) -> (B, C, H, W)
            quant_b = quant_b.permute(0, 3, 1, 2).contiguous()
        elif quant_b.dim() == 5:
            # (B, D, H, W, C) -> (B, C, D, H, W)
            quant_b = quant_b.permute(0, 4, 1, 2, 3).contiguous()
        dec = self.decode(quant_b, aux=aux)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, ind, aux = self.encode(input)
        dec = self.decode(quant, aux=aux)

        if self.alignment is not None:
            alignment_loss, _ = self.alignment(quant, input)
            diff = diff + alignment_loss
            
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff