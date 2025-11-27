import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class ToPixel(nn.Module):
    def __init__(self, to_pixel='linear', img_size=256, in_channels=3, in_dim=512, patch_size=16) -> None:
        super().__init__()
        self.to_pixel_name = to_pixel
        
        # Handle img_size: can be int (2D), tuple of 2 (2D), or tuple of 3 (3D)
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
            self.dims = 2
        elif len(img_size) == 2:
            self.img_size = img_size
            self.dims = 2
        elif len(img_size) == 3:
            self.img_size = img_size
            self.dims = 3
        else:
            raise ValueError(f"img_size must be int or tuple of length 2 or 3, got {img_size}")
        
        # Handle patch_size: can be int (2D), tuple of 2 (2D), or tuple of 3 (3D)
        if isinstance(patch_size, int):
            if self.dims == 2:
                self.patch_size = (patch_size, patch_size)
            else:  # 3D
                self.patch_size = (patch_size, patch_size, patch_size)
        elif len(patch_size) == 2:
            if self.dims == 2:
                self.patch_size = patch_size
            else:
                raise ValueError(f"patch_size tuple of length 2 not compatible with 3D img_size {img_size}")
        elif len(patch_size) == 3:
            if self.dims == 3:
                self.patch_size = patch_size
            else:
                raise ValueError(f"patch_size tuple of length 3 not compatible with 2D img_size {img_size}")
        else:
            raise ValueError(f"patch_size must be int or tuple of length 2 or 3, got {patch_size}")
        
        # Calculate number of patches
        if self.dims == 2:
            self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        else:  # 3D
            self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1]) * (self.img_size[2] // self.patch_size[2])
        
        self.in_channels = in_channels
        if to_pixel == 'linear':
            if self.dims == 2:
                self.model = nn.Linear(in_dim, in_channels * self.patch_size[0] * self.patch_size[1])
            else:  # 3D
                self.model = nn.Linear(in_dim, in_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        elif to_pixel == 'conv':
            if self.dims == 2:
                num_patches_per_dim = self.img_size[0] // self.patch_size[0]  # e.g. 256//16 = 16
                self.model = nn.Sequential(
                    # (B, L, C) -> (B, C, H, W) with H = W = num_patches_per_dim
                    Rearrange('b (h w) c -> b c h w', h=num_patches_per_dim),
                    
                    # For example, first reduce dimension via a 1x1 conv from in_dim -> 128
                    nn.Conv2d(in_dim, 128, kernel_size=1, stride=1),
                    nn.ReLU(inplace=True),

                    # Upsample from size (num_patches_per_dim) to a larger intermediate
                    nn.Upsample(scale_factor=2, mode='nearest'),  
                    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    # Repeat upsampling until we reach the final resolution
                    # For a 16x16 patch layout, we need 4x upsampling to reach 256
                    #   16 -> 32 -> 64 -> 128 -> 256
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1),
                )
            else:  # 3D
                num_patches_per_dim_d = self.img_size[0] // self.patch_size[0]
                num_patches_per_dim_h = self.img_size[1] // self.patch_size[1]
                num_patches_per_dim_w = self.img_size[2] // self.patch_size[2]
                self.model = nn.Sequential(
                    # (B, L, C) -> (B, C, D, H, W)
                    Rearrange('b (d h w) c -> b c d h w', d=num_patches_per_dim_d, h=num_patches_per_dim_h, w=num_patches_per_dim_w),
                    
                    # Reduce dimension via a 1x1 conv from in_dim -> 128
                    nn.Conv3d(in_dim, 128, kernel_size=1, stride=1),
                    nn.ReLU(inplace=True),

                    # Upsample from size (num_patches_per_dim) to a larger intermediate
                    nn.Upsample(scale_factor=2, mode='nearest'),  
                    nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    # Repeat upsampling until we reach the final resolution
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv3d(16, in_channels, kernel_size=3, stride=1, padding=1),
                )
        elif to_pixel == 'siren':
            if self.dims == 2:
                self.model = nn.Sequential(
                    SineLayer(in_dim, in_dim * 2, is_first=True, omega_0=30.),
                    SineLayer(in_dim * 2, self.img_size[0] // self.patch_size[0] * self.patch_size[0] * in_channels, is_first=False, omega_0=30)
                )
            else:  # 3D
                self.model = nn.Sequential(
                    SineLayer(in_dim, in_dim * 2, is_first=True, omega_0=30.),
                    SineLayer(in_dim * 2, self.img_size[0] // self.patch_size[0] * self.patch_size[0] * in_channels, is_first=False, omega_0=30)
                )
        elif to_pixel == 'identity':
            self.model = nn.Identity()
        else:
            raise NotImplementedError

    def get_last_layer(self):
        if self.to_pixel_name == 'linear':
            return self.model.weight
        elif self.to_pixel_name == 'siren':
            return self.model[1].linear.weight
        elif self.to_pixel_name == 'conv':
            return self.model[-1].weight
        else:
            return None

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3) for 2D or (N, L, patch_size**3 *3) for 3D
        imgs: (N, 3, H, W) for 2D or (N, 3, D, H, W) for 3D
        """
        if self.dims == 2:
            p_h = self.patch_size[0]  # Assuming square patches
            p_w = self.patch_size[1]
            h = self.img_size[0] // p_h
            w = self.img_size[1] // p_w
            assert h * w == x.shape[1], print(h, w, x.shape[1])
            x = x.reshape(shape=(x.shape[0], h, w, p_h, p_w, 3))
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], 3, h * p_h, w * p_w))
        else:  # 3D
            p_h, p_w, p_d = self.patch_size
            # Calculate actual grid dimensions from num_patches
            h = self.img_size[0] // p_h
            w = self.img_size[1] // p_w
            d = self.img_size[2] // p_d
            assert d * h * w == x.shape[1], print(d, h, w, x.shape[1])
            x = x.reshape(shape=(x.shape[0], d, h, w, p_h, p_w, p_d, 3))
            x = torch.einsum('ndhwxyzc->nchxwydz', x)
            imgs = x.reshape(shape=(x.shape[0], 3, h * p_h, w * p_w, d * p_d))
        return imgs

    def forward(self, x):
        if self.to_pixel_name == 'linear':
            x = self.model(x)
            x = self.unpatchify(x)
        elif self.to_pixel_name == 'siren':
            x = self.model(x)
            if self.dims == 2:
                x = x.view(x.shape[0], self.in_channels, self.patch_size[0] * int(self.num_patches ** 0.5),
                           self.patch_size[0] * int(self.num_patches ** 0.5))
            else:  # 3D
                d = self.img_size[0] // self.patch_size[0]
                h = self.img_size[1] // self.patch_size[1]
                w = self.img_size[2] // self.patch_size[2]
                x = x.view(x.shape[0], self.in_channels, self.patch_size[0] * d,
                           self.patch_size[1] * h, self.patch_size[2] * w)
        elif self.to_pixel_name == 'conv':
            x = self.model(x)
        elif self.to_pixel_name == 'identity':
            pass
        return x