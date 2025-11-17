"""
Mixed adaptation from:

    Liu et al. 2022, A ConvNet for the 2020s.
    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Ronneberger et al., 2015. Convolutional Networks for Biomedical Image Segmentation.

    copied from "the well" repository https://github.com/PolymathicAI/the_well/blob/master/the_well/benchmark/models/unet_convnext/__init__.py

If you use this implementation, please cite original work above.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint


from model.common import BaseModel

conv_modules = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}

permute_channel_strings = {
    2: [
        "N C H W -> N H W C",
        "N H W C -> N C H W",
    ],
    3: [
        "N C D H W -> N D H W C",
        "N D H W C -> N C D H W",
    ],
}

# For 3D, we use nn.Conv3d
conv_modules = {3: nn.Conv3d}

class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    """
    def __init__(self, normalized_shape, n_spatial_dims, eps=1e-6, data_format="channels_last"):
        super().__init__()
        if data_format == "channels_last":
            padded_shape = (normalized_shape,)
        else:
            padded_shape = (normalized_shape,) + (1,) * n_spatial_dims
        self.weight = nn.Parameter(torch.ones(padded_shape))
        self.bias = nn.Parameter(torch.zeros(padded_shape))
        self.n_spatial_dims = n_spatial_dims
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            # channels_first: normalize across channel dimension
            x = F.normalize(x, p=2, dim=1, eps=self.eps) * self.weight
            return x

class Downsample(nn.Module):
    """
    Downsample layer using a convolution with stride 2.
    """
    def __init__(self, dim_in, dim_out, n_spatial_dims=3):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm(dim_in, n_spatial_dims, data_format="channels_first"),
            conv_modules[n_spatial_dims](dim_in, dim_out, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)

class Block(nn.Module):
    """
    ConvNeXt Block: Depthwise conv -> LayerNorm -> 1x1 conv -> GELU -> 1x1 conv -> residual connection.
    """
    def __init__(self, dim, n_spatial_dims, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.dwconv = conv_modules[n_spatial_dims](
            dim, dim, kernel_size=7, padding=3, groups=dim
        )
        self.norm = LayerNorm(dim, n_spatial_dims, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) \
                     if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        # x: (B, C, D, H, W) -> (B, D, H, W, C)
        x = self.dwconv(x)
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # back to channels first
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = residual + self.drop_path(x)
        return x

class Stage(nn.Module):
    """
    A stage consisting of a series of ConvNeXt blocks, with an optional downsampling.
    """
    def __init__(self, dim_in, dim_out, n_spatial_dims, depth=1, drop_path=0.0,
                 layer_scale_init_value=1e-6, mode="down"):
        super().__init__()
        self.mode = mode
        # When mode=="down", apply a downsampling layer after the blocks.
        self.blocks = nn.Sequential(*[
            Block(dim_in, n_spatial_dims, drop_path, layer_scale_init_value)
            for _ in range(depth)
        ])
        if mode == "down":
            self.resample = Downsample(dim_in, dim_out, n_spatial_dims)
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        x = self.blocks(x)
        x = self.resample(x)
        return x

# -------------------------------------------
# New Model: ConvNeXt Encoder for Coefficient Prediction
# -------------------------------------------
class ConvNextEncoderForCoeffs(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_coeffs: int,
                 n_spatial_dims: int = 3,
                 spatial_resolution: tuple[int, ...] = (128, 128, 128),
                 stages: int = 4,
                 blocks_per_stage: int = 1,
                 blocks_at_neck: int = 1,
                 init_features: int = 32,
                 gradient_checkpointing: bool = False):
        """
        Build an encoder inspired by the ConvNeXt encoder.
        """
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.gradient_checkpointing = gradient_checkpointing
        features = init_features
        encoder_dims = [features * (2 ** i) for i in range(stages + 1)]
        
        # Initial projection: from in_channels to initial features
        self.in_proj = conv_modules[n_spatial_dims](in_channels, features, kernel_size=3, padding=1)
        
        # Encoder stages with downsampling
        self.encoder = nn.ModuleList([
            Stage(encoder_dims[i], encoder_dims[i+1], n_spatial_dims, depth=blocks_per_stage, mode="down")
            for i in range(stages)
        ])
        
        # Neck stage (without downsampling)
        self.neck = Stage(encoder_dims[-1], encoder_dims[-1], n_spatial_dims, depth=blocks_at_neck, mode="neck")
        
        # Global average pooling and FC to predict coefficients
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(encoder_dims[-1], num_coeffs)

    def optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        else:
            return layer(*inputs, **kwargs)

    def forward(self, x):
        # x: (B, in_channels, D, H, W)
        x = self.in_proj(x)
        for stage in self.encoder:
            x = self.optional_checkpointing(stage, x)
        x = self.neck(x)
        x = self.avgpool(x)  # Shape: (B, C, 1, 1, 1)
        x = torch.flatten(x, 1)  # Shape: (B, C)
        x = self.fc(x)  # Shape: (B, num_coeffs)
        return x
