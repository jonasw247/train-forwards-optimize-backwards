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


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self, normalized_shape, n_spatial_dims, eps=1e-6, data_format="channels_last"
    ):
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
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            x = F.normalize(x, p=2, dim=1, eps=self.eps) * self.weight
            return x


class Upsample(nn.Module):
    r"""Upsample layer."""

    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_transpose_modules[n_spatial_dims](
                dim_in, dim_out, kernel_size=2, stride=2
            ),
        )

    def forward(self, x):
        return self.block(x)


class Downsample(nn.Module):
    r"""Downsample layer."""

    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_modules[n_spatial_dims](dim_in, dim_out, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, n_spatial_dims, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.dwconv = conv_modules[n_spatial_dims](
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, n_spatial_dims, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = rearrange(x, permute_channel_strings[self.n_spatial_dims][0])
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # (N, H, W, C) -> (N, C, H, W)
        x = rearrange(x, permute_channel_strings[self.n_spatial_dims][1])
        x = input + self.drop_path(x)
        return x


class Stage(nn.Module):
    r"""ConvNeXt Stage.
    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        n_spatial_dims (int): Number of spatial dimensions.
        depth (int): Number of blocks in the stage.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mode (str): Down, Up, Neck. Default: "down"
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        n_spatial_dims,
        depth=1,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        mode="down",
        skip_project=False,
    ):
        super().__init__()

        if skip_project:
            self.skip_proj = conv_modules[n_spatial_dims](2 * dim_in, dim_in, 1)
        else:
            self.skip_proj = nn.Identity()
        if mode == "down":
            self.resample = Downsample(dim_in, dim_out, n_spatial_dims)
        elif mode == "up":
            self.resample = Upsample(dim_in, dim_out, n_spatial_dims)
        else:
            self.resample = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                Block(dim_in, n_spatial_dims, drop_path, layer_scale_init_value)
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        x = self.skip_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.resample(x)
        return x

# this is added
class CoeffEncoder(nn.Module):
    def __init__(self, n_coeffs: int, neck_output_shape: torch.Size):
        super().__init__()
        self.output_shape = neck_output_shape  # e.g., torch.Size([C, H, W])
        output_dim = torch.tensor(neck_output_shape).prod().item()  # total number of output elements
        self.linear = nn.Linear(n_coeffs, output_dim)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coeffs (torch.Tensor): Shape (B, n_coeffs)
        
        Returns:
            torch.Tensor: Shape (B, *self.output_shape)
        """
        x = self.linear(coeffs)  # shape: (B, output_dim)
        x = x.view(coeffs.size(0), *self.output_shape)  # reshape to (B, C, H, W)
        return x

class UNetConvNext(BaseModel):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        stages: int = 4,
        blocks_per_stage: int = 1,
        blocks_at_neck: int = 1,
        init_features: int = 32,
        gradient_checkpointing: bool = False,
        n_coeffs: int = 0,
        clipping_output = [float('-inf'), float('inf')],
        final_activation = None,    
    ):
        super().__init__(n_spatial_dims, spatial_resolution)
        self.n_spatial_dims = n_spatial_dims
        features = init_features
        self.gradient_checkpointing = gradient_checkpointing
        encoder_dims = [features * 2**i for i in range(stages + 1)]
        decoder_dims = [features * 2**i for i in range(stages, -1, -1)]
        encoder = []
        decoder = []
        self.in_proj = conv_modules[n_spatial_dims](
            dim_in, features, kernel_size=3, padding=1
        )
        self.out_proj = conv_modules[n_spatial_dims](
            features, dim_out, kernel_size=3, padding=1
        )
        for i in range(stages):
            encoder.append(
                Stage(
                    encoder_dims[i],
                    encoder_dims[i + 1],
                    n_spatial_dims,
                    blocks_per_stage,
                    mode="down",
                )
            )
            decoder.append(
                Stage(
                    decoder_dims[i],
                    decoder_dims[i + 1],
                    n_spatial_dims,
                    blocks_per_stage,
                    mode="up",
                    skip_project=i != 0,
                )
            )
        self.encoder = nn.ModuleList(encoder)
        self.neck = Stage(
            encoder_dims[-1],
            encoder_dims[-1],
            n_spatial_dims,
            blocks_at_neck,
            mode="neck",
        )
        self.decoder = nn.ModuleList(decoder)

        self.n_coeffs = n_coeffs
        # if n_coeffs > 0, add a CoeffEncoder
        if n_coeffs > 0:
            #self.coeffEncoder = CoeffEncoder(n_coeffs, self.neck.blocks[0].norm.normalized_shape)
            neck_channels = self.neck.blocks[0].norm.normalized_shape[0]
            spatial_shape = tuple(s // 2**stages for s in spatial_resolution)  # will now be (6, 6, 6)
            self.coeffEncoder = CoeffEncoder(n_coeffs, (neck_channels, *spatial_shape))
            a=1


        self.clipping_output = clipping_output
        #self.final_activation = final_activation
        if final_activation == "sigmoid":
            self.final_activation = torch.sigmoid
        elif final_activation == "relu":
            self.final_activation = F.relu
        elif final_activation == "linear":
            def linear (x):
                return x
            self.final_activation = linear
        elif final_activation == "tanh":
            self.final_activation = torch.tanh 
        else:
            raise ValueError("final_activation not found")

    def optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        else:
            return layer(*inputs, **kwargs)

    def forward(self, x, coeffs = None):
        x = self.in_proj(x)
        skips = []
        for i, enc in enumerate(self.encoder):
            skips.append(x)
            x = self.optional_checkpointing(enc, x)

        x = self.neck(x)

        # add coeffs here TODO
        if self.n_coeffs > 0:
            encodedCoeffs = self.coeffEncoder(coeffs) 
            x = x + encodedCoeffs


        for j, dec in enumerate(self.decoder):
            if j > 0:
                x = torch.cat([x, skips[-j]], dim=1)
            x = dec(x)
        x = self.out_proj(x)

        x = self.final_activation(x)
        
        x = torch.clamp(x, self.clipping_output[0], self.clipping_output[1])

        return x