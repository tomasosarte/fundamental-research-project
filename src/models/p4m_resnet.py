import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from math import sqrt

import src.models.attgconv as attgconv

# —————————————————————————————————————————————————————————————
# Module­-level setup
# —————————————————————————————————————————————————————————————
_group     = importlib.import_module("src.models.attgconv.group.E2")
_layers    = attgconv.layers(_group)
_h_grid    = _layers.H.grid_global(8)
_POOL_Rn   = _layers.max_pooling_Rn
_AVG_Rn    = _layers.average_pooling_Rn
# Shared hyperparameters
_EPS       = 2e-5
_KSIZE     = 3
_PADDING   = 1
_WSCALE    = sqrt(2.)


# —————————————————————————————————————————————————————————————
# Equivariant residual block
# —————————————————————————————————————————————————————————————
class P4MResBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = _KSIZE,
        fiber_map: str = "id",
        stride: int = 1,
        padding: int = _PADDING,
        wscale: float = _WSCALE,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        assert padding == (kernel_size - 1) // 2

        # determine if we downsample by pooling
        self.really_equivariant = (stride != 1)
        self.pool = _POOL_Rn if self.really_equivariant else None

        # BatchNorms
        self.bn1 = nn.BatchNorm3d(in_channels,  eps=_EPS)
        self.bn2 = nn.BatchNorm3d(out_channels, eps=_EPS)

        # 1×1 skip / identity mapping
        if fiber_map == "id":
            if in_channels != out_channels:
                raise ValueError("Cannot use id fiber_map when in != out")
            self.skip = nn.Identity()
        elif fiber_map == "linear":
            skip_stride = 1  # always do stride=1 on the conv, pooling handles downsample
            self.skip = _layers.ConvGG(
                N_in=in_channels,
                N_out=out_channels,
                kernel_size=1,
                h_grid=_h_grid,
                input_h_grid=_h_grid,
                stride=skip_stride,
                padding=0,
                wscale=wscale,
            )
        else:
            raise ValueError(f"Unknown fiber_map: {fiber_map}")

        # first 3×3 conv (always stride=1 at conv-level)
        self.conv1 = _layers.ConvGG(
            N_in=in_channels,
            N_out=out_channels,
            kernel_size=kernel_size,
            h_grid=_h_grid,
            input_h_grid=_h_grid,
            stride=1,
            padding=padding,
            wscale=wscale,
        )

        # second 3×3 conv
        self.conv2 = _layers.ConvGG(
            N_in=out_channels,
            N_out=out_channels,
            kernel_size=kernel_size,
            h_grid=_h_grid,
            input_h_grid=_h_grid,
            stride=1,
            padding=padding,
            wscale=wscale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv path
        h = F.relu(self.bn1(x), inplace=True)
        h = self.conv1(h)
        if self.really_equivariant:
            h = self.pool(h, kernel_size=2, stride=2, padding=0)

        h = F.relu(self.bn2(h), inplace=True)
        h = self.conv2(h)

        # skip path
        skip = self.skip(x)
        if self.really_equivariant:
            skip = self.pool(skip, kernel_size=2, stride=2, padding=0)

        return skip + h


# —————————————————————————————————————————————————————————————
# Full ResNet‐style network
# —————————————————————————————————————————————————————————————
class P4MResNet(nn.Module):
    def __init__(self, num_blocks=7, nc32=11, nc16=23, nc8=45):
        super().__init__()

        # first “ConvRnG” layer
        self.c1 = _layers.ConvRnG(
            N_in=3,
            N_out=nc32,
            kernel_size=_KSIZE,
            h_grid=_h_grid,
            stride=1,
            padding=_PADDING,
            wscale=_WSCALE,
        )

        # three stages of residual blocks
        self.stage32 = self._make_stage(nc32, nc32, num_blocks, downsample=False)
        self.stage16 = self._make_stage(nc32, nc16, num_blocks, downsample=True)
        self.stage8  = self._make_stage(nc16, nc8,  num_blocks, downsample=True)

        # final batchnorm + 1×1 conv
        self.bn_out = nn.BatchNorm3d(nc8, eps=_EPS)
        self.c_out  = _layers.ConvGG(
            N_in=nc8,
            N_out=10,
            kernel_size=1,
            h_grid=_h_grid,
            input_h_grid=_h_grid,
            stride=1,
            padding=0,
            wscale=_WSCALE,
        )

        self.global_pool = lambda x: _AVG_Rn(x, kernel_size=x.shape[-1], stride=1, padding=0)

    def _make_stage(self, in_ch, out_ch, n_blocks, downsample: bool):
        blocks = []
        for i in range(n_blocks):
            stride    = 2 if downsample and i == 0 else 1
            fiber_map = "linear" if downsample and i == 0 else "id"
            nc_in     = in_ch if i == 0 else out_ch
            blocks.append(
                P4MResBlock2D(
                    in_channels=nc_in,
                    out_channels=out_ch,
                    kernel_size=_KSIZE,
                    fiber_map=fiber_map,
                    stride=stride,
                    padding=_PADDING,
                    wscale=_WSCALE,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(x)
        h = self.stage32(h)
        h = self.stage16(h)
        h = self.stage8(h)
        h = F.relu(self.bn_out(h), inplace=True)
        h = self.global_pool(h)
        h = self.c_out(h)
        # collapse the fiber‐dimension
        return h.mean(dim=2).view(h.size(0), 10)