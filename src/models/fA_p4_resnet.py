import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import importlib
from math import sqrt

import src.models.attgconv as attgconv
from src.models.attgconv.attention_layers import fChannelAttention as ch_RnG
from src.models.attgconv.attention_layers import fChannelAttentionGG
from src.models.attgconv.attention_layers import fSpatialAttention
from src.models.attgconv.attention_layers import fSpatialAttentionGG


class fA_P4MResNet(nn.Module):
    def __init__(self, num_blocks=7, nc32=11, nc16=23, nc8=45):
        super().__init__()

        # Group setup
        self.group_name = 'E2'
        self.group = importlib.import_module(f'src.models.attgconv.group.{self.group_name}')
        self.layers = attgconv.layers(self.group)

        # Parameters
        self.n_grid = 8
        self.h_grid = self.layers.H.grid_global(self.n_grid)
        self.kernel_size = 3
        self.padding = 1
        self.stride = 1
        self.eps = 2e-5
        self.wscale = sqrt(2.)
        self.ch_ratio = 16
        self.sp_kernel_size = 7
        self.sp_padding = self.sp_kernel_size // 2

        # Attention modules
        self.ch_GG = functools.partial(fChannelAttentionGG, N_h_in=self.n_grid, group=self.group_name)
        self.sp_RnG = functools.partial(fSpatialAttention, wscale=self.wscale)
        self.sp_GG = functools.partial(fSpatialAttentionGG, group=self.group, input_h_grid=self.h_grid, wscale=self.wscale)

        # Model layers
        self.avg_pooling = self.layers.average_pooling_Rn

        self.c1 = self.layers.fAttConvRnG(
            N_in=3, N_out=nc32, kernel_size=self.kernel_size, h_grid=self.h_grid,
            stride=self.stride, padding=self.padding, wscale=self.wscale,
            channel_attention=ch_RnG(N_in=3, ratio=1),
            spatial_attention=self.sp_RnG(group=self.group, kernel_size=self.sp_kernel_size, h_grid=self.h_grid)
        )

        self.layers_nc32 = self._make_resblock_layer(nc32, nc32, num_blocks)
        self.layers_nc16 = self._make_resblock_layer(nc32, nc16, num_blocks, downsample=True)
        self.layers_nc8  = self._make_resblock_layer(nc16, nc8, num_blocks, downsample=True)

        self.bn_out = nn.BatchNorm3d(nc8, eps=self.eps)
        self.c_out = self.layers.fAttConvGG(
            N_in=nc8, N_out=10, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid,
            stride=1, padding=0, wscale=self.wscale,
            channel_attention=self.ch_GG(N_in=nc8, ratio=nc8 // 2),
            spatial_attention=self.sp_GG(kernel_size=self.sp_kernel_size)
        )

    def _make_resblock_layer(self, in_ch, out_ch, num_blocks, downsample=False):
        blocks = []
        for i in range(num_blocks):
            stride = 2 if downsample and i == 0 else 1
            fiber_map = 'linear' if downsample and i == 0 else 'id'
            blocks.append(fA_P4MResBlock2D(
                in_channels=in_ch if i == 0 else out_ch,
                out_channels=out_ch,
                kernel_size=self.kernel_size,
                fiber_map=fiber_map,
                stride=stride,
                padding=self.padding,
                wscale=self.wscale
            ))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        x = x.flip(-1)
        h = self.c1(x)
        h = self.layers_nc32(h)
        h = self.layers_nc16(h)
        h = self.layers_nc8(h)
        h = self.bn_out(h)
        h = F.relu(h, inplace=True)
        h = self.avg_pooling(h, kernel_size=h.shape[-1], stride=1, padding=0)
        h = self.c_out(h)
        return h.mean(dim=2).view(h.size(0), 10)


class fA_P4MResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, fiber_map='id', stride=1, padding=1, wscale=1.0):
        super().__init__()

        group_name = 'E2'
        group = importlib.import_module(f'src.models.attgconv.group.{group_name}')
        layers = attgconv.layers(group)
        n_grid = 8
        h_grid = layers.H.grid_global(n_grid)

        eps = 2e-5
        sp_kernel_size = 7

        ch_GG = functools.partial(fChannelAttentionGG, N_h_in=n_grid, group=group_name)
        sp_GG = functools.partial(fSpatialAttentionGG, group=group, input_h_grid=h_grid, wscale=wscale)

        self.really_equivariant = stride != 1
        self.pooling = layers.max_pooling_Rn if self.really_equivariant else None

        self.bn1 = nn.BatchNorm3d(in_channels, eps=eps)
        self.bn2 = nn.BatchNorm3d(out_channels, eps=eps)

        conv_stride = 1  # Always use stride=1 in conv, pool if needed
        self.c1 = layers.fAttConvGG(
            N_in=in_channels, N_out=out_channels, kernel_size=kernel_size,
            h_grid=h_grid, input_h_grid=h_grid, stride=conv_stride, padding=padding, wscale=wscale,
            channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
            spatial_attention=sp_GG(kernel_size=sp_kernel_size)
        )

        self.c2 = layers.fAttConvGG(
            N_in=out_channels, N_out=out_channels, kernel_size=kernel_size,
            h_grid=h_grid, input_h_grid=h_grid, stride=1, padding=padding, wscale=wscale,
            channel_attention=ch_GG(N_in=out_channels, ratio=out_channels // 2),
            spatial_attention=sp_GG(kernel_size=sp_kernel_size)
        )

        if fiber_map == 'id':
            if in_channels != out_channels:
                raise ValueError('fiber_map cannot be "id" when channel dimensions differ.')
            self.fiber_map = nn.Identity()
        elif fiber_map == 'linear':
            self.fiber_map = layers.fAttConvGG(
                N_in=in_channels, N_out=out_channels, kernel_size=1,
                h_grid=h_grid, input_h_grid=h_grid, stride=1, padding=0, wscale=wscale,
                channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
                spatial_attention=sp_GG(kernel_size=sp_kernel_size)
            )
        else:
            raise ValueError(f'Unknown fiber_map: {fiber_map}')

    def forward(self, x: torch.Tensor):
        h = self.c1(F.relu(self.bn1(x), inplace=True))
        if self.really_equivariant:
            h = self.pooling(h, kernel_size=2, stride=2, padding=0)

        h = self.c2(F.relu(self.bn2(h), inplace=True))

        hx = self.fiber_map(x)
        if self.really_equivariant:
            hx = self.pooling(hx, kernel_size=2, stride=2, padding=0)

        return hx + h