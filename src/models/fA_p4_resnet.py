
import torch
import torch.nn as nn
from math import sqrt
import functools
import importlib

import src.models.attgconv as attgconv
from src.models.attgconv.attention_layers import fChannelAttention as ch_RnG
from src.models.attgconv.attention_layers import fChannelAttentionGG
from src.models.attgconv.attention_layers import fSpatialAttention
from src.models.attgconv.attention_layers import fSpatialAttentionGG

class fA_P4MResNet(nn.Module):
    def __init__(self, num_blocks=7, nc32=11, nc16=23, nc8=45):

        super(fA_P4MResNet, self).__init__()

        group_name = 'E2'
        group = importlib.import_module('src.models.attgconv.group.' + group_name)

        e2_layers = attgconv.layers(group)

        n_grid = 8
        h_grid = e2_layers.H.grid_global(n_grid)

        stride = 1
        padding = 1
        kernel_size = 3
        eps = 2e-5

        self.group_name = group_name
        self.group = group
        self.layers = e2_layers
        self.n_grid = n_grid
        self.h_grid = h_grid

        wscale = sqrt(2.)
        ch_ratio = 16
        sp_kernel_size = 7
        sp_padding = (sp_kernel_size // 2)

        ch_GG = functools.partial(fChannelAttentionGG, N_h_in=n_grid, group=group_name)
        sp_RnG = functools.partial(fSpatialAttention, wscale=wscale)
        sp_GG = functools.partial(fSpatialAttentionGG, group=group, input_h_grid=self.h_grid, wscale=wscale)

        self.avg_pooling = e2_layers.average_pooling_Rn

        self.c1 = e2_layers.fAttConvRnG(N_in=3, N_out=nc32, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                    channel_attention=ch_RnG(N_in=3, ratio=1),
                                    spatial_attention=sp_RnG(group=group, kernel_size=sp_kernel_size, h_grid=self.h_grid)
                                    )
        layers_nc32 = []
        for i in range(num_blocks):
            layers_nc32.append(fA_P4MResBlock2D(in_channels=nc32, out_channels=nc32, kernel_size=kernel_size, fiber_map='id', stride=stride, padding=padding, wscale=wscale))
        self.layers_nc32 = nn.Sequential(*layers_nc32)

        layers_nc16 = []
        for i in range(num_blocks):
            stride_block = 1 if i > 0 else 2
            fiber_map = 'id' if i > 0 else 'linear'
            nc_in = nc16 if i > 0 else nc32
            layers_nc16.append(fA_P4MResBlock2D(in_channels=nc_in, out_channels=nc16, kernel_size=kernel_size, fiber_map=fiber_map, stride=stride_block, padding=padding, wscale=wscale))
        self.layers_nc16 = nn.Sequential(*layers_nc16)


        layers_nc8 = []
        for i in range(num_blocks):
            stride_block = 1 if i > 0 else 2
            fiber_map = 'id' if i > 0 else 'linear'
            nc_in = nc8 if i > 0 else nc16
            layers_nc8.append(fA_P4MResBlock2D(in_channels=nc_in, out_channels=nc8, kernel_size=kernel_size, fiber_map=fiber_map, stride=stride_block, padding=padding,  wscale=wscale))
        self.layers_nc8 = nn.Sequential(*layers_nc8)


        self.bn_out = nn.BatchNorm3d(num_features=nc8, eps=eps)
        self.c_out = e2_layers.fAttConvGG(N_in=nc8, N_out=10, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=1, padding=0, wscale=wscale,
                                          channel_attention=ch_GG(N_in=nc8, ratio=nc8 // 2),
                                          spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                          )

    def forward(self, x):
        x = torch.flip(x, dims=[-1])
        h = x

        h = self.c1(h)

        h = self.layers_nc32(h)
        h = self.layers_nc16(h)
        h = self.layers_nc8(h)
        h = self.bn_out(h)
        h = torch.relu(h)
        h = self.avg_pooling(h, kernel_size=h.shape[-1], stride=1, padding=0)
        h = self.c_out(h)
        h = h.mean(dim=2)
        h = h.view(h.size(0), 10)
        return h


class fA_P4MResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, fiber_map='id', stride=1, padding=1, wscale=1.0):
        super(fA_P4MResBlock2D, self).__init__()

        assert kernel_size % 2 == 1
        if not padding == (kernel_size - 1) // 2:
            raise NotImplementedError()

        group_name = 'E2'
        group = importlib.import_module('src.models.attgconv.group.' + group_name)
        e2_layers = attgconv.layers(group)
        n_grid = 8
        self.h_grid = e2_layers.H.grid_global(n_grid)  # 2*p4

        eps = 2e-5

        sp_kernel_size = 7
        sp_padding = (sp_kernel_size // 2)

        ch_GG = functools.partial(fChannelAttentionGG, N_h_in=n_grid, group=group_name)
        sp_RnG = functools.partial(fSpatialAttention, wscale=wscale)
        sp_GG = functools.partial(fSpatialAttentionGG, group=group, input_h_grid=self.h_grid, wscale=wscale)

        if stride != 1:
            self.really_equivariant = True
            self.pooling = e2_layers.max_pooling_Rn
        else:
            self.really_equivariant = False

        self.bn1 = nn.BatchNorm3d(num_features=in_channels, eps=eps)
        self.c1 = e2_layers.fAttConvGG(N_in=in_channels , N_out=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                       channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        if self.really_equivariant:
            self.c1 = e2_layers.fAttConvGG(N_in=in_channels, N_out=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=1, padding=padding, wscale=wscale,
                                           channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
                                           spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                           )

        self.bn2 = nn.BatchNorm3d(num_features=out_channels, eps=eps)
        self.c2 = e2_layers.fAttConvGG(N_in=out_channels, N_out=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=1     , padding=padding, wscale=wscale,
                                       channel_attention=ch_GG(N_in=out_channels, ratio=out_channels // 2),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        if fiber_map == 'id':
            if not in_channels == out_channels:
                raise ValueError('fiber_map cannot be identity when channel dimension is changed.')
            self.fiber_map = nn.Sequential() # Identity
        elif fiber_map == 'zero_pad':
            raise NotImplementedError()
        elif fiber_map == 'linear':
            self.fiber_map = e2_layers.fAttConvGG(N_in=in_channels, N_out=out_channels, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=0, wscale=wscale,
                                              channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
                                              spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                              )
            if self.really_equivariant:
                self.fiber_map = e2_layers.fAttConvGG(N_in=in_channels, N_out=out_channels, kernel_size=1, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=1, padding=0, wscale=wscale,
                                                      channel_attention=ch_GG(N_in=in_channels, ratio=in_channels // 2),
                                                      spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                                      )
        else:
            raise ValueError('Unknown fiber_map: ' + str(type))

    def forward(self, x):
        h = self.c1(torch.relu(self.bn1(x)))
        if self.really_equivariant:
            h = self.pooling(h, kernel_size=2, stride=2, padding=0)
        h = self.c2(torch.relu(self.bn2(h)))
        hx = self.fiber_map(x)
        if self.really_equivariant:
            hx = self.pooling(hx, kernel_size=2, stride=2, padding=0)
        return hx + h
