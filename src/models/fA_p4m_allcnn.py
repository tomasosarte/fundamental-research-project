import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import importlib

from src.models.p4m_allcnn import P4MAllCNNC
import src.models.attgconv as attgconv
from src.models.attgconv.group.E2 import H
from src.models.attgconv.attention_layers import fChannelAttention as ch_RnG
from src.models.attgconv.attention_layers import fChannelAttentionGG
from src.models.attgconv.attention_layers import fSpatialAttention
from src.models.attgconv.attention_layers import fSpatialAttentionGG


class fA_P4MAllCNNC(P4MAllCNNC):
    def __init__(self):
        super(fA_P4MAllCNNC, self).__init__()

        group_name = 'E2'
        group = importlib.import_module('src.models.attgconv.group.' + group_name)
        e2_layers = attgconv.layers(importlib.import_module('src.models.attgconv.group.E2'))


        n_grid = 8
        h_grid = e2_layers.H.grid_global(n_grid)
        p_init = 0.2
        p = 0.5
        stride = 1
        padding = 1
        kernel_size = 3
        N_channels = 32
        N_channels_2 = N_channels * 2
        eps = 2e-5
        wscale = 0.035
        ch_ratio = 16
        sp_kernel_size = 7
        sp_padding = (sp_kernel_size // 2)
        self.group_name = group_name
        self.group = group
        self.layers = e2_layers
        self.n_grid = n_grid
        self.h_grid = h_grid
        self.p = p
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.N_channels = N_channels
        self.eps = eps
        self.ch_ratio = ch_ratio
        self.sp_kernel_size = sp_kernel_size
        self.sp_dilation = sp_padding

        ch_GG = functools.partial(fChannelAttentionGG, N_h_in=n_grid, group=group_name)
        sp_RnG = functools.partial(fSpatialAttention, wscale=wscale)
        sp_GG = functools.partial(fSpatialAttentionGG, group=group, input_h_grid=self.h_grid, wscale=wscale)

        self.c1 = e2_layers.fAttConvRnG(N_in=3          , N_out=N_channels  , kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                        channel_attention=ch_RnG(N_in=3        , ratio=1),
                                        spatial_attention=sp_RnG(group=group, kernel_size=sp_kernel_size, h_grid=self.h_grid)
                                        )
        self.c2 = e2_layers.fAttConvGG(N_in=N_channels  , N_out=N_channels  , kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                       channel_attention=ch_GG(N_in=N_channels  , ratio=ch_ratio),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        self.c3 = e2_layers.fAttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                           channel_attention=ch_GG(N_in=N_channels, ratio=ch_ratio),
                                           spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                           )
        self.c4 = e2_layers.fAttConvGG(N_in=N_channels  , N_out=N_channels_2, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                       channel_attention=ch_GG(N_in=N_channels  , ratio=ch_ratio),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        self.c5 = e2_layers.fAttConvGG(N_in=N_channels_2, N_out=N_channels_2, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                       channel_attention=ch_GG(N_in=N_channels_2, ratio=ch_ratio),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        self.c6 = e2_layers.fAttConvGG(N_in=N_channels_2, N_out=N_channels_2, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                           channel_attention=ch_GG(N_in=N_channels_2, ratio=ch_ratio),
                                           spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                           )
        self.c7 = e2_layers.fAttConvGG(N_in=N_channels_2, N_out=N_channels_2, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale,
                                       channel_attention=ch_GG(N_in=N_channels_2, ratio=ch_ratio),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        self.c8 = e2_layers.fAttConvGG(N_in=N_channels_2, N_out=N_channels_2, kernel_size=1          , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=0      , wscale=wscale,
                                       channel_attention=ch_GG(N_in=N_channels_2, ratio=ch_ratio),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )
        self.c9 = e2_layers.fAttConvGG(N_in=N_channels_2, N_out=10          , kernel_size=1          , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=0      , wscale=wscale,
                                       channel_attention=ch_GG(N_in=N_channels_2, ratio=ch_ratio),
                                       spatial_attention=sp_GG(kernel_size=sp_kernel_size)
                                       )

        self.dp_init = nn.Dropout(p_init)
        self.dp = nn.Dropout(p)

        self.bn1 = nn.BatchNorm3d(num_features=N_channels  , eps=eps)
        self.bn2 = nn.BatchNorm3d(num_features=N_channels  , eps=eps)
        self.bn3 = nn.BatchNorm3d(num_features=N_channels  , eps=eps)
        self.bn4 = nn.BatchNorm3d(num_features=N_channels_2, eps=eps)
        self.bn5 = nn.BatchNorm3d(num_features=N_channels_2, eps=eps)
        self.bn6 = nn.BatchNorm3d(num_features=N_channels_2, eps=eps)
        self.bn7 = nn.BatchNorm3d(num_features=N_channels_2, eps=eps)
        self.bn8 = nn.BatchNorm3d(num_features=N_channels_2, eps=eps)
        self.bn9 = nn.BatchNorm3d(num_features=10, eps=eps)
