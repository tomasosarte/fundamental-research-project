import torch
import torch.nn as nn
import functools
import importlib

from src.models.p4_allcnn import P4AllCNNC
import src.models.attgconv as attgconv
from src.models.attgconv.attention_layers import fChannelAttention as ch_RnG
from src.models.attgconv.attention_layers import fChannelAttentionGG
from src.models.attgconv.attention_layers import fSpatialAttention
from src.models.attgconv.attention_layers import fSpatialAttentionGG


class fA_P4AllCNNC(P4AllCNNC):
    def __init__(self):
        super(fA_P4AllCNNC, self).__init__()

        # --- Configuration ---
        self.group_name = 'SE2'
        self.group = importlib.import_module(f'src.models.attgconv.group.{self.group_name}')
        self.layers = attgconv.layers(self.group)
        self.n_grid = 4
        self.h_grid = self.layers.H.grid_global(self.n_grid)

        # Hyperparameters
        self.kernel_size = 3
        self.padding = 1
        self.stride = 1
        self.N_channels = 48
        self.N_channels_2 = self.N_channels * 2
        self.ch_ratio = 16
        self.sp_kernel_size = 7
        self.sp_padding = self.sp_kernel_size // 2
        self.eps = 2e-5
        self.wscale = 0.035
        self.p = 0.5
        self.p_init = 0.2

        # --- Attention Factories ---
        self.ch_GG = functools.partial(fChannelAttentionGG, N_h_in=self.n_grid)
        self.sp_RnG = functools.partial(fSpatialAttention, group=self.group, kernel_size=self.sp_kernel_size, h_grid=self.h_grid, wscale=self.wscale)
        self.sp_GG = functools.partial(fSpatialAttentionGG, group=self.group, input_h_grid=self.h_grid, wscale=self.wscale)

        # --- Network Layers ---
        self._build_layers()

    def _build_layers(self):
        self.c1 = self.layers.fAttConvRnG(
            N_in=3, N_out=self.N_channels, kernel_size=self.kernel_size, h_grid=self.h_grid,
            stride=self.stride, padding=self.padding, wscale=self.wscale,
            channel_attention=ch_RnG(N_in=3, ratio=1),
            spatial_attention=self.sp_RnG()
        )

        def convGG(i, o, ks=3, pad=None):
            return self.layers.fAttConvGG(
                N_in=i, N_out=o, kernel_size=ks,
                h_grid=self.h_grid, input_h_grid=self.h_grid,
                stride=self.stride, padding=self.padding if pad is None else pad, wscale=self.wscale,
                channel_attention=self.ch_GG(N_in=i, ratio=self.ch_ratio),
                spatial_attention=self.sp_GG(kernel_size=self.sp_kernel_size)
            )

        self.c2 = convGG(self.N_channels, self.N_channels)
        self.c3 = convGG(self.N_channels, self.N_channels)
        self.c4 = convGG(self.N_channels, self.N_channels_2)
        self.c5 = convGG(self.N_channels_2, self.N_channels_2)
        self.c6 = convGG(self.N_channels_2, self.N_channels_2)
        self.c7 = convGG(self.N_channels_2, self.N_channels_2)
        self.c8 = convGG(self.N_channels_2, self.N_channels_2, ks=1, pad=0)
        self.c9 = convGG(self.N_channels_2, 10, ks=1, pad=0)

        self.dp_init = nn.Dropout(self.p_init)
        self.dp = nn.Dropout(self.p)

        # BatchNorm layers
        self.bn1 = nn.BatchNorm3d(self.N_channels, eps=self.eps)
        self.bn2 = nn.BatchNorm3d(self.N_channels, eps=self.eps)
        self.bn3 = nn.BatchNorm3d(self.N_channels, eps=self.eps)
        self.bn4 = nn.BatchNorm3d(self.N_channels_2, eps=self.eps)
        self.bn5 = nn.BatchNorm3d(self.N_channels_2, eps=self.eps)
        self.bn6 = nn.BatchNorm3d(self.N_channels_2, eps=self.eps)
        self.bn7 = nn.BatchNorm3d(self.N_channels_2, eps=self.eps)
        self.bn8 = nn.BatchNorm3d(self.N_channels_2, eps=self.eps)
        self.bn9 = nn.BatchNorm3d(10, eps=self.eps)