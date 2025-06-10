import torch
import torch.nn as nn
import functools
import importlib

from src.models.p4m_allcnn import P4MAllCNNC
import src.models.attgconv as attgconv
from src.models.attgconv.attention_layers import fChannelAttention as ch_RnG
from src.models.attgconv.attention_layers import fChannelAttentionGG
from src.models.attgconv.attention_layers import fSpatialAttention
from src.models.attgconv.attention_layers import fSpatialAttentionGG


class fA_P4MAllCNNC(P4MAllCNNC):
    def __init__(self):
        super().__init__()

        # === Config ===
        self.config = {
            "group_name": "E2",
            "n_grid": 8,
            "p_init": 0.2,
            "p": 0.5,
            "stride": 1,
            "padding": 1,
            "kernel_size": 3,
            "N_channels": 32,
            "eps": 2e-5,
            "wscale": 0.035,
            "ch_ratio": 16,
            "sp_kernel_size": 7,
        }
        cfg = self.config

        # === Group and Attention Setup ===
        group = importlib.import_module(f'src.models.attgconv.group.{cfg["group_name"]}')
        layers = attgconv.layers(group)
        h_grid = layers.H.grid_global(cfg["n_grid"])

        ch_GG = functools.partial(fChannelAttentionGG, N_h_in=cfg["n_grid"], group=cfg["group_name"])
        sp_RnG = functools.partial(fSpatialAttention, wscale=cfg["wscale"])
        sp_GG = functools.partial(fSpatialAttentionGG, group=group, input_h_grid=h_grid, wscale=cfg["wscale"])

        # === Save components ===
        self.group_name = cfg["group_name"]
        self.group = group
        self.layers = layers
        self.h_grid = h_grid
        self.sp_dilation = cfg["sp_kernel_size"] // 2

        # === Layer Definitions ===
        N = cfg["N_channels"]
        N2 = 2 * N

        self.c1 = layers.fAttConvRnG(
            N_in=3, N_out=N, kernel_size=cfg["kernel_size"], h_grid=h_grid, stride=cfg["stride"], padding=cfg["padding"], wscale=cfg["wscale"],
            channel_attention=ch_RnG(N_in=3, ratio=1),
            spatial_attention=sp_RnG(group=group, kernel_size=cfg["sp_kernel_size"], h_grid=h_grid)
        )

        # Layers c2 to c9
        self.c2 = self._make_conv_block(N,  N,  cfg, ch_GG, sp_GG)
        self.c3 = self._make_conv_block(N,  N,  cfg, ch_GG, sp_GG)
        self.c4 = self._make_conv_block(N,  N2, cfg, ch_GG, sp_GG)
        self.c5 = self._make_conv_block(N2, N2, cfg, ch_GG, sp_GG)
        self.c6 = self._make_conv_block(N2, N2, cfg, ch_GG, sp_GG)
        self.c7 = self._make_conv_block(N2, N2, cfg, ch_GG, sp_GG)
        self.c8 = self._make_conv_block(N2, N2, cfg, ch_GG, sp_GG, kernel_size=1, padding=0)
        self.c9 = self._make_conv_block(N2, 10, cfg, ch_GG, sp_GG, kernel_size=1, padding=0)

        # === Dropout ===
        self.dp_init = nn.Dropout(cfg["p_init"])
        self.dp = nn.Dropout(cfg["p"])

        # === BatchNorms ===
        self.bn1 = nn.BatchNorm3d(N, eps=cfg["eps"])
        self.bn2 = nn.BatchNorm3d(N, eps=cfg["eps"])
        self.bn3 = nn.BatchNorm3d(N, eps=cfg["eps"])
        self.bn4 = nn.BatchNorm3d(N2, eps=cfg["eps"])
        self.bn5 = nn.BatchNorm3d(N2, eps=cfg["eps"])
        self.bn6 = nn.BatchNorm3d(N2, eps=cfg["eps"])
        self.bn7 = nn.BatchNorm3d(N2, eps=cfg["eps"])
        self.bn8 = nn.BatchNorm3d(N2, eps=cfg["eps"])
        self.bn9 = nn.BatchNorm3d(10, eps=cfg["eps"])

    def _make_conv_block(self, in_ch, out_ch, cfg, ch_GG, sp_GG, kernel_size=None, padding=None):
        """Helper to make a SE(2)-equivariant attention convolution block."""
        k = cfg["kernel_size"] if kernel_size is None else kernel_size
        p = cfg["padding"] if padding is None else padding
        return self.layers.fAttConvGG(
            N_in=in_ch, N_out=out_ch, kernel_size=k,
            h_grid=self.h_grid, input_h_grid=self.h_grid,
            stride=cfg["stride"], padding=p, wscale=cfg["wscale"],
            channel_attention=ch_GG(N_in=in_ch, ratio=cfg["ch_ratio"]),
            spatial_attention=sp_GG(kernel_size=cfg["sp_kernel_size"])
        )