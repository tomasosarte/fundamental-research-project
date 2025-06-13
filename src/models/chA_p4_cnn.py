import torch
import torch.nn as nn
import importlib
import functools

import src.models.attgconv as attgconv
from src.models.attgconv.attention_layers import (
    ChannelAttention as ch_RnG,
    ChannelAttentionGG,
    SpatialAttention,
    SpatialAttentionGG,
)

class A_Ch_P4CNN(nn.Module):
    def __init__(self, n_channels: int = 10):
        super().__init__()

        # === Configuration ===
        self.cfg = {
            "n_grid": 4,
            "p": 0.3,
            "stride": 1,
            "padding": 0,
            "kernel_size": 3,
            "N_channels": n_channels,
            "eps": 2e-5,
            "ch_ratio": 2,
            "sp_kernel_size": 7,
        }
        cfg = self.cfg

        # === Group structure and layers ===
        se2_group = importlib.import_module("src.models.attgconv.group.SE2")
        self.layers = attgconv.layers(se2_group)
        self.pooling = self.layers.max_pooling_Rn
        self.h_grid = self.layers.H.grid_global(cfg["n_grid"])

        # === Attention definitions ===
        ch_GG = functools.partial(ChannelAttentionGG, N_h=cfg["n_grid"], N_h_in=cfg["n_grid"])

        # === Network layers ===
        self.c1 = self._make_conv(self.layers.AttConvRnG, 1, n_channels, ch_att=ch_RnG(N_out=n_channels, N_in=1, ratio=1))
        self.c2 = self._make_conv(self.layers.AttConvGG, n_channels, n_channels, ch_att=ch_GG(N_out=n_channels, N_in=n_channels, ratio=cfg["ch_ratio"]))
        self.c3 = self._make_conv(self.layers.AttConvGG, n_channels, n_channels, ch_att=ch_GG(N_out=n_channels, N_in=n_channels, ratio=cfg["ch_ratio"]))
        self.c4 = self._make_conv(self.layers.AttConvGG, n_channels, n_channels, ch_att=ch_GG(N_out=n_channels, N_in=n_channels, ratio=cfg["ch_ratio"]))
        self.c5 = self._make_conv(self.layers.AttConvGG, n_channels, n_channels, ch_att=ch_GG(N_out=n_channels, N_in=n_channels, ratio=cfg["ch_ratio"]))
        self.c6 = self._make_conv(self.layers.AttConvGG, n_channels, n_channels, ch_att=ch_GG(N_out=n_channels, N_in=n_channels, ratio=cfg["ch_ratio"]))
        self.c7 = self._make_conv(self.layers.AttConvGG, n_channels, 10, ch_att=ch_GG(N_out=10, N_in=n_channels, ratio=cfg["ch_ratio"]))

        # === BatchNorm and Dropout ===
        self.bn = nn.ModuleList([nn.BatchNorm3d(n_channels, eps=cfg["eps"]) for _ in range(6)])
        self.dp = nn.Dropout(cfg["p"])

    def _make_conv(self, conv_cls, in_ch, out_ch, ch_att=None, sp_att=None):
        cfg = self.cfg
        kwargs = dict(
            N_in=in_ch,
            N_out=out_ch,
            kernel_size=cfg["kernel_size"],
            h_grid=self.h_grid,
            stride=cfg["stride"],
            padding=cfg["padding"],
        )
        if conv_cls.__name__.endswith("GG"):
            kwargs["input_h_grid"] = self.h_grid
        if ch_att is not None:
            kwargs["channel_attention"] = ch_att
        if sp_att is not None:
            kwargs["spatial_attention"] = sp_att
        return conv_cls(**kwargs)

    def forward(self, x):
        out = self.dp(torch.relu(self.bn[0](self.c1(x))))
        out = torch.relu(self.bn[1](self.c2(out)))
        out = self.pooling(out, kernel_size=2, stride=2, padding=0)

        out = self.dp(torch.relu(self.bn[2](self.c3(out))))
        out = self.dp(torch.relu(self.bn[3](self.c4(out))))
        out = self.dp(torch.relu(self.bn[4](self.c5(out))))
        out = self.dp(torch.relu(self.bn[5](self.c6(out))))

        out = self.c7(out)
        out, _ = torch.max(out, dim=-3)  # Reduce group dim
        out = torch.nn.functional.avg_pool2d(out, kernel_size=out.shape[-2:])  # Reduce spatial dims
        out = out.view(out.size(0), 10)
        return out
