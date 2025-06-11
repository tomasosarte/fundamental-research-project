import torch
import torch.nn as nn
import importlib
import functools

import src.models.attgconv as attgconv
from src.models.attgconv.attention_layers import SpatialAttention, SpatialAttentionGG


class A_Sp_P4CNN(nn.Module):
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
            "sp_kernel_size": 7,
        }
        cfg = self.cfg
        self.really_equivariant = True

        # === Layers and group ===
        se2_group = importlib.import_module("src.models.attgconv.group.SE2")
        self.layers = attgconv.layers(se2_group)
        self.pooling = self.layers.max_pooling_Rn
        self.h_grid = self.layers.H.grid_global(cfg["n_grid"])

        sp_RnG = functools.partial(
            SpatialAttention,
            group=se2_group,
            h_grid=self.h_grid,
            stride=cfg["stride"],
        )
        sp_GG = functools.partial(
            SpatialAttentionGG,
            group=se2_group,
            h_grid=self.h_grid,
            input_h_grid=self.h_grid,
            stride=cfg["stride"],
        )

        # === Convolutional blocks (Spatial Attention only) ===
        self.c1 = self._make_conv(self.layers.AttConvRnG, 1, n_channels, sp_att=sp_RnG(N_out=n_channels, N_in=1, kernel_size=cfg["sp_kernel_size"]))
        self.c2 = self._make_conv(self.layers.AttConvGG, n_channels, n_channels, sp_att=sp_GG(N_out=n_channels, N_in=n_channels, kernel_size=cfg["sp_kernel_size"]))
        self.c3 = self._make_conv(self.layers.AttConvGG, n_channels, n_channels, sp_att=sp_GG(N_out=n_channels, N_in=n_channels, kernel_size=cfg["sp_kernel_size"]))
        self.c4 = self._make_conv(self.layers.AttConvGG, n_channels, n_channels, sp_att=sp_GG(N_out=n_channels, N_in=n_channels, kernel_size=cfg["sp_kernel_size"]))
        self.c5 = self._make_conv(self.layers.AttConvGG, n_channels, n_channels, sp_att=sp_GG(N_out=n_channels, N_in=n_channels, kernel_size=cfg["sp_kernel_size"]))
        self.c6 = self._make_conv(self.layers.AttConvGG, n_channels, n_channels, sp_att=sp_GG(N_out=n_channels, N_in=n_channels, kernel_size=cfg["sp_kernel_size"]))
        self.c7 = self._make_conv(self.layers.AttConvGG, n_channels, 10, sp_att=sp_GG(N_out=10, N_in=n_channels, kernel_size=cfg["sp_kernel_size"]))

        # === Dropout ===
        self.dp = nn.Dropout(cfg["p"])

        # === BatchNorms ===
        self.bn1 = nn.BatchNorm3d(n_channels, eps=cfg["eps"])
        self.bn2 = nn.BatchNorm3d(n_channels, eps=cfg["eps"])
        self.bn3 = nn.BatchNorm3d(n_channels, eps=cfg["eps"])
        self.bn4 = nn.BatchNorm3d(n_channels, eps=cfg["eps"])
        self.bn5 = nn.BatchNorm3d(n_channels, eps=cfg["eps"])
        self.bn6 = nn.BatchNorm3d(n_channels, eps=cfg["eps"])

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
        out = self.dp(torch.relu(self.bn1(self.c1(x))))
        out = torch.relu(self.bn2(self.c2(out)))
        out = self.pooling(out, kernel_size=2, stride=2, padding=0)

        out = self.dp(torch.relu(self.bn3(self.c3(out))))
        out = self.dp(torch.relu(self.bn4(self.c4(out))))
        out = self.dp(torch.relu(self.bn5(self.c5(out))))
        out = self.dp(torch.relu(self.bn6(self.c6(out))))

        out = self.c7(out)
        out, _ = torch.max(out, dim=-3)
        out = nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), 10)
        return out
