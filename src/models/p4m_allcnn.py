import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

import src.models.attgconv as attgconv


class P4MAllCNNC(nn.Module):
    def __init__(self, n_channels: int = 32):
        super().__init__()

        # === Config ===
        self.cfg = {
            "group_name": "E2",
            "n_grid": 8,
            "p_init": 0.2,
            "p": 0.5,
            "stride": 1,
            "padding": 1,
            "kernel_size": 3,
            "N_channels": n_channels,
            "eps": 2e-5,
            "wscale": 0.035,
        }
        cfg = self.cfg
        self.really_equivariant = True

        # === Group & Layers ===
        group_mod = importlib.import_module(f"src.models.attgconv.group.{cfg['group_name']}")
        layers = attgconv.layers(group_mod)
        self.pooling = layers.max_pooling_Rn
        self.h_grid = layers.H.grid_global(cfg["n_grid"])

        N  = cfg["N_channels"]
        N2 = 2 * N

        # === Convolutions ===
        self.c1 = layers.ConvRnG(
            N_in=3, N_out=N, kernel_size=cfg["kernel_size"],
            h_grid=self.h_grid, stride=cfg["stride"],
            padding=cfg["padding"], wscale=cfg["wscale"]
        )

        self.c2 = self._make_conv(layers.ConvGG, N,  N,  cfg)
        self.c3 = self._make_conv(layers.ConvGG, N,  N,  cfg)
        self.c4 = self._make_conv(layers.ConvGG, N,  N2, cfg)
        self.c5 = self._make_conv(layers.ConvGG, N2, N2, cfg)
        self.c6 = self._make_conv(layers.ConvGG, N2, N2, cfg)
        self.c7 = self._make_conv(layers.ConvGG, N2, N2, cfg)
        self.c8 = self._make_conv(layers.ConvGG, N2, N2, cfg, kernel_size=1, padding=0)
        self.c9 = self._make_conv(layers.ConvGG, N2, 10, cfg, kernel_size=1, padding=0)

        # === Dropouts ===
        self.dp_init = nn.Dropout(cfg["p_init"])
        self.dp      = nn.Dropout(cfg["p"])

        # === BatchNorms ===
        self.bn1 = nn.BatchNorm3d(N,  eps=cfg["eps"])
        self.bn2 = nn.BatchNorm3d(N,  eps=cfg["eps"])
        self.bn3 = nn.BatchNorm3d(N,  eps=cfg["eps"])
        self.bn4 = nn.BatchNorm3d(N2, eps=cfg["eps"])
        self.bn5 = nn.BatchNorm3d(N2, eps=cfg["eps"])
        self.bn6 = nn.BatchNorm3d(N2, eps=cfg["eps"])
        self.bn7 = nn.BatchNorm3d(N2, eps=cfg["eps"])
        self.bn8 = nn.BatchNorm3d(N2, eps=cfg["eps"])
        self.bn9 = nn.BatchNorm3d(10, eps=cfg["eps"])

    def _make_conv(self, conv_cls, in_ch, out_ch, cfg, kernel_size=None, padding=None):
        k = cfg["kernel_size"] if kernel_size is None else kernel_size
        p = cfg["padding"]     if padding     is None else padding
        return conv_cls(
            N_in=in_ch, N_out=out_ch, kernel_size=k,
            h_grid=self.h_grid, input_h_grid=self.h_grid,
            stride=cfg["stride"], padding=p, wscale=cfg["wscale"]
        )

    def forward(self, x):
        out = self.dp_init(x)
        out = F.relu(self.bn1(self.c1(out)), inplace=True)
        out = F.relu(self.bn2(self.c2(out)), inplace=True)

        out = self._maybe_pool(self.c3, self.bn3, out)
        out = F.relu(self.bn4(self.c4(out)), inplace=True)
        out = F.relu(self.bn5(self.c5(out)), inplace=True)
        out = self._maybe_pool(self.c6, self.bn6, out)

        out = F.relu(self.bn7(self.c7(out)), inplace=True)
        out = F.relu(self.bn8(self.c8(out)), inplace=True)
        out = F.relu(self.bn9(self.c9(out)), inplace=True)

        out = nn.functional.avg_pool3d(out, kernel_size=out.shape[2:]).squeeze()
        return out

    def _maybe_pool(self, conv, bn, x):
        h = conv(x)
        if self.really_equivariant:
            h = self.pooling(h, kernel_size=2, stride=2, padding=0)
        return self.dp(F.relu(bn(h), inplace=True))
