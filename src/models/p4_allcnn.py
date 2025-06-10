import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

import src.models.attgconv as attgconv
from src.models.attgconv.group.SE2 import H

class P4AllCNNC(nn.Module):
    def __init__(self):
        super(P4AllCNNC, self).__init__()

        se2_layers = attgconv.layers(importlib.import_module('src.models.attgconv.group.SE2'))

        self.h_grid = se2_layers.H.grid_global(4)
        p_init = 0.2
        p = 0.5
        stride = 1
        padding = 1
        kernel_size = 3
        N_channels = 48
        N_channels_2 = N_channels * 2
        eps = 2e-5
        wscale = 0.035
        self.really_equivariant = True

        # Convolution layers
        self.pooling = se2_layers.max_pooling_Rn
        self.c1 = se2_layers.ConvRnG(N_in=3          , N_out=N_channels  , kernel_size=kernel_size, h_grid=self.h_grid,                           stride=stride, padding=padding, wscale=wscale)
        self.c2 = se2_layers.ConvGG(N_in=N_channels  , N_out=N_channels  , kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        self.c3 = se2_layers.ConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        self.c4 = se2_layers.ConvGG(N_in=N_channels  , N_out=N_channels_2, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        self.c5 = se2_layers.ConvGG(N_in=N_channels_2, N_out=N_channels_2, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        self.c6 = se2_layers.ConvGG(N_in=N_channels_2, N_out=N_channels_2, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,  wscale=wscale)
        self.c7 = se2_layers.ConvGG(N_in=N_channels_2, N_out=N_channels_2, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        self.c8 = se2_layers.ConvGG(N_in=N_channels_2, N_out=N_channels_2, kernel_size=1          , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=0      , wscale=wscale)
        self.c9 = se2_layers.ConvGG(N_in=N_channels_2, N_out=10          , kernel_size=1          , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=0      , wscale=wscale)

        # Dropouts
        self.dp_init = nn.Dropout(p_init)
        self.dp = nn.Dropout(p)

        # BatchNorms
        self.bn1 = nn.BatchNorm3d(num_features=N_channels  , eps=eps)
        self.bn2 = nn.BatchNorm3d(num_features=N_channels  , eps=eps)
        self.bn3 = nn.BatchNorm3d(num_features=N_channels  , eps=eps)
        self.bn4 = nn.BatchNorm3d(num_features=N_channels_2, eps=eps)
        self.bn5 = nn.BatchNorm3d(num_features=N_channels_2, eps=eps)
        self.bn6 = nn.BatchNorm3d(num_features=N_channels_2, eps=eps)
        self.bn7 = nn.BatchNorm3d(num_features=N_channels_2, eps=eps)
        self.bn8 = nn.BatchNorm3d(num_features=N_channels_2, eps=eps)
        self.bn9 = nn.BatchNorm3d(num_features=10, eps=eps)

    def forward(self, x):

        out = torch.relu(self.bn1(self.c1(self.dp_init(x))))
        out = torch.relu(self.bn2(self.c2(out)))

        if self.really_equivariant:
            out = self.c3(out)
            out = self.pooling(out, kernel_size=2, stride=2, padding=0)
            out = self.dp(torch.relu(self.bn3(out)))
        else:
            out = self.dp(torch.relu(self.bn3(self.c3(out))))

        out = torch.relu(self.bn4(self.c4(out)))
        out = torch.relu(self.bn5(self.c5(out)))

        if self.really_equivariant:
            out = self.c6(out)
            out = self.pooling(out, kernel_size=2, stride=2, padding=0)
            out = self.dp(torch.relu(self.bn6(out)))
        else:
            out = self.dp(torch.relu(self.bn6(self.c6(out))))

        out = torch.relu(self.bn7(self.c7(out)))
        out = torch.relu(self.bn8(self.c8(out)))
        out = torch.relu(self.bn9(self.c9(out)))
        out = torch.nn.functional.avg_pool3d(out, out.size()[2:]).squeeze()

        return out