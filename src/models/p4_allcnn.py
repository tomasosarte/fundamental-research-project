import torch
import torch.nn as nn
import functools

class P4AllCNNC(nn.Module):
    def __init__(self, use_bias=False):
        super(P4AllCNNC, self).__init__()
        import importlib
        group_name = 'SE2'
        group = importlib.import_module('src.models.attgconv.group.' + group_name)
        import src.models.attgconv as attgconv
        se2_layers = attgconv.layers(group)
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
        if self.really_equivariant:
            self.pooling = se2_layers.max_pooling_Rn
        self.c1 = se2_layers.ConvRnG(N_in=3          , N_out=N_channels  , kernel_size=kernel_size, h_grid=self.h_grid,                           stride=stride, padding=padding, wscale=wscale)
        self.c2 = se2_layers.ConvGG(N_in=N_channels  , N_out=N_channels  , kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        self.c3 = se2_layers.ConvGG(N_in=N_channels  , N_out=N_channels  , kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=2     , padding=padding, wscale=wscale)
        if self.really_equivariant:
            self.c3 = se2_layers.ConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        self.c4 = se2_layers.ConvGG(N_in=N_channels  , N_out=N_channels_2, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        self.c5 = se2_layers.ConvGG(N_in=N_channels_2, N_out=N_channels_2, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        self.c6 = se2_layers.ConvGG(N_in=N_channels_2, N_out=N_channels_2, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=2     , padding=padding, wscale=wscale)
        if self.really_equivariant:
            self.c6 = se2_layers.ConvGG(N_in=N_channels_2, N_out=N_channels_2, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,  wscale=wscale)
        self.c7 = se2_layers.ConvGG(N_in=N_channels_2, N_out=N_channels_2, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding, wscale=wscale)
        self.c8 = se2_layers.ConvGG(N_in=N_channels_2, N_out=N_channels_2, kernel_size=1          , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=0      , wscale=wscale)
        self.c9 = se2_layers.ConvGG(N_in=N_channels_2, N_out=10          , kernel_size=1          , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=0      , wscale=wscale)
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
        if False:
            from src.models.attgconv.attention_layers import fSpatialAttentionGG
            from src.models.attgconv.attention_layers import fSpatialAttention
            import numpy as np
            import matplotlib.pyplot as plt
            inx = 0
            B = 60
            maps = []
            for m in self.modules():
                if isinstance(m, fSpatialAttention):
                    map = m.att_map.cpu().detach()
                    inx = map.shape[-2]
            upsample = torch.nn.UpsamplingBilinear2d(size=inx)
            for m in self.modules():
                if isinstance(m, fSpatialAttentionGG):
                    map = m.att_map.cpu().detach()
                    map = map.reshape(map.shape[0], 4, map.shape[-2], map.shape[-1])
                    map = upsample(map)
                    map = map.reshape(map.shape[0], 1, 4, map.shape[-2], map.shape[-1])
                    maps.append(map)
            map_0 = maps[0]
            for i in range(len(maps) - 1):
                map_0 = map_0 * maps[i + 1]
            plt.figure()
            plt.imshow(map_0.sum(-3)[B, 0])
            plt.show()
            cmap = plt.cm.jet
            time_samples = 4
            scale = 10
            z = np.zeros([inx, inx])
            plt.figure(dpi=600)
            for t in range(4):
                plt.imshow(map_0.sum(-3)[B, 0])
                if t == 0:
                    plt.quiver(z, map_0[B, 0, t, :, :], color='red', label=r'$0^{\circ}$', scale=scale)
                if t == 2:
                    plt.quiver(z, -map_0[B, 0, t, :, :], color=cmap(t / time_samples), label=r'$180^{\circ}$', scale=scale)
                if t == 1:
                    plt.quiver(-map_0[B, 0, t, :, :], z, color='cyan', label=r'$90^{\circ}$', scale=scale)
                if t == 3:
                    plt.quiver(map_0[B, 0, t, :, :], z, color=cmap(t / time_samples), label=r'$270^{\circ}$',  scale=scale)
            plt.legend(loc='upper right')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        return out