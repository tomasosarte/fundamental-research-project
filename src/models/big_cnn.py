import torch
import torch.nn as nn
import functools
import importlib


class B11_P4CNN(nn.Module):
    def __init__(self, n_channels: int = 10, use_bias: bool = False):
        super(B11_P4CNN, self).__init__()

        # --- Configuration ---
        self.group_name = 'SE2'
        self.group = importlib.import_module(f'src.models.attgconv.group.{self.group_name}')
        import src.models.attgconv as attgconv
        self.layers = attgconv.layers(self.group)

        self.n_grid = 4
        self.h_grid = self.layers.H.grid_global(self.n_grid)

        # Hyperparameters
        self.kernel_size = 3
        self.padding = 0
        self.stride = 1
        self.N_channels = n_channels
        self.eps = 2e-5
        self.p = 0.3

        # --- Network Layers ---
        self._build_layers()

    def _build_layers(self):
        self.c1 = self.layers.ConvRnG(
            N_in=1, N_out=self.N_channels, kernel_size=self.kernel_size,
            h_grid=self.h_grid, stride=self.stride, padding=self.padding
        )

        def convGG():
            return self.layers.ConvGG(
                N_in=self.N_channels, N_out=self.N_channels, kernel_size=self.kernel_size,
                h_grid=self.h_grid, input_h_grid=self.h_grid,
                stride=self.stride, padding=self.padding
            )

        self.c2 = convGG()
        self.c3 = convGG()
        self.c4 = convGG()
        self.c5 = convGG()
        self.c6 = convGG()

        self.c7 = self.layers.ConvGG(
            N_in=self.N_channels, N_out=10, kernel_size=4,
            h_grid=self.h_grid, input_h_grid=self.h_grid,
            stride=self.stride, padding=self.padding
        )

        # Dropout & Normalization
        self.dp = nn.Dropout(self.p)

        self.bn1 = nn.BatchNorm3d(self.N_channels, eps=self.eps)
        self.bn2 = nn.BatchNorm3d(self.N_channels, eps=self.eps)
        self.bn3 = nn.BatchNorm3d(self.N_channels, eps=self.eps)
        self.bn4 = nn.BatchNorm3d(self.N_channels, eps=self.eps)
        self.bn5 = nn.BatchNorm3d(self.N_channels, eps=self.eps)
        self.bn6 = nn.BatchNorm3d(self.N_channels, eps=self.eps)

        self.max_pooling = self.layers.max_pooling_Rn

    def forward(self, x):
        out = self.dp(torch.relu(self.bn1(self.c1(x))))
        out = torch.relu(self.bn2(self.c2(out)))
        out = self.max_pooling(out, kernel_size=2, stride=2, padding=0)

        out = self.dp(torch.relu(self.bn3(self.c3(out))))
        out = self.dp(torch.relu(self.bn4(self.c4(out))))
        out = self.dp(torch.relu(self.bn5(self.c5(out))))
        out = self.dp(torch.relu(self.bn6(self.c6(out))))

        out = self.c7(out)
        out, _ = torch.max(out, dim=-3)
        out = out.view(out.size(0), 10)
        return out



class B15_P4CNN(nn.Module):
    def __init__(self, n_channels: int = 10, use_bias: bool = False):
        super(B15_P4CNN, self).__init__()

        # --- Configuration ---
        self.group_name = 'SE2'
        self.group = importlib.import_module(f'src.models.attgconv.group.{self.group_name}')
        import src.models.attgconv as attgconv
        self.layers = attgconv.layers(self.group)

        self.n_grid = 4
        self.h_grid = self.layers.H.grid_global(self.n_grid)

        # Hyperparameters
        self.kernel_size = 3
        self.padding = 1
        self.stride = 1
        self.N_channels = n_channels
        self.eps = 2e-5
        self.p = 0.3

        # --- Network Layers ---
        self._build_layers()

    def _build_layers(self):
        self.c1 = self.layers.ConvRnG(
            N_in=1, N_out=self.N_channels, kernel_size=self.kernel_size,
            h_grid=self.h_grid, stride=self.stride, padding=self.padding
        )

        def convGG():
            return self.layers.ConvGG(
                N_in=self.N_channels, N_out=self.N_channels, kernel_size=self.kernel_size,
                h_grid=self.h_grid, input_h_grid=self.h_grid,
                stride=self.stride, padding=self.padding
            )

        # BIG15: 10 intermediate ConvGG layers before the final classifier
        self.c2 = convGG()
        self.c3 = convGG()
        self.c4 = convGG()
        self.c5 = convGG()
        self.c6 = convGG()
        self.c7 = convGG()
        self.c8 = convGG()
        self.c9 = convGG()
        self.c10 = convGG()
        self.c11 = convGG()

        self.c_final = self.layers.ConvGG(
            N_in=self.N_channels, N_out=10, kernel_size=4,
            h_grid=self.h_grid, input_h_grid=self.h_grid,
            stride=self.stride, padding=self.padding
        )

        # Dropout & Normalization
        self.dp = nn.Dropout(self.p)

        self.bn = nn.ModuleList([
            nn.BatchNorm3d(self.N_channels, eps=self.eps) for _ in range(11)
        ])

        self.max_pooling = self.layers.max_pooling_Rn

    def forward(self, x):
        out = self.dp(torch.relu(self.bn[0](self.c1(x))))
        out = torch.relu(self.bn[1](self.c2(out)))
        out = self.max_pooling(out, kernel_size=2, stride=2, padding=0)

        out = self.dp(torch.relu(self.bn[2](self.c3(out))))
        out = self.dp(torch.relu(self.bn[3](self.c4(out))))
        out = self.dp(torch.relu(self.bn[4](self.c5(out))))
        out = self.dp(torch.relu(self.bn[5](self.c6(out))))
        out = self.dp(torch.relu(self.bn[6](self.c7(out))))
        out = self.dp(torch.relu(self.bn[7](self.c8(out))))
        out = self.dp(torch.relu(self.bn[8](self.c9(out))))
        out = self.dp(torch.relu(self.bn[9](self.c10(out))))
        out = self.dp(torch.relu(self.bn[10](self.c11(out))))

        out = self.c_final(out)
        out, _ = torch.max(out, dim=-3)
        out = torch.mean(out, dim=[-2, -1])
        return out
