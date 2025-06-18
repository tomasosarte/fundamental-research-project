import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- layers ----------------------------

class DBAGEConv(nn.Module):
    """
    Directional Bias-Aware Grouped Convolution + attention on 90° rotations.
    Produces a 5-D tensor: (B, C_out, N_ori, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        n_orientations: int = 4,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.n_ori = n_orientations

        # weight: (C_out, C_in, k, k)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels,
                                               kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode="fan_out",
                                nonlinearity="relu")

        # one bias per *destination* orientation
        self.bias = nn.Parameter(torch.zeros(n_orientations))

        self.stride = stride
        self.padding = padding

    # ---------------------------------------------------------

    @staticmethod
    def _rotate_x90(t: torch.Tensor, k: int) -> torch.Tensor:
        """Rotate tensor by k×90° CCW on the last two dims."""
        k = k % 4
        return t if k == 0 else torch.rot90(t, k=k, dims=(-2, -1))

    # ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : shape (B, C_in, H, W)   **or**
            shape (B, C_in, N_ori, H, W)  ← comes from previous DBAGEConv
        """
        # If the previous layer already has orientation channels, merge them.
        if x.dim() == 5:                               # (B, C_in, N, H, W)
            x = x.mean(dim=2)                          # orientation-avg

        # ---- grouped 2-D convolutions, one per orientation ---------------
        convs = []
        for g in range(self.n_ori):
            w = self._rotate_x90(self.weight, g)       # rotate the kernel
            conv_g = F.conv2d(x, w, stride=self.stride,
                              padding=self.padding)    # (B, C_out, H', W')
            convs.append(conv_g)

        convs = torch.stack(convs, dim=2)              # (B, C_out, N, H', W')

        # -------------- self-attention across orientations ----------------
        # global pooling → (B, C_out, N) → (B, N, C_out)
        pooled = convs.mean(dim=[3, 4]).permute(0, 2, 1)

        # scaled dot-product
        scale = 1.0 / math.sqrt(self.out_channels)
        s = torch.einsum("bgc,bkc->bgk", pooled, pooled) * scale
        s = s + self.bias.view(1, 1, -1)               # broadcast

        alpha = F.softmax(s, dim=2)                    # (B, N, N)

        # weighted, rotation-aligned sum
        outs = []
        for g in range(self.n_ori):
            acc = 0
            for gp in range(self.n_ori):
                w_att = alpha[:, g, gp].view(-1, 1, 1, 1)  # (B,1,1,1)
                feat = self._rotate_x90(convs[:, :, gp], (g - gp) % self.n_ori)
                acc = acc + w_att * feat
            outs.append(acc)

        return torch.stack(outs, dim=2)                # (B, C_out, N, H', W')

# ----------------------- whole network ----------------------

class DBAGEConvNet(nn.Module):
    """CIFAR-style backbone with three DBAGE blocks."""

    def __init__(self, n_classes: int = 10, n_ori: int = 4):
        super().__init__()
        self.n_ori = n_ori

        self.c1 = DBAGEConv(3,   32, 3, padding=1, n_orientations=n_ori)
        self.bn1 = nn.BatchNorm3d(32)

        self.c2 = DBAGEConv(32,  64, 3, padding=1, n_orientations=n_ori)
        self.bn2 = nn.BatchNorm3d(64)

        self.c3 = DBAGEConv(64, 128, 3, padding=1, n_orientations=n_ori)
        self.bn3 = nn.BatchNorm3d(128)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, n_classes)

    # ---------------------------------------------------------

    @staticmethod
    def _max_pool_2x2(x: torch.Tensor) -> torch.Tensor:
        """
        3-D max-pool (H, W) **while** keeping orientation dimension.
        Input  : (B, C, N, H, W)
        Return : (B, C, N, H/2, W/2)
        """
        b, c, n, h, w = x.size()
        x = x.view(b, c * n, h, w)                 # fold N into channels
        x = F.max_pool2d(x, 2, 2)                  # (B, C*N, h/2, w/2)
        h2, w2 = x.size(-2), x.size(-1)
        return x.view(b, c, n, h2, w2)

    # ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.c1(x)), inplace=True)
        x = self._max_pool_2x2(x)

        x = F.relu(self.bn2(self.c2(x)), inplace=True)
        x = self._max_pool_2x2(x)

        x = F.relu(self.bn3(self.c3(x)), inplace=True)

        # shape  (B, C, N, 1, 1)  → (B, C, N)
        x = self.global_pool(x).squeeze(-1).squeeze(-1)

        # pool across orientations
        x = x.mean(dim=2)                          # (B, C)

        return self.fc(x)