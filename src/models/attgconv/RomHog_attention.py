import torch
from src.models.attgconv.attention_layers import ConvRnGLayer, ConvGGLayer

class fSpatialAttention(ConvRnGLayer):
    def __init__(
        self,
        group,
        N_in,
        h_grid,
        stride=1,
        dilation=1,
        groups=1,
        wscale=1.0,
    ):
        super().__init__(
            group=group,
            N_in=N_in,
            N_out=1,
            kernel_size=1,
            h_grid=h_grid,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            wscale=wscale,
        )

    def forward(self, x, visualize=False):
        out = self.conv_Rn_G(x)
        out, _ = out.max(dim=2)  # Max over group elements
        return torch.sigmoid(out)


class fSpatialAttentionGG(ConvGGLayer):
    def __init__(
        self,
        group,
        N_in,
        input_h_grid,
        stride=1,
        dilation=1,
        groups=1,
        wscale=1.0,
    ):
        super().__init__(
            group=group,
            N_in=N_in,
            N_out=1,
            kernel_size=1,
            input_h_grid=input_h_grid,
            h_grid=input_h_grid,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            wscale=wscale,
        )

    def forward(self, x, visualize=False):
        out = self.conv_G_G(x)
        return torch.sigmoid(out)