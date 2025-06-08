
import torch
import math
import numpy as np

from src.models.attgconv.attention_layers import ConvRnGLayer, ConvGGLayer


class fSpatialAttention(ConvRnGLayer):
    def __init__(self,
                 group,
                 N_in,
                 h_grid,
                 stride=1,
                 dilation=1,
                 groups=1,
                 wscale=1.0
                 ):

        kernel_size = 1
        padding = 0
        N_out = 1
        super(fSpatialAttention, self).__init__(group, N_in, N_out, kernel_size, h_grid, stride, padding, dilation, groups, wscale)

    def forward(self, input, visualize=False):
        return self.f_att_conv2d(input, visualize)

    def f_att_conv2d(self, input, visualize):

        output = self.conv_Rn_G(input)

        output, _ = output.max(dim=2)

        output = torch.sigmoid(output)

        return output


class fSpatialAttentionGG(ConvGGLayer):
    def __init__(self,
                 group,
                 N_in,
                 input_h_grid,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 wscale=1.0
                 ):
        kernel_size = 1
        N_out = 1
        padding = 0
        super(fSpatialAttentionGG, self).__init__(group, N_in, N_out, kernel_size, input_h_grid, input_h_grid, stride, padding, dilation, groups, wscale)

    def forward(self, input, visualize=False):
        return self.f_att_conv_GG(input, visualize)

    def f_att_conv_GG(self, input, visualize):

        output = self.conv_G_G(input)

        output = torch.sigmoid(output)

        return output
