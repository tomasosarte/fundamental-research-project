import torch
import math
import numpy as np
import importlib

import src.models.attgconv as attgconv
from src.models.attgconv import ConvRnGLayer
from src.models.attgconv import ConvGGLayer


class ChannelAttention(torch.nn.Module):
    def __init__(self, N_out, N_in, ratio=1):
        super(ChannelAttention, self).__init__()
        self.linear = torch.nn.functional.linear
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.N_in = N_in
        self.N_out = N_out
        self.weight_fc1 = torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in // ratio, self.N_in))
        self.weight_fc2 = torch.nn.Parameter(torch.Tensor(self.N_out, self.N_in, self.N_in // ratio))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight_fc1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight_fc2, a=math.sqrt(5))

    def forward(self, input):
        input_mean = input.mean(dim=[-2, -1]).unsqueeze(-1)
        input_max = input.max(dim=-2)[0].max(dim=-1)[0].unsqueeze(-1)
        avg_out = self._linear(torch.relu(self._linear(input_mean, self.weight_fc1)), self.weight_fc2)
        max_out = self._linear(torch.relu(self._linear(input_max, self.weight_fc1)), self.weight_fc2)
        out = torch.sigmoid(avg_out + max_out)
        out = torch.reshape(out, [input.shape[0], self.N_out, input.shape[2], self.N_in, 1, 1])
        return out

    def _linear(self, input, w):
        in_reshaped = input.unsqueeze(-3)
        w_reshaped = w.reshape(1, w.shape[0], 1, w.shape[1], w.shape[2], 1)
        output = (in_reshaped * w_reshaped).sum(-2)
        return output

class ChannelAttentionGG(ChannelAttention):
    def __init__(self, N_h, N_out, N_h_in, N_in, ratio=1, bias=False):
        super(ChannelAttentionGG, self).__init__(N_out, N_in, ratio=ratio)
        self.N_h_in = N_h_in
        self.N_h = N_h
        self.weight_fc1 = torch.nn.Parameter(torch.rand(self.N_out, self.N_in // ratio, self.N_in, self.N_h_in))
        self.weight_fc2 = torch.nn.Parameter(torch.rand(self.N_out, self.N_in, self.N_in // ratio, self.N_h_in))
        self.reset_parameters()

    def forward(self, input):
        fc1, fc2 = self._left_action_of_h_grid()
        input_mean = input.mean(dim=[-2, -1]).unsqueeze(-1)
        input_max = input.max(dim=-2)[0].max(dim=-1)[0].unsqueeze(-1)
        avg_out = self._linear(torch.relu(self._linear(input_mean, fc1)), fc2)
        max_out = self._linear(torch.relu(self._linear(input_max, fc1)), fc2)
        out = torch.sigmoid(avg_out + max_out)
        out = torch.reshape(out, [input.shape[0], self.N_out, self.N_h, -1, self.N_h_in, 1, 1])
        return out

    def _linear(self, input, w):
        in_reshaped = input.unsqueeze(-4)
        w_reshaped = torch.reshape(w, [1, w.shape[0], w.shape[1], w.shape[2], w.shape[3], w.shape[4], 1])
        output = (in_reshaped * w_reshaped).sum(-3)
        return output

    def _left_action_of_h_grid(self):
        fc1 = torch.stack([self.weight_fc1.roll(shifts=i, dims=-1) for i in range(self.N_h)], dim=1)
        fc2 = torch.stack([self.weight_fc2.roll(shifts=i, dims=-1) for i in range(self.N_h)], dim=1)
        return fc1, fc2

class SpatialAttention(ConvRnGLayer):
    def __init__(
            self,
            group,
            N_in,
            N_out,
            kernel_size,
            h_grid,
            stride,
            dilation=1,
            wscale=1.0
        ):
        N_in = 2
        padding = dilation * (kernel_size //2)
        super(SpatialAttention, self).__init__(group, N_in, N_out, kernel_size, h_grid, stride, padding=padding, dilation=dilation, conv_groups=len(h_grid.grid), wscale=wscale)

    def forward(self, input):
        return self.conv_Rn_G(input)

    def conv_Rn_G(self, input):
        avg_in = torch.mean(input, dim=-3, keepdim=True)
        max_in, _ = torch.max(input, dim=-3, keepdim=True)
        input = torch.cat([avg_in, max_in], dim=-3)
        kernel_stack = torch.stack([self.kernel(self.h_grid.grid[i]) for i in range(self.N_h)], dim=1)
        kernel_stack_as_if_Rn = torch.reshape(kernel_stack, [self.N_out * self.N_h * self.N_in, 1, kernel_stack.shape[-2], kernel_stack.shape[-1]])
        input_as_if_Rn = torch.reshape(input, [input.shape[0], self.N_out * self.N_h * self.N_in, input.shape[-2], input.shape[-1]])
        output = torch.conv2d(
            input=input_as_if_Rn,
            weight=kernel_stack_as_if_Rn,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.N_out*self.N_h*self.N_in)
        output = torch.reshape(output, [output.shape[0], self.N_out, self.N_h, self.N_in, output.shape[-2], output.shape[-1]])
        output = output.sum(-3, keepdim=True)
        output = torch.sigmoid(output)
        return output

class SpatialAttentionGG(ConvGGLayer):
    def __init__(
            self,
            group,
            N_in,
            N_out,
            kernel_size,
            h_grid,
            input_h_grid,
            stride,
            dilation=1,
            wscale=1.0
        ):
        N_in = 2
        padding = dilation * (kernel_size//2)
        super(SpatialAttentionGG, self).__init__(group, N_in, N_out, kernel_size, h_grid, input_h_grid, stride, padding=padding, dilation=dilation, conv_groups=len(h_grid.grid), wscale=wscale)


    def forward(self, input):
        return self.conv_G_G(input)

    def conv_G_G(self, input):
        avg_in = torch.mean(input, dim=-4, keepdim=True)
        max_in, _ = torch.max(input, dim=-4, keepdim=True)
        input = torch.cat([avg_in, max_in], dim=-4)
        kernel_stack = torch.stack([self.kernel(self.h_grid.grid[i]) for i in range(self.N_h)], dim=1)
        kernel_stack_as_if_Rn = torch.reshape(kernel_stack, [self.N_h * self.N_out * self.N_in * self.N_h_in, 1, self.kernel_size, self.kernel_size])
        input_tensor_as_if_Rn = torch.reshape(input, [input.shape[0], self.N_h * self.N_out * self.N_in * self.N_h_in, input.shape[-2], input.shape[-1]])

        output = torch.conv2d(
            input=input_tensor_as_if_Rn,
            weight=kernel_stack_as_if_Rn,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.N_h * self.N_h_in * self.N_out * self.N_in)

        output = torch.reshape(output, [input.shape[0], self.N_out, self.N_h, self.N_in, self.N_h_in, input.shape[-2], input.shape[-1]])
        output = output.sum(-4, keepdim=True)

        output = torch.sigmoid(output)

        return output

class fChannelAttention(torch.nn.Module):
    def __init__(self, N_in, ratio=1):
        super(fChannelAttention, self).__init__()
        self.N_in = N_in
        self.ratio = ratio
        self.weight_fc1 = torch.nn.Parameter(torch.Tensor(self.N_in // ratio, self.N_in))
        self.weight_fc2 = torch.nn.Parameter(torch.Tensor(self.N_in, self.N_in // ratio))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight_fc1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight_fc2, a=math.sqrt(5))

    def forward(self, input):
        input_mean = input.mean(dim=[-2, -1]).unsqueeze(-1)
        input_max = input.max(dim=-2)[0].max(dim=-1)[0].unsqueeze(-1)
        avg_out = self._linear(torch.relu(self._linear(input_mean, self.weight_fc1)), self.weight_fc2)
        max_out = self._linear(torch.relu(self._linear(input_max, self.weight_fc1)), self.weight_fc2)
        out = torch.sigmoid(avg_out + max_out)
        out = torch.reshape(out, [input.shape[0], self.N_in, 1, 1])
        return out

    def _linear(self, input, w):
        in_reshaped = input.unsqueeze(-3)
        w_reshaped = w.reshape(1, w.shape[0], w.shape[1], 1)
        output = (in_reshaped * w_reshaped).sum(-2)
        return output


class fChannelAttentionGG(torch.nn.Module):
    def __init__(self, N_h_in, N_in, ratio=1, group='SE2'):
        super(fChannelAttentionGG, self).__init__()
        self.N_in = N_in
        self.ratio = ratio
        self.N_h_in = N_h_in
        self.N_h = N_h_in
        self.weight_fc1 = torch.nn.Parameter(torch.rand(self.N_in // ratio, self.N_in, self.N_h_in))
        self.weight_fc2 = torch.nn.Parameter(torch.rand(self.N_in, self.N_in // ratio, self.N_h_in))

        self.action = self._left_action_of_h_grid_se2
        if group == 'E2':
            
            group = importlib.import_module('src.models.attgconv.group.' + group)
            e2_layers = attgconv.layers(group)
            n_grid = 8
            self.h_grid = e2_layers.H.grid_global(n_grid)
            self.action = self._left_action_on_grid_e2
        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight_fc1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight_fc2, a=math.sqrt(5))

    def forward(self, input):
        fc1, fc2 = self.action()
        input_mean = input.mean(dim=[-2, -1]).unsqueeze(-1)
        input_max = input.max(dim=-2)[0].max(dim=-1)[0].unsqueeze(-1)
        avg_out = self._linear(torch.relu(self._linear(input_mean, fc1)), fc2)
        max_out = self._linear(torch.relu(self._linear(input_max, fc1)), fc2)
        out = torch.sigmoid(avg_out + max_out)
        out = torch.reshape(out, [out.shape[0], self.N_in, self.N_h_in, 1, 1])
        return out

    def _linear(self, input, w):
        in_reshaped = input.unsqueeze(-4).unsqueeze(-5)
        w_reshaped = torch.reshape(w, [1, w.shape[0], w.shape[1], w.shape[2], w.shape[3], 1])
        output = (in_reshaped * w_reshaped).sum(dim=[-3,-2])
        return output

    def _left_action_of_h_grid_se2(self):
        fc1 = torch.stack([self.weight_fc1.roll(shifts=i, dims=-1) for i in range(self.N_h)], dim=1)
        fc2 = torch.stack([self.weight_fc2.roll(shifts=i, dims=-1) for i in range(self.N_h)], dim=1)
        return fc1, fc2

    def _left_action_on_grid_e2(self):
        fc1 = torch.stack([self._left_action_of_h_grid_e2(h, self.weight_fc1) for h in self.h_grid.grid], dim=1)
        fc2 = torch.stack([self._left_action_of_h_grid_e2(h, self.weight_fc2) for h in self.h_grid.grid], dim=1)
        return fc1, fc2

    def _left_action_of_h_grid_e2(self, h, fx):
        shape = fx.shape
        Lgfx = fx.clone()
        Lgfx = torch.reshape(Lgfx, [shape[0], shape[1], 2, 4])
        if h[0] != 0:
            Lgfx[:, :, 0, :] = torch.roll(Lgfx[:, :, 0, :], shifts=int(torch.round((1. / (np.pi / 2.) * h[0])).item()), dims=-1)
            Lgfx[:, :, 1, :] = torch.roll(Lgfx[:, :, 1, :], shifts=-int(torch.round((1. / (np.pi / 2.) * h[0])).item()), dims=-1)
        if h[-1] == -1:
            Lgfx = torch.roll(Lgfx, shifts=1, dims=-2)
        Lgfx = torch.reshape(Lgfx, shape)
        return Lgfx


class fSpatialAttention(ConvRnGLayer):
    def __init__(
            self,
            group,
            kernel_size,
            h_grid,
            stride=1,
            dilation=1,
            groups=1,
            wscale=1.0
        ):
        N_in = 2
        N_out = 1
        padding = dilation * (kernel_size // 2)
        super(fSpatialAttention, self).__init__(group, N_in, N_out, kernel_size, h_grid, stride, padding, dilation, groups, wscale)

    def forward(self, input, visualize=False):
        return self.f_att_conv2d(input)

    def f_att_conv2d(self, input):
        avg_in = torch.mean(input, dim=-3, keepdim=True)
        max_in, _ = torch.max(input, dim=-3, keepdim=True)
        input = torch.cat([avg_in, max_in], dim=-3)
        output = self.conv_Rn_G(input)
        output, _ = output.max(dim=2)
        output = torch.sigmoid(output)
        return output

class fSpatialAttentionGG(ConvGGLayer):
    def __init__(
            self,
            group,
            kernel_size,
            input_h_grid,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            wscale=1.0
        ):
        N_in = 2
        N_out = 1
        padding = dilation * (kernel_size // 2)
        super(fSpatialAttentionGG, self).__init__(group, N_in, N_out, kernel_size, input_h_grid, input_h_grid, stride, padding, dilation, groups, wscale)

    def forward(self, input, visualize=False):
        return self.f_att_conv_GG(input)

    def f_att_conv_GG(self, input):
        avg_in = torch.mean(input, dim=-4, keepdim=True)
        max_in, _ = torch.max(input, dim=-4, keepdim=True)
        input = torch.cat([avg_in, max_in], dim=-4)
        output = self.conv_G_G(input)
        output = torch.sigmoid(output)
        return output
