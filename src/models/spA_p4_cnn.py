# torch
import torch
import torch.nn as nn
# built-in
import functools



class A_P4CNN(nn.Module):
    def __init__(self, use_bias=False, attention=False):
        super(A_P4CNN, self).__init__()

        #Parameters of the group

        # Import the group structure
        import importlib
        group_name = 'SE2'
        group = importlib.import_module('src.models.attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import src.models.attgconv as attgconv
        se2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        n_grid = 4
        self.h_grid = se2_layers.H.grid_global(n_grid)

        # Parameters of the model
        p = 0.3
        stride = 1
        padding = 0
        kernel_size = 3
        N_channels = 10
        eps = 2e-5

        # Parameters of attention
        ch_ratio = 2        #(N_channels // 2)    # Hidden layer consists of 2 neurons
        sp_kernel_size = 7

        # Store in self such that all models share it
        self.ch_ratio = ch_ratio
        self.sp_kernel_size = sp_kernel_size

        from src.models.attgconv.attention_layers import ChannelAttention as ch_RnG
        from src.models.attgconv.attention_layers import ChannelAttentionGG #as ch_GG
        from src.models.attgconv.attention_layers import SpatialAttention #as sp_RnG
        from src.models.attgconv.attention_layers import SpatialAttentionGG

        ch_GG = functools.partial(ChannelAttentionGG, N_h=n_grid, N_h_in=n_grid)
        sp_RnG = functools.partial(SpatialAttention, group=group, h_grid=self.h_grid, stride=stride)
        sp_GG = functools.partial(SpatialAttentionGG, group=group, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride)

        self.c1 = se2_layers.AttConvRnG(N_in=1        , N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding,
                                        channel_attention=ch_RnG(N_out=N_channels, N_in=1        , ratio=1),
                                        spatial_attention=sp_RnG(N_out=N_channels, N_in=1        , kernel_size=sp_kernel_size)
                                        )
        self.c2 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out= N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out= N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c3 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c4 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c5 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c6 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c7 = se2_layers.AttConvGG(N_in=N_channels, N_out=10        , kernel_size=4         , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       channel_attention=ch_GG(N_out=10        , N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=10        , N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        # Dropout
        self.dp = nn.Dropout(p)
        # Batch Normalization
        self.bn1 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn2 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn3 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn4 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn5 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn6 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        # Max Pooling
        self.max_pooling = se2_layers.max_pooling_Rn

    def forward(self, x):
        #x = torch.rot90(x, k=3, dims=[-2, -1])  # Sanity check. Equivariance.
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



class A_Sp_P4CNN(A_P4CNN):
    # Inherits forward
    def __init__(self, use_bias=False, attention=False):
        super(A_Sp_P4CNN, self).__init__()

        #Parameters of the group

        # Import the group structure
        import importlib
        group_name = 'SE2'
        group = importlib.import_module('src.models.attgconv.group.' + group_name)
        # Import the gsplintes package and the layers
        import src.models.attgconv as attgconv
        se2_layers = attgconv.layers(group)  # The layers is instantiated with the group structure as input
        # Create H grid for p4 group
        n_grid = 4
        self.h_grid = se2_layers.H.grid_global(n_grid)

        # Parameters of the model
        p = 0.3
        stride = 1
        padding = 0
        kernel_size = 3
        N_channels = 10
        eps = 2e-5

        # Parameters of attention
        ch_ratio = self.ch_ratio        #(N_channels // 2)    # Hidden layer consists of 2 neurons
        sp_kernel_size = self.sp_kernel_size

        from src.models.attgconv.attention_layers import ChannelAttention as ch_RnG
        from src.models.attgconv.attention_layers import ChannelAttentionGG #as ch_GG
        from src.models.attgconv.attention_layers import SpatialAttention #as sp_RnG
        from src.models.attgconv.attention_layers import SpatialAttentionGG

        ch_GG = functools.partial(ChannelAttentionGG, N_h=n_grid, N_h_in=n_grid)
        sp_RnG = functools.partial(SpatialAttention, group=group, h_grid=self.h_grid, stride=stride)
        sp_GG = functools.partial(SpatialAttentionGG, group=group, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride)

        self.c1 = se2_layers.AttConvRnG(N_in=1        , N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride, padding=padding,
                                        #channel_attention=ch_RnG(N_out=N_channels, N_in=1        , ratio=1),
                                        spatial_attention=sp_RnG(N_out=N_channels, N_in=1        , kernel_size=sp_kernel_size)
                                        )
        self.c2 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       #channel_attention=ch_GG(N_out= N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out= N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c3 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       #channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c4 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       #channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c5 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       #channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c6 = se2_layers.AttConvGG(N_in=N_channels, N_out=N_channels, kernel_size=kernel_size, h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       #channel_attention=ch_GG(N_out=N_channels, N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=N_channels, N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        self.c7 = se2_layers.AttConvGG(N_in=N_channels, N_out=10        , kernel_size=4         , h_grid=self.h_grid, input_h_grid=self.h_grid, stride=stride, padding=padding,
                                       #channel_attention=ch_GG(N_out=10        , N_in=N_channels, ratio=ch_ratio),
                                       spatial_attention=sp_GG(N_out=10        , N_in=N_channels, kernel_size=sp_kernel_size)
                                       )
        # Dropout
        self.dp = nn.Dropout(p)
        # Batch Normalization
        self.bn1 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn2 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn3 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn4 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn5 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        self.bn6 = nn.BatchNorm3d(num_features=N_channels, eps=eps)
        # Max Pooling
        self.max_pooling = se2_layers.max_pooling_Rn

