import torch
import torch.nn as nn
import functools


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        use_bias = False
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        use_bias = False
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
        use_bias = True
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer, use_bias


def get_norm_layer_1D(norm_type):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
        use_bias = False
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
        use_bias = False
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
        use_bias = True
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer, use_bias


def get_act_layer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU
    elif act_type == 'lrelu':
        act_layer = functools.partial(nn.LeakyReLU, negative_slope=0.01)
    elif act_type == 'none':
        def act_layer(): return Identity()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer


class CkBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, act_type, stride):
        super(CkBlock, self).__init__()
        norm_layer, use_bias = get_norm_layer(norm_type)
        act_layer = get_act_layer(act_type)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=use_bias),
            norm_layer(out_channels),
            act_layer()
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class LocalPathBlock(nn.Module):
    def __init__(self, in_channels, norm_type, act_type):
        super(LocalPathBlock, self).__init__()
        norm_layer, use_bias = get_norm_layer(norm_type)
        act_layer = get_act_layer(act_type)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(in_channels),
            act_layer(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(in_channels),
            act_layer(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(in_channels),
            act_layer(),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class GlobalPathBlock(nn.Module):
    def __init__(self, in_channels, norm_type, act_type):
        super(GlobalPathBlock, self).__init__()
        norm_layer, use_bias = get_norm_layer(norm_type)
        act_layer = get_act_layer(act_type)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=use_bias),
            norm_layer(in_channels),
            act_layer(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=4, dilation=4, bias=use_bias),
            norm_layer(in_channels),
            act_layer(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=8, dilation=8, bias=use_bias),
            norm_layer(in_channels),
            act_layer(),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class GlobalAveragePooling2D(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling2D, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=(-1, -2))
        return x


class StartConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, act_type):
        super(StartConvBlock, self).__init__()
        norm_layer, use_bias = get_norm_layer(norm_type)
        act_layer = get_act_layer(act_type)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_channels),
            act_layer()
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class LastConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, act_type):
        super(LastConvBlock, self).__init__()
        norm_layer, use_bias = get_norm_layer(norm_type)
        act_layer = get_act_layer(act_type)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(in_channels * 4),
            act_layer(),
            nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(in_channels * 2),
            act_layer(),
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class CpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, act_type):
        super(CpBlock, self).__init__()
        norm_layer, use_bias = get_norm_layer_1D(norm_type)
        act_layer = get_act_layer(act_type)
        self.layers = nn.Sequential(
            GlobalAveragePooling2D(),
            nn.Linear(in_channels, in_channels * 4),
            norm_layer(in_channels * 4),
            act_layer(),
            nn.Linear(in_channels * 4, in_channels * 2),
            norm_layer(in_channels * 2),
            act_layer(),
            nn.Linear(in_channels * 2, out_channels),
        )

    def forward(self, x):
        out = self.layers(x)
        return out
