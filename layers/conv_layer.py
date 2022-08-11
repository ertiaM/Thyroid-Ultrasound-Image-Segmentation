# encoding: utf-8
from torch import nn


def conv_layer(in_channels, out_channles, kernel_size, stride=1, padding=0, bias=True):
    layer = nn.Conv2d(in_channels, out_channles, kernel_size, stride, padding, bias=bias)
    layer.weight.data.zero_()
    if bias:
        layer.bias.data.zero_()
    return layer
