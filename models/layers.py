import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, num_inputs, num_filters, bn=True, kernel_size=3, stride=1,
                 padding=None, transpose=False, dilation=1):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = (kernel_size-1)//2 if transpose is not None else 0
        if transpose:
            self.layer = nn.ConvTranspose2d(num_inputs, num_filters, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation)
        else:
            self.layer = nn.Conv2d(num_inputs, num_filters, kernel_size=kernel_size,
                                   stride=stride, padding=padding)
        nn.init.kaiming_uniform_(self.layer.weight, a=np.sqrt(5))
        self.bn_layer = nn.BatchNorm2d(num_filters) if bn else None

    def forward(self, x):
        out = self.layer(x)
        out = F.relu(out)
        return out if self.bn_layer is None else self.bn_layer(out)


def stacked_down_conv(n, num_inputs):
    return nn.Sequential(*[ConvLayer((2**i) * num_inputs, (2**(i+1)) * num_inputs,
                                     stride=2, padding=1, kernel_size=4) for i in range(n)])


def stacked_up_conv(n, num_inputs):
    return nn.Sequential(*[ConvLayer(num_inputs // (2**i), num_inputs // (2**(i+1)),
                                     transpose=True, stride=2, padding=1, kernel_size=4)
                           for i in range(n)])


def upsampling_combiners(n, num_inputs):
    return nn.Sequential(*[ConvLayer(2*num_inputs // (2**i), num_inputs // (2**i))
                           for i in range(n)])


def atrous_conv(n, num_inputs):
    return nn.ModuleList([ConvLayer(num_inputs, num_inputs, dilation=2**i)
                          for i in range(n)])


def atrous_upsampling_combiners(n, num_inputs):
    return nn.Sequential(*[ConvLayer(2*num_inputs // (2**i), num_inputs // (2**i))
                           for i in range(n)])


class AtrousConv(nn.Module):
    def __init__(self, n, num_inputs):
        super(AtrousConv, self).__init__()
        self.atrous_conv_layers = atrous_conv(n, num_inputs)
        self.combiner = ConvLayer(n * num_inputs, num_inputs // 2)
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=True)

    def forward(self, x):
        out = torch.cat([layer(x) for layer in self.atrous_conv_layers], dim=1)
        out = self.combiner(out)
        return self.upsampler(out)


def stacked_upsampler(n, num_inputs, num_scales):
    return nn.Sequential(*[AtrousConv(num_scales, num_inputs // (2**i))
                           for i in range(n)])
