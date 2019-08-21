from .layers import (
    AtrousConv,
    ConvLayer,
    stacked_down_conv,
    stacked_up_conv,
    upsampling_combiners,
    atrous_conv,
    atrous_upsampling_combiners,
    stacked_upsampler
)

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n, num_classes, dilated=True):
        super(UNet, self).__init__()
        self.encoder = stacked_down_conv(n, 1)
        if dilated:
            self.decoder = stacked_upsampler(n, 2**n, 3)
            self.combiners = upsampling_combiners(n, 2**n)
        else:
            self.decoder = stacked_up_conv(n, 2**n)
            self.combiners = upsampling_combiners(n, 2**n)
        self.output = ConvLayer(1, num_classes, bn=False, kernel_size=1)
        self.epoch = 0

    def encode(self, x):
        input_activations = x
        encoder_outputs = []
        for layer in self.encoder:
            output_activations = layer(input_activations)
            encoder_outputs.append(output_activations)
            input_activations = output_activations
        return output_activations, encoder_outputs

    def decode(self, x, encoder_outputs):
        input_activations = x
        for encoder_output, decoder_layer, combiner in zip(encoder_outputs[::-1], self.decoder, self.combiners):
            stacked_input = torch.cat(
                (encoder_output, input_activations), dim=1)
            input_activations = combiner(stacked_input)
            output_activations = decoder_layer(input_activations)
            input_activations = output_activations
        return output_activations

    def forward(self, x):
        activation, encoder_outputs = self.encode(x)
        decoder_output = self.decode(activation, encoder_outputs)
        pixelwise_class_output = self.output(decoder_output)
        return pixelwise_class_output
