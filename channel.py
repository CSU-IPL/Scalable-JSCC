import torch.nn as nn
import numpy as np
import os
import torch
import time


class Channel(nn.Module):
    """
    Currently the channel model is either error free, erasure channel,
    rayleigh channel or the AWGN channel.
    """

    def __init__(self, args):
        super(Channel, self).__init__()
        self.args = args
        self.chan_type = args.channel_type
        self.device = args.device
        self.h = torch.sqrt(torch.randn(1) ** 2
                            + torch.randn(1) ** 2) / 1.414

    def gaussian_noise_layer(self, input_layer, std, name=None):
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag

        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std, name=None):
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise_real + 1j * noise_imag
        noise = noise.to(input_layer.get_device())

        self.h = self.h.to(input_layer.get_device())

        return input_layer * self.h + noise

    def complex_normalize(self, feature, target_power=1.0):
        in_shape = feature.shape
        sig_in = feature.view(in_shape[0], -1)

        power = torch.mean(sig_in ** 2, dim=1, keepdim=True)

        sig_out = sig_in / torch.sqrt(power + 1e-8) * torch.sqrt(torch.tensor(target_power, device=feature.device))

        return sig_out.view(in_shape)

    def forward(self, input, chan_param, avg_pwr=False):

        if avg_pwr:
            power = 1
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx = self.complex_normalize(input)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j
        channel_output = self.complex_forward(channel_in, chan_param)
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)
        if self.chan_type == 1 or self.chan_type == 'awgn':
            noise = (channel_output - channel_tx).detach()
            noise.requires_grad = False
            channel_tx = channel_tx + noise
            if avg_pwr:
                return channel_tx
            else:
                return channel_tx
        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            if avg_pwr:
                return channel_output
            else:
                return channel_output

    def complex_forward(self, channel_in, chan_param):
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            sigma = 10 ** (-chan_param * 1.0 / 10 / 2)
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="awgn_chan_noise")
            return chan_output

        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            channel_tx = channel_in
            sigma = 10 ** (-chan_param * 1.0 / 10 / 2)
            chan_output = self.rayleigh_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="rayleigh_chan_noise")
            return chan_output


    def noiseless_forward(self, channel_in):
        channel_tx = self.normalize(channel_in, power=1)
        return channel_tx

