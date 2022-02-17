#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class unettiny(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=1, out_channels=1, num_classes=1, dim=256):
        """Initializes U-Net."""

        super(unettiny, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv1d(in_channels, 4, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(4, 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv1d(4, 4, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv1d(4, 4, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(4, 4, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv1d(4*2, 4*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(4*2, 4*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(4*2, 4*2, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv1d(4*3, 4*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(4*2, 4*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(4*2, 4*2, 3, stride=2, padding=1, output_padding=1))
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv1d(4*2 + in_channels, 4*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(4*2, 4, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(4, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        self.dim = dim

        self.num_classes = num_classes
        self.fc_layers = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes), nn.Softmax(dim=1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        # N, C, D = x.shape
        # x = F.interpolate(x, size=256, mode="nearest")

        """Through encoder, then decoder by adding U-skip connections. """
        # print(x.shape)
        # Encoder
        pool1 = self._block1(x)
        # print(pool1.shape)
        pool2 = self._block2(pool1)
        # print(pool2.shape)
        pool3 = self._block2(pool2)
        # print(pool3.shape)
        pool4 = self._block2(pool3)
        # print(pool4.shape)
        pool5 = self._block2(pool4)
        # print(pool5.shape)

        # Decoder
        upsample5 = self._block3(pool5)
        # print(upsample5.shape)
        # print(upsample5.shape, pool4.shape)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        # print(upsample4.shape, pool3.shape)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        # print(concat4.shape)
        upsample3 = self._block5(concat4)
        # print(upsample3.shape)
        # print(pool2.shape)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        # print(concat1.shape)
        # Final activation
        out = self._block6(concat1)

        if self.num_classes > 1:
            out = out.view(-1, self.dim * 1)
            out = self.fc_layers(out)
        # else:
        #     out = F.interpolate(out, size=D, mode="nearest")

        return out


if __name__ == '__main__':

    with torch.no_grad():
        
        num_classes = 1
        net = unettiny(num_classes=num_classes, dim=256).cuda()

        x = torch.randn(64, 1, 256).cuda()
        y = net(x)
        print(y.shape)
