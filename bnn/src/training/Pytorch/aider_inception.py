import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules import *
parser = argparse.ArgumentParser(description='PyTorch Quantized Inception (MNIST) Example')
parser.add_argument('--wb', type=int, default=1, metavar='N', choices=[1, 2, 4], help='number of bits for weights (default: 1)')
parser.add_argument('--ab', type=int, default=1, metavar='N', choices=[1, 2, 4], help='number of bits for activations (default: 1)')
wb =2
ab =2
def init_weights(m):
    if type(m) == BinarizeLinear or type(m) == BinarizeConv2d:
        torch.nn.init.uniform_(m.weight, -1, 1)
        m.bias.data.fill_(0.01)


class BinarizeConv2d_block(nn.Module):
    def __init__(self, wb, ifm_ch, num_filt, kernel_size, stride, padding, bias):
        super(BinarizeConv2d_block, self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(wb, ifm_ch, num_filt, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_filt),
            nn.Hardtanh(inplace=True),
            Quantizer(ab))

    def forward(self, x):
        return self.features(x)


class inception_module(nn.Module):
    def __init__(self,in_channels):
        super(inception_module, self).__init__()
        self.conv3x3 = BinarizeConv2d_block(wb, in_channels, in_channels, kernel_size=3, stride=1, padding=0,
                                            bias=True)
        self.conv5x5 = BinarizeConv2d_block(wb, in_channels, in_channels, kernel_size=5, stride=1, padding=1,
                                            bias=True)
        self.conv7x7 = BinarizeConv2d_block(wb, in_channels, in_channels, kernel_size=7, stride=1, padding=2,
                                            bias=True)

    def forward(self, x):
        x_3x3 = self.conv3x3(x)
        x_5x5 = self.conv5x5(x)
        x_7x7 = self.conv7x7(x)
        x = torch.cat((x_3x3, x_5x5, x_7x7), 1)
        return x


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(wb, 3, 8, kernel_size=3, stride=1, padding=1, bias=True),  # first conv layer
            nn.BatchNorm2d(8),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),
            nn.MaxPool2d(kernel_size=2, stride=2),

            inception_module(8),  # first inception module
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.2),

            BinarizeConv2d(wb, 3 * 8, 64, kernel_size=1, stride=1, padding=1, bias=True),  # fusion conv layer
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinarizeConv2d(wb, 64, 32, kernel_size=1, stride=1, padding=1, bias=True),  # reduced conv layer
            nn.BatchNorm2d(32),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),
            nn.MaxPool2d(kernel_size=2, stride=1),

            inception_module(32),  # second inception module
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.2),

            BinarizeConv2d(wb, 3 * 32, 128, kernel_size=1, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),
            nn.MaxPool2d(kernel_size=2, stride=1),

            BinarizeConv2d(wb, 128, 64, kernel_size=1, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),
            nn.MaxPool2d(kernel_size=1, stride=1),

            inception_module(64),  # third inception module
            nn.MaxPool2d(kernel_size=1, stride=1),
            nn.Dropout(0.2),

            BinarizeConv2d(wb, 3 * 64, 256, kernel_size=1, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinarizeConv2d(wb, 256, 128, kernel_size=1, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),
            nn.MaxPool2d(kernel_size=1, stride=1))

        self.classifier = nn.Sequential(
            BinarizeLinear(wb, 128*32* 32, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),

            nn.Dropout(0.5),

            BinarizeLinear(wb, 1024, 5, bias=True),
            nn.BatchNorm1d(5),
            nn.LogSoftmax())

        self.features.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(self, x):
        x = self.features(x)
#        print(x.shape)
        x = x.view(-1, 128*32*32)
        x = self.classifier(x)
        return x
