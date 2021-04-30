# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import numpy as np

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

class Generator(nn.Module):
    # initializers
    def __init__(self, num_classes, shape_img, batchsize,channel=3, g_in=128, d = 32):
        super(Generator, self).__init__()
        self.g_in = g_in
        self.batchsize = batchsize

        self.fc2 = nn.Sequential(
            nn.Linear(g_in, num_classes)
        )
        block_num = int(np.log2(shape_img) - 3)
        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(g_in, d*pow(2,block_num) * 2, 4, 1, 0),
            GLU()
        )
        self.blocks = nn.ModuleList()
        for bn in reversed(range(block_num)):
            self.blocks.append(self.upBlock(pow(2, bn + 1) * d, pow(2, bn) * d))
        self.deconv_out = self.upBlock(d, channel)

    @staticmethod
    def upBlock(in_planes, out_planes):
        def conv3x3(in_planes, out_planes):
            "3x3 convolution with padding"
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                             padding=1, bias=False)

        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes*2),
            nn.BatchNorm2d(out_planes*2),
            GLU()
        )
        return block

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        y = torch.softmax(self.fc2(x), 1)
        x = x.view(self.batchsize, self.g_in, 1, 1)
        output = self.block0(x)
        # output = output.view(output.size(0), -1, 4, 4)
        for block in self.blocks:
            output = block(output)
        output = self.deconv_out(output)
        output = torch.sigmoid(output)
        return output, y

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
