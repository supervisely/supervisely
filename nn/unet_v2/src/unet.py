# coding: utf-8

import torch
from torch import nn
from torchvision.models.vgg import vgg16_bn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.seq(inputs)


class Block(nn.Module):
    def __init__(self, src_channels, dst_channels):
        super().__init__()
        self.seq1 = ConvBNAct(src_channels, dst_channels)
        self.seq2 = ConvBNAct(dst_channels, dst_channels)
        self.seq3 = ConvBNAct(dst_channels, dst_channels)

    def forward(self, x):
        result = self.seq1(x)
        result = self.seq2(result)
        result = self.seq3(result)
        return result


class UNetUp(nn.Module):
    def __init__(self, down_channels,  right_channels):
        super().__init__()
        self.bottom_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(down_channels, right_channels, kernel_size=1, stride=1)

    def forward(self, left, bottom):
        from_bottom = self.bottom_up(bottom)
        from_bottom = self.conv(from_bottom)
        result = torch.cat([left, from_bottom], 1)
        return result


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), dilation=2, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), dilation=2, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.conv2(self.relu(out))
        out = self.bn2(out)
        return torch.cat((x, self.relu2(out)), dim=1)


class UNet(nn.Module):

    def __init__(self, encoder_blocks,  encoder_channels, n_cls):
        self.encoder_channels = encoder_channels
        self.depth = len(self.encoder_channels)
        assert len(encoder_blocks) == self.depth
        super().__init__()
        
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        
        self.blocks = nn.ModuleList()
        # add bottleneck
        self.blocks.append(Block(
            self.encoder_channels[-1],
            self.encoder_channels[-1]
        ))
        
        self.ups = nn.ModuleList()
        for i in range(1, self.depth):
            bottom_channels = self.encoder_channels[self.depth - i]
            left_channels = self.encoder_channels[self.depth - i - 1]
            right_channels = left_channels
            self.ups.append(UNetUp(bottom_channels,  right_channels))
            self.blocks.append(Block(
                left_channels + right_channels,
                right_channels
            ))
        self.last_conv = nn.Conv2d(encoder_channels[0], n_cls, 1)
        # self.dropout = nn.Dropout2d(p=0.1)
        self.bottle = Bottleneck(512, 512)

    def forward(self, x):
        encoder_outputs = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_outputs.append(x)
        x = self.bottle(encoder_outputs[self.depth - 1])
        for i in range(self.depth):
            if i > 0:
                encoder_output = encoder_outputs[self.depth - i - 1]
                x = self.ups[i - 1](encoder_output, x)
                x = self.blocks[i](x)
        # x = self.dropout(x)
        x = self.last_conv(x)
        return x  # no softmax or log_softmax


def _get_encoder_blocks(model):
    # last modules (ReLUs) of VGG blocks
    layers_last_module_names = ['5', '12', '22', '32', '42']
    result = []
    cur_block = nn.Sequential()
    for name, child in model.named_children():
        if name == 'features':
            for name2, child2 in child.named_children():
                cur_block.add_module(name2, child2)
                if name2 in layers_last_module_names:
                    result.append(cur_block)
                    cur_block = nn.Sequential()
            break

    return result


def construct_unet(n_cls):  # no weights inited
    model = vgg16_bn(pretrained=False)
    encoder_blocks = _get_encoder_blocks(model)
    encoder_channels = [64, 128, 256, 512, 1024]  # vgg16 channels
    # prev_channels = encoder_channels[-1]

    return UNet(encoder_blocks, encoder_channels, n_cls)
