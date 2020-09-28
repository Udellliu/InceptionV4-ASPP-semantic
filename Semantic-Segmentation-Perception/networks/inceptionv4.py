import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        #self.conv = BasicConv2d(64, 96, kernel_size=(3,3), stride=2)
        self.conv=nn.Sequential(
            nn.Conv2d(64,96,3,stride=2),
            nn.BatchNorm2d(96,eps=0.001,momentum=0.1,affine=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        #print("stem_1:")
        #print(out.size())
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            #BasicConv2d(160, 64, kernel_size=(1,1), stride=1),
            nn.Conv2d(160, 64,1,stride=1),
            nn.BatchNorm2d(64,
                            eps=0.001,  # value found in tensorflow
                            momentum=0.1,  # default pytorch value
                            affine=True),
            nn.ReLU(inplace=True),

            #BasicConv2d(64, 96, kernel_size=(3,3), stride=1)
            nn.Conv2d(64, 96, 3, stride=1),
            nn.BatchNorm2d(96,
                            eps=0.001,  # value found in tensorflow
                            momentum=0.1,  # default pytorch value
                            affine=True),
            nn.ReLU(inplace=True)
        )

        self.branch1 = nn.Sequential(
            #BasicConv2d(160, 64, kernel_size=(1,1), stride=1),
            nn.Conv2d(160, 64, 1, stride=1),
            nn.BatchNorm2d(64,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            nn.Conv2d(64, 64, (7,1), stride=1,padding=(0,3)),
            nn.BatchNorm2d(64,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            nn.Conv2d(64, 64, (1, 7), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(64,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(64, 96, kernel_size=(3,3), stride=1)
            nn.Conv2d(64, 96, 3, stride=1),
            nn.BatchNorm2d(96,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        #print("stem_2:")
        #print(out.size())
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        #self.conv = BasicConv2d(192, 192, kernel_size=(3,3), stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(192, 192, 3, stride=2),
            nn.BatchNorm2d(192,
                       eps=0.001,  # value found in tensorflow
                       momentum=0.1,  # default pytorch value
                       affine=True),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        #print("stem_3:")
        #print(out.size())
        return out


class Inception_A(nn.Module):
    def __init__(self):
        super(Inception_A, self).__init__()
        #self.branch0 = BasicConv2d(384, 96, kernel_size=(1,1), stride=1)
        self.branch0 = nn.Sequential(
            nn.Conv2d(384, 96, 1, stride=1),
            nn.BatchNorm2d(96,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )
        self.branch1 = nn.Sequential(
            #BasicConv2d(384, 64, kernel_size=(1,1), stride=1),
            nn.Conv2d(384, 64, 1, stride=1),
            nn.BatchNorm2d(64,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(64, 96, kernel_size=(3,3), stride=1, padding=1)
            nn.Conv2d(64, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            #BasicConv2d(384, 64, kernel_size=(1,1), stride=1),
            nn.Conv2d(384, 64, 1, stride=1),
            nn.BatchNorm2d(64,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(64, 96, kernel_size=(3,3), stride=1, padding=1),
            nn.Conv2d(64, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(96, 96, kernel_size=(3,3), stride=1, padding=1)
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.BatchNorm2d(96,
                            eps=0.001,  # value found in tensorflow
                            momentum=0.1,  # default pytorch value
                            affine=True),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            #BasicConv2d(384, 96, kernel_size=(1,1), stride=1)
            nn.Conv2d(384, 96, 1, stride=1),
            nn.BatchNorm2d(96,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        #print("Inception-A:")
        #print(out.size())
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        #self.branch0 = BasicConv2d(384, 384, kernel_size=(3,3), stride=2)
        self.branch0 = nn.Sequential(
            nn.Conv2d(384, 384, 3, stride=2),
            nn.BatchNorm2d(384,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )
        self.branch1 = nn.Sequential(
            #BasicConv2d(384, 192, kernel_size=(1,1), stride=1),
            nn.Conv2d(384, 192, 1, stride=1),
            nn.BatchNorm2d(192,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(192, 224, kernel_size=(3,3), stride=1, padding=1),
            nn.Conv2d(192, 224, 3, stride=1, padding=1),
            nn.BatchNorm2d(224,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(224, 256, kernel_size=(3,3), stride=2)
            nn.Conv2d(224, 256, 3, stride=2),
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        #print("Reduction-A:")
        #print(out.size())
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        #self.branch0 = BasicConv2d(1024, 384, kernel_size=(1,1), stride=1)
        self.branch0 = nn.Sequential(
            nn.Conv2d(1024, 384, 1, stride=1),
            nn.BatchNorm2d(384,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )
        self.branch1 = nn.Sequential(
            #BasicConv2d(1024, 192, kernel_size=(1,1), stride=1),
            nn.Conv2d(1024, 192, 1, stride=1),
            nn.BatchNorm2d(192,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            nn.Conv2d(192, 224, (7,1), stride=1, padding=(0,3)),
            nn.BatchNorm2d(224,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(224, 256, kernel_size=(7,1), stride=1, padding=(3,0))
            nn.Conv2d(224, 256, (1,7), stride=1, padding=(3,0)),
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            #BasicConv2d(1024, 192, kernel_size=(1,1), stride=1),
            nn.Conv2d(1024, 192, 1, stride=1),
            nn.BatchNorm2d(192,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(192, 192, kernel_size=(7,1), stride=1, padding=(3,0)),
            nn.Conv2d(192, 192, (1,7), stride=1, padding=(3,0)),
            nn.BatchNorm2d(192,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            nn.Conv2d(192, 224, (7,1), stride=1, padding=(0,3)),
            nn.BatchNorm2d(224,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(224, 224, kernel_size=(7,1), stride=1, padding=(3,0)),
            nn.Conv2d(224, 224, (1,7), stride=1, padding=(3,0)),
            nn.BatchNorm2d(224,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(224, 256, kernel_size=(1,7), stride=1, padding=(0,3))
            nn.Conv2d(224, 256, (7,1), stride=1, padding=(0,3)),
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            #BasicConv2d(1024, 128, kernel_size=(1,1), stride=1)
            nn.Conv2d(1024, 128, 1, stride=1),
            nn.BatchNorm2d(128,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        #print("Inception-B:")
        #print(out.size())
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            #BasicConv2d(1024, 192, kernel_size=(1,1), stride=1),
            nn.Conv2d(1024, 192, 1, stride=1),
            nn.BatchNorm2d(192,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(192, 192, kernel_size=(3,3), stride=2)
            nn.Conv2d(192, 192, 3, stride=2),
            nn.BatchNorm2d(192,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )

        self.branch1 = nn.Sequential(
            #BasicConv2d(1024, 256, kernel_size=(1,1), stride=1),
            nn.Conv2d(1024, 256, 1, stride=1),
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(256, 256, kernel_size=(1,7), stride=1, padding=(0,3)),
            nn.Conv2d(256, 256, (1,7), stride=1, padding=(0,3)),
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(256, 320, kernel_size=(7,1), stride=1, padding=(3,0)),
            nn.Conv2d(256, 320, (7,1), stride=1, padding=(3,0)),
            nn.BatchNorm2d(320,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            #BasicConv2d(320, 320, kernel_size=(3,3), stride=2)
            nn.Conv2d(320, 320, 3, stride=2),
            nn.BatchNorm2d(320,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        #print("Reduction-B:")
        #print(out.size())
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()

        #self.branch0 = BasicConv2d(1536, 256, kernel_size=(1,1), stride=1)
        self.branch0 = nn.Sequential(
            nn.Conv2d(1536, 256, 1, stride=1),
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )
        #self.branch1_0 = BasicConv2d(1536, 384, kernel_size=(1,1), stride=1)
        self.branch1_0 = nn.Sequential(
            nn.Conv2d(1536, 384, 1, stride=1),
            nn.BatchNorm2d(384,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )
        #self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_1a = nn.Sequential(
            nn.Conv2d(384, 256, (1,3), stride=1, padding=(0,1)),
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )
        #self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch1_1b = nn.Sequential(
            nn.Conv2d(384, 256, (3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )
        #self.branch2_0 = BasicConv2d(1536, 384, kernel_size=(1,1), stride=1)
        self.branch2_0 = nn.Sequential(
            nn.Conv2d(1536, 384, 1, stride=1),
            nn.BatchNorm2d(384,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )
        #self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_1 = nn.Sequential(
            nn.Conv2d(384, 448, (3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(448,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )
        #self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_2 = nn.Sequential(
            nn.Conv2d(448, 512, (1,3), stride=1, padding=(0,1)),
            nn.BatchNorm2d(512,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )
        #self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3a = nn.Sequential(
            nn.Conv2d(512, 256, (1,3), stride=1, padding=(0,1)),
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )
        #self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_3b = nn.Sequential(
            nn.Conv2d(512, 256, (3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            #BasicConv2d(1536, 256, kernel_size=(1,1), stride=1)
            nn.Conv2d(1536, 256, 1, stride=1),
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        #print("Inception-C:")
        #print(out.size())
        return out


class UNetEnc(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class ASPP1(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP1, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
        self.endpool=nn.MaxPool2d(4,stride=1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        pool=self.endpool(net)
        return pool


class ASPP2(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP2, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        #self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 4, depth, 1, 1)
        self.endpool=nn.MaxPool2d(2,stride=1)
    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        #atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12], dim=1))
        pool=self.endpool(net)
        return pool



class inception_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=2,dilation=(2,2)),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
                      nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
                      nn.BatchNorm2d(in_channels // 2),
                      nn.ReLU(inplace=True),
                  ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)
    def forward(self,x):
        return self.encode(x)








class InceptionV4(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        #299*299*3--35*35*384
        self.stems = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.BatchNorm2d(32,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            # BasicConv2d(32, 32, kernel_size=(3,3), stride=1),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.BatchNorm2d(32,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            # BasicConv2d(32, 64, kernel_size=(3,3), stride=1, padding=1),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=True),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a()
        )
        self.aspp1=nn.Sequential(
            ASPP1(384, 256)
        )

        # 35*35*384--17*17*1024
        self.inceptionA = nn.Sequential(
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A()
        )
        self.center = nn.Sequential(
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(),
            nn.MaxPool2d(2, 2)
        )
        self.aspp2 = ASPP2(1024, 512)
        # 17*17*1024--8*8*1536
        self.inceptionB = nn.Sequential(
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B()

        )
        self.inceptionC = nn.Sequential(
            Inception_C(),
            Inception_C(),
            Inception_C()
        )
        self.enc3 = inception_Decoder(2560, 1024, 1)
        self.enc2 = inception_Decoder(1536, 512, 0)
        self.enc2_1 = inception_Decoder(768, 384, 0)
        self.enc1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(384, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 3, padding=1)
    def forward(self, x):
        enc1 = self.stems(x)
        #print('enc1: ', enc1.size())
        aspp1 = self.aspp1(enc1)
        #print('aspp1: ',aspp1.size())
        enc2 = self.inceptionA(enc1)
        #print('enc2: ', enc2.size())
        aspp2=self.aspp2(enc2)
        #print('aspp2: ', aspp2.size())
        enc3 = self.inceptionB(enc2)
        #print('enc3: ', enc3.size())
        enc4 = self.inceptionC(enc3)
        center = self.center(enc1)
        center = torch.cat([center, enc4], 1)
        #print('enc4: ', enc4.size())
        #cen = torch.cat([aspp1,enc4],1)
        dec3 = self.enc3(center)
        #print('dec3: ', dec3.size())
        fusion1=torch.cat([aspp2,dec3],1)
        #print("dec3:")
        #print(dec3.size())
        dec2 = self.enc2(fusion1)
        #print("dec2:")
        #print(dec2.size())
        fusion2=torch.cat([aspp1,dec2],1)
        dec2_1 = self.enc2_1(fusion2)
        #print("dec2_1:")
        #print(dec2_1.size())
        dec1 = self.enc1(dec2_1)
        #print("dec1:")
        #print(dec1.size())
        f=F.upsample_bilinear(self.final(dec1), x.size()[2:])
        #print("final:")
        #print(f.size())

        return f




