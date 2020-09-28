import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os
import sys


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        #self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch0 = nn.Sequential(
            nn.Conv2d(192, 96, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(96,
                        eps=0.001,  # value found in tensorflow
                        momentum=0.1,  # default pytorch value
                        affine=True),
            nn.ReLU(inplace=False)
        )
        self.branch1 = nn.Sequential(
            #BasicConv2d(192, 48, kernel_size=1, stride=1),
            nn.Conv2d(192, 48, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(48,
                       eps=0.001,  # value found in tensorflow
                       momentum=0.1,  # default pytorch value
                       affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
            nn.Conv2d(48, 64, 5, stride=1,padding=2),  # verify bias false
            nn.BatchNorm2d(64,
                                       eps=0.001,  # value found in tensorflow
                                       momentum=0.1,  # default pytorch value
                                       affine=True),
            nn.ReLU(inplace=False)
        )

        self.branch2 = nn.Sequential(
            #BasicConv2d(192, 64, kernel_size=1, stride=1),
            nn.Conv2d(192, 64, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(64,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 96, 3, stride=1,padding=1),  # verify bias false
            nn.BatchNorm2d(96,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
            nn.Conv2d(96, 96, 3, stride=1, padding=1),  # verify bias false
            nn.BatchNorm2d(96,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            #BasicConv2d(192, 64, kernel_size=1, stride=1)
            nn.Conv2d(192, 64, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(64,
                       eps=0.001,  # value found in tensorflow
                       momentum=0.1,  # default pytorch value
                       affine=True),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        #self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch0=nn.Sequential(
            nn.Conv2d(320, 32, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(32,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False)
        )
        self.branch1 = nn.Sequential(
            #BasicConv2d(320, 32, kernel_size=1, stride=1),
            nn.Conv2d(320, 32, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(32,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, stride=1,padding=1),  # verify bias false
            nn.BatchNorm2d(32,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False)

        )

        self.branch2 = nn.Sequential(
            #BasicConv2d(320, 32, kernel_size=1, stride=1),
            nn.Conv2d(320, 32, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(32,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 48, 3, stride=1,padding=1),  # verify bias false
            nn.BatchNorm2d(48,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(48, 64, 3, stride=1,padding=1),  # verify bias false
            nn.BatchNorm2d(64,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False)
        )

        self.conv2d = nn.Conv2d(128, 320, 1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        #self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch0=nn.Sequential(
            nn.Conv2d(320, 384, 3, stride=2),  # verify bias false
            nn.BatchNorm2d(384,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False)
        )
        self.branch1 = nn.Sequential(
            #BasicConv2d(320, 256, kernel_size=1, stride=1),
            nn.Conv2d(320, 256, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, 3, stride=1,padding=1),  # verify bias false
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(256, 384, kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, stride=2),  # verify bias false
            nn.BatchNorm2d(384,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        #self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch0=nn.Sequential(
            nn.Conv2d(1088, 192, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(192,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False)
        )
        self.branch1 = nn.Sequential(
            #BasicConv2d(1088, 128, kernel_size=1, stride=1),
            nn.Conv2d(1088, 128, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(128,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
            nn.Conv2d(128, 160, (1,7), stride=1,padding=(0,3)),  # verify bias false
            nn.BatchNorm2d(160,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0)),
            nn.Conv2d(160, 192, (7, 1), stride=1, padding=(3, 0)),  # verify bias false
            nn.BatchNorm2d(192,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
        )

        self.conv2d = nn.Conv2d(384, 1088, 1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            #BasicConv2d(1088, 256, kernel_size=1, stride=1),
            nn.Conv2d(1088, 256, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(256, 384, kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, stride=2),  # verify bias false
            nn.BatchNorm2d(384,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False)
        )

        self.branch1 = nn.Sequential(
            #BasicConv2d(1088, 256, kernel_size=1, stride=1),
            nn.Conv2d(1088, 256, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(256, 288, kernel_size=3, stride=2),
            nn.Conv2d(256, 288, 3, stride=2),  # verify bias false
            nn.BatchNorm2d(288,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False)
        )

        self.branch2 = nn.Sequential(
            #BasicConv2d(1088, 256, kernel_size=1, stride=1),
            nn.Conv2d(1088, 256, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 288, 3, stride=1,padding=1),  # verify bias false
            nn.BatchNorm2d(288,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(288, 320, kernel_size=3, stride=2)
            nn.Conv2d(288, 320, 3, stride=2),  # verify bias false
            nn.BatchNorm2d(320,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        #self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch0=nn.Sequential(
            nn.Conv2d(2080, 192, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(192,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False)
        )
        self.branch1 = nn.Sequential(
            #BasicConv2d(2080, 192, kernel_size=1, stride=1),
            nn.Conv2d(2080, 192, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(192,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
            nn.Conv2d(192, 224, (1,3), stride=1,padding=(0,1)),  # verify bias false
            nn.BatchNorm2d(224,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(224, 256, (3,1), stride=1,padding=(1,0)),  # verify bias false
            nn.BatchNorm2d(256,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False)
        )

        self.conv2d = nn.Conv2d(448, 2080, 1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class inception_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1,dilation=(2,2)),
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
        self.decode = nn.Sequential(*layers)
    def forward(self,x):
        return self.decode(x)


class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        # Modules
        self.stem=nn.Sequential(
            #self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
            nn.Conv2d(3, 32, 3, stride=2),  # verify bias false
            nn.BatchNorm2d(32,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
            nn.Conv2d(32, 32, 3, stride=1),  # verify bias false
            nn.BatchNorm2d(32,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
            nn.Conv2d(32, 64, 3, stride=1,padding=1),  # verify bias false
            nn.BatchNorm2d(64,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2),
            #self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
            nn.Conv2d(64, 80, 1, stride=1),  # verify bias false
            nn.BatchNorm2d(80,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            #self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
            nn.Conv2d(80, 192, 3, stride=1),  # verify bias false
            nn.BatchNorm2d(192,
                           eps=0.001,  # value found in tensorflow
                           momentum=0.1,  # default pytorch value
                           affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2),
            Mixed_5b()
        )
        self.repeat = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )


        '''
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)
        '''

        self.dec4 = inception_Decoder(2080, 512, 1)
        self.dec3 = inception_Decoder(512, 256, 0)
        self.dec2 = inception_Decoder(256, 128, 0)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 3, padding=1)



    def forward(self, input):
        x = self.stem(input)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        #x = self.Block8(x)
        #x = self.conv2d_7b(x)
        dec = self.dec4(x)
        dec = self.dec3(dec)
        dec = self.dec2(dec)
        dec = self.dec1(dec)
        f = F.upsample_bilinear(self.final(dec), input.size()[2:])
        return f
