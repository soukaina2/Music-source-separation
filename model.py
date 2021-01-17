'''Imports'''

import torch
import torch.nn as nn
import torch.nn.functional as F

'''The model'''
class Leaky_conv2d(nn.Module):
    '''(conv2d => BatchNorm2d => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(Leaky_conv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Simple_conv2d(nn.Module):
    '''(conv2d => BatchNorm2d => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(Simple_conv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = Leaky_conv2d(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            Leaky_conv2d(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights

        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = Simple_conv2d(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 1):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 16)
        self.down1 = down(16, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, 256)
        self.down5 = down(256, 512)
        self.down6 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 32)
        self.up5 = up(64, 16)
        self.up6 = up(32, 16)
        self.outc = outconv(16, n_classes)

    def forward(self, x):
        self.x1 = self.inc(x)
        self.x2 = self.down1(self.x1)
        self.x3 = self.down2(self.x2)
        self.x4 = self.down3(self.x3)
        self.x5 = self.down4(self.x4)
        self.x6 = self.down5(self.x5)
        self.x7 = self.down6(self.x6)
        self.x8 = self.up1(self.x7, self.x6)
        self.x9 = self.up2(self.x8, self.x5)
        self.x10 = self.up3(self.x9, self.x4)
        self.x11 = self.up4(self.x10, self.x3)
        self.x12 = self.up5(self.x11, self.x2)
        self.x13 = self.up6(self.x12, self.x1)
        self.y = self.outc(self.x13)
        return self.y
