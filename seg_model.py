import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.enc_conv1 = self.conv_block(in_channels, 32)
        self.enc_conv2 = self.conv_block(32, 64)
        self.enc_conv3 = self.conv_block(64, 128)

        self.bottleneck_conv = self.bottleneck(128, 128)

        self.dec_conv2 = self.deconv_block(128 + 128, 64)
        self.dec_conv3 = self.deconv_block(64 + 64, 32)
        self.dec_conv4 = self.deconv_block(32 + 32, 32)

        self.output_layer = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

    def bottleneck(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(enc1)
        enc3 = self.enc_conv3(enc2)

        bottleneck = self.bottleneck_conv(enc3)

        dec2 = self.dec_conv2(torch.cat([enc3, bottleneck], dim=1))
        dec3 = self.dec_conv3(torch.cat([enc2, dec2], dim=1))
        dec4 = self.dec_conv4(torch.cat([enc1, dec3], dim=1))

        output = self.output_layer(dec4)
        # create segmentation map from 5 classes
        output = F.softmax(output, dim=1)
        _, output = torch.max(x, dim=1)
        
        return output
