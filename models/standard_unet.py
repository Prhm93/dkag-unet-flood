"""
Standard U-Net baseline.

ImageNet-pretrained ResNet34 encoder + plain skip-connection decoder.
No attention; encoder skips are concatenated directly to the upsampled
decoder feature map without any filtering.
"""
import torch
import torch.nn as nn

from .blocks import ConvBnRelu, DecoderBlock, ResNet34Encoder


class StandardUNet(nn.Module):
    """ResNet34 + plain U-Net decoder. The reference baseline."""

    def __init__(self):
        super().__init__()
        self.encoder    = ResNet34Encoder(pretrained=True)
        self.bottleneck = nn.Sequential(
            ConvBnRelu(512, 512),
            ConvBnRelu(512, 512),
            nn.Dropout2d(0.3),
        )
        # in_c   = channels coming in (bottleneck or previous decoder)
        # skip_c = encoder skip channels at this level
        # out_c  = decoder output channels
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128,  64,  64)
        self.dec1 = DecoderBlock( 64,  64,  32)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            ConvBnRelu(16, 16),
        )
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0, s1, s2, s3, s4 = self.encoder(x)
        x = self.bottleneck(s4)
        x = self.dec4(x, s3)
        x = self.dec3(x, s2)
        x = self.dec2(x, s1)
        x = self.dec1(x, s0)
        x = self.dec0(x)
        return self.head(x)
