"""
Shared building blocks used by all three U-Net variants.

Keeping the encoder and the basic conv block identical across models means
any performance difference is attributable to the skip-handling block alone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBnRelu(nn.Module):
    """Conv → BN → ReLU. The basic unit of the decoder."""

    def __init__(self, in_c: int, out_c: int, kernel: int = 3, pad: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Plain decoder block: transpose-conv up → concat skip → 2× ConvBnRelu.

    This is the baseline skip-handling — no attention, no filtering — used
    by `StandardUNet`.
    """

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(out_c + skip_c, out_c),
            ConvBnRelu(out_c, out_c),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNet34Encoder(nn.Module):
    """
    ImageNet-pretrained ResNet34 backbone, exposed as five skip outputs.

    All three U-Net variants in this repo use this *exact* encoder, so any
    accuracy delta is attributable to the decoder/gate, not the encoder.

    Skip outputs (input H × W = 256 × 256):
        s0 : 64ch,  H/2  = 128 × 128   (stem: conv1 + bn1 + relu)
        s1 : 64ch,  H/4  =  64 ×  64   (layer1)
        s2 : 128ch, H/8  =  32 ×  32   (layer2)
        s3 : 256ch, H/16 =  16 ×  16   (layer3)
        s4 : 512ch, H/32 =   8 ×   8   (layer4 — bottleneck input)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = (
            models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )
        bb = models.resnet34(weights=weights)
        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.pool   = bb.maxpool
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4

    def forward(self, x: torch.Tensor):
        s0 = self.stem(x)        # H/2
        p  = self.pool(s0)       # H/4
        s1 = self.layer1(p)      # H/4
        s2 = self.layer2(s1)     # H/8
        s3 = self.layer3(s2)     # H/16
        s4 = self.layer4(s3)     # H/32
        return s0, s1, s2, s3, s4
