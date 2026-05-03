"""
Attention U-Net baseline (Oktay et al., 2018).

Same ResNet34 encoder as StandardUNet — only the skip-handling differs.
At every decoder level, the encoder skip `x` is gated by an attention
coefficient α ∈ [0, 1] computed from `x` and the upsampled gating signal `g`.

The gate uses 1×1 convs at every level; that uniform 1×1 choice is precisely
what DKAG replaces with a level-adaptive kernel.

Reference:
    Oktay, O. et al. "Attention U-Net: Learning Where to Look for the
    Pancreas." MIDL 2018.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBnRelu, ResNet34Encoder


class AttentionGate(nn.Module):
    """Standard 1×1 attention gate from Oktay et al. (2018)."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class AttDecoderBlock(nn.Module):
    """Up → attention-gate skip → concat → 2× ConvBnRelu."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.attn = AttentionGate(F_g=out_c, F_l=skip_c, F_int=out_c // 2)
        self.conv = nn.Sequential(
            ConvBnRelu(out_c + skip_c, out_c),
            ConvBnRelu(out_c, out_c),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, self.attn(g=x, x=skip)], dim=1))


class AttentionUNet(nn.Module):
    """Oktay-style Attention U-Net with ResNet34 encoder. 1×1 gates at every level."""

    def __init__(self):
        super().__init__()
        self.encoder    = ResNet34Encoder(pretrained=True)
        self.bottleneck = nn.Sequential(
            ConvBnRelu(512, 512),
            ConvBnRelu(512, 512),
            nn.Dropout2d(0.3),
        )
        self.dec4 = AttDecoderBlock(512, 256, 256)
        self.dec3 = AttDecoderBlock(256, 128, 128)
        self.dec2 = AttDecoderBlock(128,  64,  64)
        self.dec1 = AttDecoderBlock( 64,  64,  32)
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
