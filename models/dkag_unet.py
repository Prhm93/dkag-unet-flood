"""
DKAG U-Net — Dynamic Kernel Attention Gate (proposed).

The ONLY structural change relative to Oktay's attention gate is the
projection of the gating signal `W_g`. Instead of a 1×1 conv at every
decoder level, DKAG uses a depthwise-separable conv whose kernel size
adapts to the decoder level:

    dec4 (16×16 spatial) → W_g kernel = 5×5     (coarse, semantic)
    dec3 (32×32 spatial) → W_g kernel = 5×5
    dec2 (64×64 spatial) → W_g kernel = 3×3
    dec1 (128×128 spatial) → W_g kernel = 1×1   (low-level, pointwise)

Design rule: ONE conv per signal, ONE gradient path per gate.
No branch summing, no routing, no fusion. The added parameters relative
to Oktay total ≈ 10 K (≈ 0.035 % of the model).

Set `gate_kernel=1` and DKAG reduces exactly to Oktay's gate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBnRelu, ResNet34Encoder


class DKAttentionGate(nn.Module):
    """
    Dynamic Kernel Attention Gate.

    Parameters
    ----------
    F_g : int
        Channels of the gating (decoder) signal `g`.
    F_l : int
        Channels of the encoder skip feature `x`.
    F_int : int
        Intermediate (bottleneck) channels inside the gate.
    gate_kernel : int, default = 1
        Kernel size used by the depthwise-separable W_g.
        - `gate_kernel == 1` → identical to Oktay's gate.
        - `gate_kernel  > 1` → depthwise (k×k, groups=F_g) + pointwise (1×1).
    """

    def __init__(self, F_g: int, F_l: int, F_int: int, gate_kernel: int = 1):
        super().__init__()
        if gate_kernel > 1:
            # Depthwise-separable conv:
            #   - depthwise kxk (one filter per channel) gives local context
            #   - pointwise 1x1 mixes channels and projects to F_int
            self.W_g = nn.Sequential(
                nn.Conv2d(
                    F_g, F_g,
                    kernel_size=gate_kernel,
                    padding=gate_kernel // 2,
                    groups=F_g, bias=False,
                ),
                nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
                nn.BatchNorm2d(F_int),
            )
        else:
            # Identical to Oktay (1×1 only)
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


class DKDecoderBlock(nn.Module):
    """Up → DKAG-gated skip → concat → 2× ConvBnRelu."""

    def __init__(self, in_c: int, skip_c: int, out_c: int, gate_kernel: int = 1):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.attn = DKAttentionGate(
            F_g=out_c, F_l=skip_c, F_int=out_c // 2,
            gate_kernel=gate_kernel,
        )
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


class DKAGUNet(nn.Module):
    """
    DKAG U-Net (proposed).

    Same ResNet34 encoder and decoder topology as the Oktay AttentionUNet.
    The only difference is the level-adaptive kernel inside `W_g`:
    {dec4: 5, dec3: 5, dec2: 3, dec1: 1}.
    """

    GATE_KERNELS = {"dec4": 5, "dec3": 5, "dec2": 3, "dec1": 1}

    def __init__(self):
        super().__init__()
        self.encoder    = ResNet34Encoder(pretrained=True)
        self.bottleneck = nn.Sequential(
            ConvBnRelu(512, 512),
            ConvBnRelu(512, 512),
            nn.Dropout2d(0.3),
        )
        self.dec4 = DKDecoderBlock(512, 256, 256, gate_kernel=self.GATE_KERNELS["dec4"])
        self.dec3 = DKDecoderBlock(256, 128, 128, gate_kernel=self.GATE_KERNELS["dec3"])
        self.dec2 = DKDecoderBlock(128,  64,  64, gate_kernel=self.GATE_KERNELS["dec2"])
        self.dec1 = DKDecoderBlock( 64,  64,  32, gate_kernel=self.GATE_KERNELS["dec1"])
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
