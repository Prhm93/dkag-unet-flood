# DKAG U-Net — Dynamic Kernel Attention Gate for Flood Image Segmentation

PyTorch implementation of a level-aware attention gate for ground-level flood image segmentation, with reproducible baselines (Standard U-Net and Oktay Attention U-Net) under an identical ImageNet-pretrained ResNet34 encoder.

> Submitted to **IPGRC 2026**.
> Author: Parham Imanzadeh Charandabi — University of Salford .

---

## What is DKAG?

The **Dynamic Kernel Attention Gate (DKAG)** is a one-line modification of the standard attention gate of Oktay et al. (2018). Everything in the gate is preserved except the projection of the gating signal `W_g`, which is replaced by a **depthwise-separable convolution whose kernel size adapts to the decoder level**:

| Decoder level | Spatial size | Encoder skip channels | DKAG kernel |
|---|---|---|---|
| dec4 | 16 × 16 | 256 | **5 × 5** |
| dec3 | 32 × 32 | 128 | **5 × 5** |
| dec2 | 64 × 64 | 64 | **3 × 3** |
| dec1 | 128 × 128 | 64 | **1 × 1** |

The intuition: deep decoder levels carry small, semantic feature maps where a wider kernel gives the gate useful local context; shallow levels carry large, low-level texture features where a 1×1 kernel keeps the gate stable. Total added parameters versus the Oktay gate: **≈ 10 K (≈ 0.035 % of the model)** — DKAG adds 10,176 parameters on top of Oktay's 29.16 M.

The design rule is: **one convolution per signal, one gradient path per gate**. No branch summing, no routing, no fusion modules.

---

## Results

All three models share the **identical** ImageNet-pretrained ResNet34 encoder. Only the skip-handling differs. Reported on a held-out 20% validation split of a 320-image custom ground-level flood dataset.

| Model                 | IoU         | Dice        | Precision   | Recall      | Accuracy    |
|-----------------------|-------------|-------------|-------------|-------------|-------------|
| Standard U-Net        | 90.35 %     | 94.69 %     | —           | —           | —           |
| Attention U-Net (Oktay) | 89.23 %   | 93.99 %     | —           | —           | —           |
| **DKAG U-Net (ours)** | **93.76 %** | **96.65 %** | —           | —           | —           |

Run `python train.py` end-to-end to reproduce. Final numbers in your run will be written to `runs/<timestamp>/results.json`.

---

## Repository structure

```
dkag-unet-flood/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── config.py                  # all hyper-parameters in one place
├── train.py                   # main training entry point
├── models/
│   ├── __init__.py
│   ├── blocks.py              # ConvBnRelu, ResNet34Encoder, DecoderBlock
│   ├── standard_unet.py       # Baseline: ResNet34 + plain skips
│   ├── attention_unet.py      # Oktay 2018 attention gate baseline
│   └── dkag_unet.py           # Proposed: Dynamic Kernel Attention Gate
├── utils/
│   ├── __init__.py
│   ├── dataset.py             # FloodDataset + augmentations
│   ├── losses.py              # BCE-Dice loss
│   ├── metrics.py             # IoU, Dice, Precision, Recall, Accuracy
│   └── trainer.py             # train / validate loops
└── notebooks/
    └── colab_notebook.py      # single-file Colab-friendly version
```

---

## Installation

Tested with Python 3.10, PyTorch 2.2 (CUDA 12.1), Ubuntu 22.04.

```bash
git clone https://github.com/<your-username>/dkag-unet-flood.git
cd dkag-unet-flood
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you only have CPU, the code runs but training will be slow. A 12 GB GPU comfortably handles `BATCH_SIZE=8` at `IMG_SIZE=256`.

---

## Dataset format

The training script expects two parallel directories — one of RGB images and one of binary masks — with **matching filenames** (the extension may differ):

```
data/
├── images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── masks/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

Masks are read as grayscale and binarised at `> 127` (flood = 1, non-flood = 0). Supported mask extensions: `.png .jpg .jpeg .bmp .tif .tiff`.

Set `IMG_DIR` and `MASK_DIR` in `config.py` (or pass them as CLI flags — see below).

---

## Usage

### Train all three models end-to-end

```bash
python train.py \
    --img-dir   /path/to/data/images \
    --mask-dir  /path/to/data/masks \
    --save-dir  ./runs/exp1 \
    --epochs    60 \
    --batch-size 8
```

This will train each of `StandardUNet`, `AttentionUNet`, and `DKAGUNet` in turn, save the best-IoU checkpoint for each, write a learning-curve plot, and print a final comparison table.

### Train a single model

```bash
python train.py --model dkag --epochs 60
# choices: standard | attention | dkag | all  (default: all)
```

### Programmatic use

```python
import torch
from models import DKAGUNet

model = DKAGUNet().eval().cuda()
ckpt  = torch.load("runs/exp1/DKAG_U-Net_best.pth", map_location="cuda")
model.load_state_dict(ckpt["model_state"])

with torch.no_grad():
    logits = model(image_tensor.cuda())  # image_tensor: (B, 3, 256, 256), ImageNet-normalised
    mask   = (torch.sigmoid(logits) > 0.5).float()
```

---

## Notes on reproducibility

- All three models share the same encoder, the same loss (BCE + Dice, weighted 0.5/0.5), the same optimiser (AdamW, lr 1e-4, weight decay 1e-5), and the same cosine-annealed LR schedule. Only the skip-handling block differs.
- Seed is set at 42 across `random`, `numpy`, and `torch`. Mixed-precision training (AMP) is enabled, which introduces minor non-determinism on some GPUs even with fixed seeds — expect ±0.3 % run-to-run drift in IoU.
- Augmentations follow the original notebook: Albumentations stack with horizontal/vertical flips, 90° rotations, shift–scale–rotate, brightness/contrast, hue/saturation, Gaussian noise, elastic transform, and coarse dropout.

---

## Citation

If you use this code or the DKAG idea, please cite:

```bibtex
@inproceedings{imanzadeh2026dkag,
  title     = {Dynamic Kernel Attention Gate for Ground-Level Flood Image Segmentation},
  author    = {{Imanzadeh Charandabi}, Parham and Palaiahnakote, Shivakumara and Antonacopoulos, Apostolos},
  booktitle = {IPGRC 2026},
  year      = {2026},
  note      = {To appear}
}
```

The baseline attention gate is from:

> Oktay, O. et al. *Attention U-Net: Learning Where to Look for the Pancreas.* MIDL 2018.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

The pretrained ResNet34 weights are loaded from `torchvision` (ImageNet-1K), licensed under the BSD 3-Clause License of the PyTorch project.
