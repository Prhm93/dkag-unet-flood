"""
DKAG U-Net — single-file Colab-friendly version.

Paste this into a Colab notebook (or split it cell-by-cell along the
`# %%` markers) for a one-shot reproduction of the IPGRC 2026 experiments.

Sections:
    1. Mount Drive & install
    2. Imports
    3. Configuration
    4. Dataset & augmentation
    5. Loss & metrics
    6. Shared building blocks (ResNet34 encoder)
    7. Model 1 — Standard U-Net
    8. Model 2 — Attention U-Net (Oktay)
    9. Model 3 — DKAG U-Net (proposed)
    10. Training engine
    11. Train all three models
    12. Plot learning curves
"""

# %% 1. MOUNT DRIVE & INSTALL ────────────────────────────────────────────────
# Uncomment when running in Colab:
# from google.colab import drive
# drive.mount("/content/drive")
#
# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pip", "install", "-q",
#                 "albumentations", "tqdm"], check=True)


# %% 2. IMPORTS ───────────────────────────────────────────────────────────────
import os, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")


# %% 3. CONFIGURATION ────────────────────────────────────────────────────────
# ── Paths (edit these for your environment) ──────────────────────────────────
IMG_DIR  = "/content/drive/MyDrive/your_dataset/images"
MASK_DIR = "/content/drive/MyDrive/your_dataset/masks"
SAVE_DIR = "/content/drive/MyDrive/dkag_unet_runs/exp1"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
IMG_SIZE     = 256
BATCH_SIZE   = 8
NUM_EPOCHS   = 60
LR           = 1e-4
WEIGHT_DECAY = 1e-5
VAL_SPLIT    = 0.2
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


set_seed()


# %% 4. DATASET & AUGMENTATION ───────────────────────────────────────────────
TRAIN_AUG = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                       rotate_limit=15, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.2,
                               contrast_limit=0.2, p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                         val_shift_limit=10, p=0.3),
    A.GaussNoise(var_limit=(5, 30), p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    A.CoarseDropout(max_holes=4, max_height=32,
                    max_width=32, fill_value=0, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

VAL_AUG = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_mask(stem):
    for ext in MASK_EXTS:
        p = os.path.join(MASK_DIR, stem + ext)
        if os.path.exists(p):
            return p
    return None


class FloodDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs     = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)
        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        return img, mask.unsqueeze(0)


def get_dataloaders():
    all_imgs = sorted(
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    pairs, missing = [], 0
    for fname in all_imgs:
        stem = os.path.splitext(fname)[0]
        mp   = find_mask(stem)
        if mp:
            pairs.append((os.path.join(IMG_DIR, fname), mp))
        else:
            missing += 1
    print(f"Valid pairs : {len(pairs)}  |  Missing masks: {missing}")

    random.seed(SEED); random.shuffle(pairs)
    n_val       = int(len(pairs) * VAL_SPLIT)
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    train_dl = DataLoader(FloodDataset(train_pairs, TRAIN_AUG),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(FloodDataset(val_pairs, VAL_AUG),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
    return train_dl, val_dl


# %% 5. LOSS & METRICS ───────────────────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-5):
        super().__init__()
        self.bce        = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth     = smooth

    def forward(self, logits, targets):
        bce   = self.bce(logits, targets)
        prob  = torch.sigmoid(logits)
        inter = (prob * targets).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice  = 1.0 - (2 * inter + self.smooth) / (union + self.smooth)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice.mean()


def compute_metrics(logits, targets, threshold=0.5, smooth=1e-6):
    pred = (torch.sigmoid(logits) > threshold).float()
    targ = targets.float()
    TP = (pred * targ).sum(dim=(2, 3))
    FP = (pred * (1 - targ)).sum(dim=(2, 3))
    FN = ((1 - pred) * targ).sum(dim=(2, 3))
    TN = ((1 - pred) * (1 - targ)).sum(dim=(2, 3))
    return {
        "iou"      : ((TP + smooth) / (TP + FP + FN + smooth)).mean().item(),
        "dice"     : ((2*TP + smooth) / (2*TP + FP + FN + smooth)).mean().item(),
        "precision": ((TP + smooth) / (TP + FP + smooth)).mean().item(),
        "recall"   : ((TP + smooth) / (TP + FN + smooth)).mean().item(),
        "accuracy" : ((TP + TN + smooth) / (TP + TN + FP + FN + smooth)).mean().item(),
    }


# %% 6. SHARED BLOCKS — ResNet34 encoder ─────────────────────────────────────
class ConvBnRelu(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.block(x)


class DecoderBlock(nn.Module):
    """Plain upsample → concat skip → conv×2 (Standard U-Net)."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(out_c + skip_c, out_c),
            ConvBnRelu(out_c, out_c),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNet34Encoder(nn.Module):
    """ImageNet-pretrained ResNet34. Identical across all three models."""
    def __init__(self, pretrained=True):
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

    def forward(self, x):
        s0 = self.stem(x)
        p  = self.pool(s0)
        s1 = self.layer1(p)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return s0, s1, s2, s3, s4


# %% 7. MODEL 1 — Standard U-Net ─────────────────────────────────────────────
class StandardUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = ResNet34Encoder(pretrained=True)
        self.bottleneck = nn.Sequential(
            ConvBnRelu(512, 512), ConvBnRelu(512, 512), nn.Dropout2d(0.3))
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128,  64,  64)
        self.dec1 = DecoderBlock( 64,  64,  32)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2), ConvBnRelu(16, 16))
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        s0, s1, s2, s3, s4 = self.encoder(x)
        x = self.bottleneck(s4)
        x = self.dec4(x, s3); x = self.dec3(x, s2)
        x = self.dec2(x, s1); x = self.dec1(x, s0)
        return self.head(self.dec0(x))


# %% 8. MODEL 2 — Attention U-Net (Oktay 2018) ───────────────────────────────
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False),
                                  nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False),
                                  nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True),
                                  nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class AttDecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.attn = AttentionGate(F_g=out_c, F_l=skip_c, F_int=out_c // 2)
        self.conv = nn.Sequential(ConvBnRelu(out_c + skip_c, out_c),
                                  ConvBnRelu(out_c, out_c))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, self.attn(g=x, x=skip)], dim=1))


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = ResNet34Encoder(pretrained=True)
        self.bottleneck = nn.Sequential(
            ConvBnRelu(512, 512), ConvBnRelu(512, 512), nn.Dropout2d(0.3))
        self.dec4 = AttDecoderBlock(512, 256, 256)
        self.dec3 = AttDecoderBlock(256, 128, 128)
        self.dec2 = AttDecoderBlock(128,  64,  64)
        self.dec1 = AttDecoderBlock( 64,  64,  32)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2), ConvBnRelu(16, 16))
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        s0, s1, s2, s3, s4 = self.encoder(x)
        x = self.bottleneck(s4)
        x = self.dec4(x, s3); x = self.dec3(x, s2)
        x = self.dec2(x, s1); x = self.dec1(x, s0)
        return self.head(self.dec0(x))


# %% 9. MODEL 3 — DKAG U-Net (proposed) ──────────────────────────────────────
# Dynamic Kernel Attention Gate
#   Identical to Oktay's gate when gate_kernel == 1.
#   When gate_kernel > 1, W_g uses a depthwise-separable conv whose kernel
#   size adapts to the decoder level: {dec4: 5, dec3: 5, dec2: 3, dec1: 1}.
class DKAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, gate_kernel=1):
        super().__init__()
        if gate_kernel > 1:
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_g, gate_kernel,
                          padding=gate_kernel // 2,
                          groups=F_g, bias=False),         # depthwise
                nn.Conv2d(F_g, F_int, 1, bias=False),       # pointwise
                nn.BatchNorm2d(F_int))
        else:
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, 1, bias=False),
                nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class DKDecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, gate_kernel=1):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.attn = DKAttentionGate(F_g=out_c, F_l=skip_c,
                                    F_int=out_c // 2,
                                    gate_kernel=gate_kernel)
        self.conv = nn.Sequential(ConvBnRelu(out_c + skip_c, out_c),
                                  ConvBnRelu(out_c, out_c))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, self.attn(g=x, x=skip)], dim=1))


class DKAGUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = ResNet34Encoder(pretrained=True)
        self.bottleneck = nn.Sequential(
            ConvBnRelu(512, 512), ConvBnRelu(512, 512), nn.Dropout2d(0.3))
        self.dec4 = DKDecoderBlock(512, 256, 256, gate_kernel=5)
        self.dec3 = DKDecoderBlock(256, 128, 128, gate_kernel=5)
        self.dec2 = DKDecoderBlock(128,  64,  64, gate_kernel=3)
        self.dec1 = DKDecoderBlock( 64,  64,  32, gate_kernel=1)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2), ConvBnRelu(16, 16))
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        s0, s1, s2, s3, s4 = self.encoder(x)
        x = self.bottleneck(s4)
        x = self.dec4(x, s3); x = self.dec3(x, s2)
        x = self.dec2(x, s1); x = self.dec1(x, s0)
        return self.head(self.dec0(x))


# Parameter count comparison
print(f"\n{'Model':<30} {'Total':>8} {'Encoder':>9} {'Decoder':>9}")
print("-" * 58)
for name, Cls in [("Standard U-Net",          StandardUNet),
                  ("Attention U-Net",         AttentionUNet),
                  ("DKAG U-Net (Proposed)",   DKAGUNet)]:
    m = Cls()
    t = sum(p.numel() for p in m.parameters())
    e = sum(p.numel() for p in m.encoder.parameters())
    print(f"{name:<30} {t/1e6:>7.2f}M {e/1e6:>8.2f}M {(t-e)/1e6:>8.2f}M")


# %% 10. TRAINING ENGINE ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    agg = {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "accuracy": 0}
    for imgs, masks in tqdm(loader, leave=False, desc="  train"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(imgs)
            loss   = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
        for k, v in compute_metrics(logits.detach(), masks).items():
            agg[k] += v
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in agg.items()}


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    agg = {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "accuracy": 0}
    for imgs, masks in tqdm(loader, leave=False, desc="  val  "):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        with torch.amp.autocast("cuda"):
            logits = model(imgs)
            loss   = criterion(logits, masks)
        total_loss += loss.item()
        for k, v in compute_metrics(logits, masks).items():
            agg[k] += v
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in agg.items()}


def train_model(model_name, model, train_dl, val_dl):
    print(f"\n{'='*65}")
    print(f"  Training: {model_name}")
    print(f"{'='*65}")
    model     = model.to(DEVICE)
    criterion = BCEDiceLoss(bce_weight=0.5).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda")
    best_iou  = 0.0
    save_path = os.path.join(
        SAVE_DIR, f"{model_name.replace(' ', '_')}_best.pth")
    history = {k: [] for k in [
        "train_loss", "val_loss", "train_iou", "val_iou",
        "train_dice", "val_dice", "val_precision",
        "val_recall", "val_accuracy",
    ]}

    for epoch in range(1, NUM_EPOCHS + 1):
        t_loss, t_m = train_one_epoch(
            model, train_dl, optimizer, criterion, scaler)
        v_loss, v_m = validate(model, val_dl, criterion)
        scheduler.step()
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_iou"].append(t_m["iou"])
        history["val_iou"].append(v_m["iou"])
        history["train_dice"].append(t_m["dice"])
        history["val_dice"].append(v_m["dice"])
        history["val_precision"].append(v_m["precision"])
        history["val_recall"].append(v_m["recall"])
        history["val_accuracy"].append(v_m["accuracy"])
        if v_m["iou"] > best_iou:
            best_iou = v_m["iou"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "best_iou": best_iou, "val_metrics": v_m},
                       save_path)
            star = " ★ SAVED"
        else:
            star = ""
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:03d}/{NUM_EPOCHS} | "
                  f"Loss {v_loss:.4f} | IoU {v_m['iou']*100:.2f}% | "
                  f"Dice {v_m['dice']*100:.2f}% | "
                  f"Prec {v_m['precision']*100:.2f}% | "
                  f"Rec {v_m['recall']*100:.2f}% | "
                  f"Acc {v_m['accuracy']*100:.2f}% | "
                  f"LR {scheduler.get_last_lr()[0]:.2e}{star}")

    print(f"\n  ✔ Best Val IoU: {best_iou*100:.2f}%  → {save_path}")
    ckpt = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    _, final_m = validate(model, val_dl, BCEDiceLoss())
    print(f"  [Final best] "
          f"IoU={final_m['iou']*100:.2f}%  "
          f"Dice={final_m['dice']*100:.2f}%  "
          f"Prec={final_m['precision']*100:.2f}%  "
          f"Rec={final_m['recall']*100:.2f}%  "
          f"Acc={final_m['accuracy']*100:.2f}%")
    return model, history, final_m


# %% 11. TRAIN ALL THREE MODELS ──────────────────────────────────────────────
set_seed()
train_dl, val_dl = get_dataloaders()
print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}\n")

set_seed()
std_model, std_hist, std_m = train_model(
    "Standard U-Net", StandardUNet(), train_dl, val_dl)

set_seed()
att_model, att_hist, att_m = train_model(
    "Attention U-Net", AttentionUNet(), train_dl, val_dl)

set_seed()
dkag_model, dkag_hist, dkag_m = train_model(
    "DKAG U-Net (Proposed)", DKAGUNet(), train_dl, val_dl)

print(f"\n{'='*78}")
print(f"  {'Model':<30} {'IoU':>7} {'Dice':>7} {'Prec':>7} "
      f"{'Recall':>7} {'Acc':>7}")
print(f"{'='*78}")
for name, m in [("Standard U-Net",         std_m),
                ("Attention U-Net",        att_m),
                ("DKAG U-Net (Proposed)",  dkag_m)]:
    print(f"  {name:<30} "
          f"{m['iou']*100:>6.2f}%  {m['dice']*100:>6.2f}%  "
          f"{m['precision']*100:>6.2f}%  {m['recall']*100:>6.2f}%  "
          f"{m['accuracy']*100:>6.2f}%")
print(f"{'='*78}")


# %% 12. LEARNING CURVES ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ["#1976D2", "#FF9800", "#4CAF50"]
names  = ["Standard U-Net", "Attention U-Net", "DKAG U-Net (Proposed)"]
hists  = [std_hist, att_hist, dkag_hist]

panels = [
    ("val_loss",  "train_loss", "Loss",       "Loss"),
    ("val_iou",   "train_iou",  "IoU",        "IoU"),
    ("val_dice",  "train_dice", "Dice score", "Dice Score"),
]
for ax, (val_k, train_k, ylabel, title) in zip(axes, panels):
    for i, (nm, h) in enumerate(zip(names, hists)):
        ax.plot(h[val_k], label=f"{nm} (val)",
                color=colors[i], linewidth=2)
        ax.plot(h[train_k], label=f"{nm} (train)",
                color=colors[i], linewidth=1.2, ls="--", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7.5); ax.grid(True, ls="--", alpha=0.4)

plt.suptitle("DKAG U-Net — Model Comparison",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(SAVE_DIR, "learning_curves.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"[Saved] {out_path}")
