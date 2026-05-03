"""
FloodDataset and Albumentations pipelines.

Expects two directories with matching filenames (extensions may differ):

    img_dir/    image files (.jpg / .jpeg / .png)
    mask_dir/   binary mask files (.png / .jpg / .jpeg / .bmp / .tif / .tiff)

Masks are read as grayscale and binarised at > 127 (flood = 1, else 0).
"""
import os
import random
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
IMG_EXTS  = (".jpg", ".jpeg", ".png")


# ─── Augmentation pipelines ──────────────────────────────────────────────────

def build_train_aug(img_size: int, mean, std) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.4
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
        ),
        A.GaussNoise(var_limit=(5, 30), p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.CoarseDropout(
            max_holes=4, max_height=32, max_width=32, fill_value=0, p=0.2
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def build_val_aug(img_size: int, mean, std) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# ─── Dataset ─────────────────────────────────────────────────────────────────

class FloodDataset(Dataset):
    """Image/mask dataset for binary flood segmentation."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        transform: Optional[A.Compose] = None,
    ):
        self.pairs     = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img  = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)
        if self.transform is not None:
            aug  = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        return img, mask.unsqueeze(0)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _find_mask(stem: str, mask_dir: str) -> Optional[str]:
    for ext in MASK_EXTS:
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def collect_pairs(img_dir: str, mask_dir: str):
    """Return list of (image_path, mask_path) for files that have both."""
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    images = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith(IMG_EXTS)
    )
    pairs, missing = [], 0
    for fname in images:
        stem = os.path.splitext(fname)[0]
        mp   = _find_mask(stem, mask_dir)
        if mp is not None:
            pairs.append((os.path.join(img_dir, fname), mp))
        else:
            missing += 1
    return pairs, missing


def get_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from config."""
    pairs, missing = collect_pairs(cfg.img_dir, cfg.mask_dir)
    if not pairs:
        raise RuntimeError(
            f"No matched image/mask pairs found. "
            f"Checked images in '{cfg.img_dir}' and masks in '{cfg.mask_dir}'."
        )
    print(f"Valid pairs : {len(pairs)}  |  Missing masks: {missing}")

    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    n_val       = int(len(pairs) * cfg.val_split)
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    train_aug = build_train_aug(cfg.img_size, cfg.norm_mean, cfg.norm_std)
    val_aug   = build_val_aug(cfg.img_size,  cfg.norm_mean, cfg.norm_std)

    train_dl = DataLoader(
        FloodDataset(train_pairs, train_aug),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    val_dl = DataLoader(
        FloodDataset(val_pairs, val_aug),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_dl, val_dl
