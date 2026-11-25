import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

CONFIG_PATH = Path("/content/dataset_config.json")
WEIGHTS_OUTPUT = Path("/content/best_segformer.pth")

ENCODER_NAME = "mit_b2"
IMG_SIZE = 640
EPOCHS = 60
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-2
VAL_FALLBACK_RATIO = 0.2
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_pairs(split_dir):
    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    if not img_dir.exists() or not mask_dir.exists():
        return []

    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        mask_path = mask_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def load_splits(data_dir):
    train_pairs = collect_pairs(data_dir / "train")
    val_pairs = collect_pairs(data_dir / "valid")

    if not train_pairs:
        raise RuntimeError(f"No image/mask pairs found in {data_dir / 'train'}")

    if not val_pairs:
        rng = random.Random(SEED)
        shuffled = list(train_pairs)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * VAL_FALLBACK_RATIO)))
        val_pairs = shuffled[:val_count]
        train_pairs = shuffled[val_count:]
        print(f"No valid split found. Auto-split: {len(train_pairs)} train / {len(val_pairs)} valid")

    return train_pairs, val_pairs


class AerialSegDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"].long()


def build_transforms(img_size):
    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.3),
	A.OneOf([
            A.ToGray(p=1.0),
            A.ToSepia(p=1.0),
        ], p=0.2),
        A.GaussianBlur(blur_limit=(3, 3), p=0.15),
        A.GaussNoise(p=0.15),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tf, val_tf


def evaluate(model, loader, num_classes, device, use_amp):
    model.eval()
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast(device, enabled=use_amp):
                preds = model(images).argmax(dim=1)

            for cls_id in range(num_classes):
                pred_c = preds == cls_id
                true_c = masks == cls_id
                inter[cls_id] += (pred_c & true_c).sum().item()
                union[cls_id] += (pred_c | true_c).sum().item()

    ious = np.where(union > 0, inter / np.maximum(union, 1.0), np.nan)
    valid_fg = ~np.isnan(ious[1:])
    fg_miou = float(np.mean(ious[1:][valid_fg])) if np.any(valid_fg) else 0.0
    return fg_miou, ious


def download_in_colab(file_path):
    try:
        from google.colab import files
        files.download(str(file_path))
    except ImportError:
        pass


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Run 05_prepare_dataset.py first. Missing: {CONFIG_PATH}")

    set_seed(SEED)
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data_dir = Path(cfg["data_dir"])
    class_names = cfg["class_names"]
    num_classes = len(class_names)

    train_pairs, val_pairs = load_splits(data_dir)
    train_tf, val_tf = build_transforms(IMG_SIZE)

    train_ds = AerialSegDataset(train_pairs, train_tf)
    val_ds = AerialSegDataset(val_pairs, val_tf)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    drop_last = len(train_ds) > BATCH_SIZE

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=use_amp,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=use_amp,
    )

    model = smp.Segformer(
        encoder_name=ENCODER_NAME,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    ).to(device)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True, smooth=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    best_miou = -1.0
    best_epoch = 0

    print(f"Training {ENCODER_NAME} on {device} | train: {len(train_ds)}, valid: {len(val_ds)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device, enabled=use_amp):
                logits = model(images)
                loss = 0.5 * ce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / max(1, len(train_loader))
        fg_miou, ious = evaluate(model, val_loader, num_classes, device, use_amp)

        improved = fg_miou > best_miou
        if improved:
            best_miou = fg_miou
            best_epoch = epoch
            checkpoint = {
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "encoder_name": ENCODER_NAME,
                "img_size": IMG_SIZE,
                "best_miou": best_miou,
                "best_epoch": best_epoch,
            }
            torch.save(checkpoint, WEIGHTS_OUTPUT)

        if epoch == 1 or epoch == EPOCHS or epoch % 5 == 0 or improved:
            cls_metrics = []
            for cls_id in range(1, num_classes):
                val = ious[cls_id]
                val_str = "n/a" if np.isnan(val) else f"{val:.3f}"
                cls_metrics.append(f"{class_names[cls_id]}: {val_str}")
            marker = " *" if improved else ""
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | loss: {avg_loss:.4f} | "
                f"mIoU(fg): {fg_miou:.4f} | {' | '.join(cls_metrics)}{marker}"
            )

    print(f"Best mIoU(fg): {best_miou:.4f} at epoch {best_epoch}")
    print("Weights saved:", WEIGHTS_OUTPUT)
    download_in_colab(WEIGHTS_OUTPUT)


if __name__ == "__main__":
    main()
