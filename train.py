"""
SkinAid — Improved Training Script
===================================
Key improvements over the original:
  1. Class-weighted loss     — fixes HAM10000 imbalance (67% nv vs 2% vasc)
  2. Train / val split       — monitors generalisation, stops overfitting
  3. LR scheduler            — ReduceLROnPlateau halves LR when val plateaus
  4. Early stopping          — stops when val accuracy stops improving
  5. Best model checkpoint   — saves the best val-accuracy model, not last
  6. More augmentation       — colour jitter, random erasing, vertical flip
  7. Frozen backbone warmup  — first 3 epochs train only the head, then unfreeze
  8. Mixed precision (AMP)   — ~2x faster on GPU with no accuracy loss
  9. Per-epoch metrics       — loss, accuracy, per-class breakdown on val set
 10. Configurable via CONFIG — change one block at the top, nothing else

Usage:
    python train.py

Folder structure expected:
    dataset/
        images/
            akiec/  *.jpg
            bcc/    *.jpg
            bkl/    *.jpg
            df/     *.jpg
            mel/    *.jpg
            nv/     *.jpg
            vasc/   *.jpg
    models/           (created automatically)
"""

import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from collections import Counter

# ─────────────────────────────────────────────────────────────────────
# CONFIG — change anything here, leave the rest of the file alone
# ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_path":        "dataset/images",
    "model_save_path":  "models/skin_model.pth",
    "log_save_path":    "models/training_log.txt",

    "epochs":           50,          # max epochs (early stopping may end sooner)
    "batch_size":       32,          # reduce to 16 if you get OOM errors
    "lr":               1e-4,        # initial learning rate
    "warmup_epochs":    3,           # epochs to train only the final layer
    "val_split":        0.2,         # 20% of data held out for validation
    "early_stop_patience": 8,        # stop if val acc doesn't improve for N epochs
    "num_workers":      4,           # set to 0 on Windows if you get errors
    "img_size":         224,
    "seed":             42,
}
# ─────────────────────────────────────────────────────────────────────

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()   # mixed precision only on GPU


# ── Transforms ───────────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3,
        saturation=0.2, hue=0.1
    ),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Dataset split helper ──────────────────────────────────────────────

def split_dataset(dataset, val_fraction, seed):
    """Split an ImageFolder into train/val subsets."""
    from torch.utils.data import Subset
    n       = len(dataset)
    indices = list(range(n))
    rng     = np.random.RandomState(seed)
    rng.shuffle(indices)
    split   = int(n * val_fraction)
    return Subset(dataset, indices[split:]), Subset(dataset, indices[:split])


# ── Weighted sampler (fixes class imbalance) ─────────────────────────

def make_weighted_sampler(subset):
    """Return a WeightedRandomSampler that balances class frequency."""
    targets = [subset.dataset.targets[i] for i in subset.indices]
    counts  = Counter(targets)
    n_cls   = len(counts)
    weights = [1.0 / counts[t] for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Training loop ─────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, phase, classes):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    class_correct = Counter()
    class_total   = Counter()

    pbar = tqdm(loader, desc=f"  {phase:5s}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=USE_AMP):
                outputs = model(images)
                loss    = criterion(outputs, labels)

        if is_train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        for p, l in zip(preds.cpu(), labels.cpu()):
            class_total[l.item()]   += 1
            class_correct[l.item()] += int(p == l)

        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         acc=f"{100*correct/total:.1f}%")

    epoch_loss = total_loss / total
    epoch_acc  = 100.0 * correct / total

    # Per-class accuracy
    per_class = {
        classes[k]: f"{100*class_correct[k]/max(class_total[k],1):.1f}%"
        for k in sorted(class_total)
    }
    return epoch_loss, epoch_acc, per_class


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  SkinAid Training — Device: {DEVICE}")
    print(f"  Mixed precision (AMP): {USE_AMP}")
    print(f"  Max epochs: {CONFIG['epochs']}  |  Patience: {CONFIG['early_stop_patience']}")
    print(f"{'='*60}\n")

    # ── Load full dataset (train transform for now; val will override) ──
    full_dataset = datasets.ImageFolder(CONFIG["data_path"], transform=train_transform)
    classes      = full_dataset.classes
    print(f"Classes ({len(classes)}): {classes}")

    # Count per class
    counts = Counter(full_dataset.targets)
    print("Samples per class:")
    for i, cls in enumerate(classes):
        print(f"  {cls:8s}: {counts[i]:5d}")
    print(f"  Total   : {len(full_dataset):5d}\n")

    # ── Split ──
    train_subset, val_subset = split_dataset(
        full_dataset, CONFIG["val_split"], CONFIG["seed"]
    )

    # Apply val transform to val subset (without augmentation)
    val_subset.dataset = copy.deepcopy(full_dataset)
    val_subset.dataset.transform = val_transform

    # Weighted sampler for training
    sampler = make_weighted_sampler(train_subset)

    train_loader = DataLoader(
        train_subset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train: {len(train_subset)} samples  |  Val: {len(val_subset)} samples\n")

    # ── Model ──
    model = models.resnet50(weights="IMAGENET1K_V1")

    # Freeze all layers initially (warmup phase)
    for param in model.parameters():
        param.requires_grad = False

    # Replace head
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model    = model.to(DEVICE)

    # ── Class-weighted loss ──
    # Inverse frequency weighting
    total_samples = sum(counts.values())
    class_weights = torch.tensor(
        [total_samples / (len(classes) * counts[i]) for i in range(len(classes))],
        dtype=torch.float
    ).to(DEVICE)
    print("Class weights (for loss):")
    for i, cls in enumerate(classes):
        print(f"  {cls:8s}: {class_weights[i]:.3f}")
    print()

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer — only head params at first ──
    optimizer = optim.Adam(model.fc.parameters(), lr=CONFIG["lr"])

    # LR scheduler — halve LR if val accuracy doesn't improve for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=3, verbose=True
    )

    scaler = GradScaler(enabled=USE_AMP)

    # ── Training ──
    best_val_acc    = 0.0
    best_state      = None
    patience_counter= 0
    log_lines       = []

    os.makedirs("models", exist_ok=True)

    for epoch in range(1, CONFIG["epochs"] + 1):

        # Unfreeze backbone after warmup
        if epoch == CONFIG["warmup_epochs"] + 1:
            print(f"\n  [Epoch {epoch}] Unfreezing full backbone...\n")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5,
                patience=3, verbose=True
            )

        t0 = time.time()
        print(f"Epoch {epoch}/{CONFIG['epochs']}  LR={optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, "train", classes
        )
        val_loss, val_acc, val_per_class = run_epoch(
            model, val_loader, criterion, optimizer, scaler, "val", classes
        )

        scheduler.step(val_acc)

        elapsed = time.time() - t0
        line = (f"Epoch {epoch:3d} | "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
                f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}% | "
                f"{elapsed:.0f}s")
        print(f"  {line}")
        print(f"  Val per class: {val_per_class}\n")
        log_lines.append(line)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_state    = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_state, CONFIG["model_save_path"])
            print(f"  ✓ New best val accuracy: {best_val_acc:.2f}% — model saved\n")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CONFIG['early_stop_patience']})\n")

        if patience_counter >= CONFIG["early_stop_patience"]:
            print(f"  Early stopping triggered at epoch {epoch}\n")
            break

    # ── Final summary ──
    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"  Best val accuracy : {best_val_acc:.2f}%")
    print(f"  Model saved to    : {CONFIG['model_save_path']}")
    print(f"{'='*60}\n")

    # Save log
    with open(CONFIG["log_save_path"], "w") as f:
        f.write("\n".join(log_lines))
    print(f"  Training log saved to: {CONFIG['log_save_path']}")


if __name__ == "__main__":
    main()