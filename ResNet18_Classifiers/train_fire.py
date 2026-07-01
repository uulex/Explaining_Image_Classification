"""
ResNet-18 zur binaeren Klassifikation von Feuer/Rauch (D-Fire Datensatz):
`fire` vs. `no_fire`.

Bereinigte, lauffaehige Fassung des urspruenglichen Notebooks
`fire_detection.ipynb`. Korrigiert wurde u. a.:

  * Toter/irrefuehrender Code entfernt: `CLASS_MAP = {0: "fire", 1: "smoke"}`,
    `MIN_CROP_SIZE` und `crop_from_yolo()` wurden nie benutzt. Das Preprocessing
    croppt keine Bounding-Boxen und trennt kein "smoke" -- es sortiert GANZE
    Bilder nach "hat mindestens eine Box" (-> fire) bzw. "kein Label" (-> no_fire).
    Der Code sagt jetzt, was er tut.
  * Der Val-Transform-Hack (`val_set.dataset = ImageFolder(...)` auf einem
    random_split-Subset) war fragil (funktionierte nur wegen deterministischer
    Dateireihenfolge). Ersetzt durch einen sauberen Split auf Sample-Ebene mit
    getrennten Transform-Wrappern.

Wichtig fuer Mirror-CFE (siehe Uebergabeprotokoll GAP/CAM):
Die Architektur bleibt bewusst die selbst implementierte ResNet-18 mit der
Struktur `encoder.blocks[0..3]` + `decoder.decoder` (GAP + FC), weil die
Mirror-CFE-Pipeline (mirrorcfe-fire / mirror-decoder-fire) genau an diesen
Namen haengt (Skip-Features f1..f4 via Hook auf `encoder.blocks[0..3]`,
CAM/CSP via `decoder.decoder.weight/bias`). `ResNet.forward(x,
return_features=True)` liefert die Feature-Map f^l_k [B,512,7,7] direkt.

Aufruf:
    .venv/bin/python train_fire.py --dfire-root ../Fire\\ _Detection/fire_detection/D-Fire
"""

import argparse
import time
from collections import Counter, OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

torch.manual_seed(42)
np.random.seed(42)

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Modell -- identisch zu train_xray.py (drop-in-kompatibel mit Mirror-CFE)
# --------------------------------------------------------------------------- #
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x) if self.should_apply_shortcut else x
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1,
                 downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion = expansion
        self.downsampling = downsampling
        self.conv = conv
        self.shortcut = (
            nn.Sequential(OrderedDict({
                "conv": nn.Conv2d(self.in_channels, self.expanded_channels,
                                  kernel_size=1, stride=self.downsampling, bias=False),
                "bn": nn.BatchNorm2d(self.expanded_channels),
            }))
            if self.should_apply_shortcut else None
        )

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({
        "conv": conv(in_channels, out_channels, *args, **kwargs),
        "bn": nn.BatchNorm2d(out_channels),
    }))


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv,
                    bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation(),
            conv_bn(self.out_channels, self.out_channels, self.conv,
                    kernel_size=3, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, out_channels,
                    downsampling=1, *args, **kwargs) for _ in range(n - 1)],
        )

    def forward(self, x):
        return self.blocks(x)


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=3, blocks_sizes=(64, 128, 256, 512),
                 depths=(2, 2, 2, 2), activation=nn.ReLU,
                 block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, blocks_sizes[0], kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0],
                        activation=activation, block=block, *args, **kwargs),
            *[ResNetLayer(in_ch * block.expansion, out_ch, n=n,
                          activation=activation, block=block, *args, **kwargs)
              for (in_ch, out_ch), n in zip(self.in_out_block_sizes, depths[1:])],
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResNetDecoder(
            self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes
        )

    def forward(self, x, return_features=False):
        """return_features=True -> (Logits, f^l_k [B,512,7,7]) fuer CAM."""
        feat = self.encoder(x)
        logits = self.decoder(feat)
        if return_features:
            return logits, feat
        return logits


def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, depths=[2, 2, 2, 2])


@torch.no_grad()
def compute_cam(model, x):
    """CAM aus der zugaenglichen Feature-Map: U[b,k] = sum_c W[k,c]*f[b,c]."""
    model.eval()
    logits, feat = model(x, return_features=True)
    W = model.decoder.decoder.weight
    cams = torch.einsum("kc,bchw->bkhw", W, feat)
    return logits, cams


# --------------------------------------------------------------------------- #
# Preprocessing: D-Fire (YOLO) -> ImageFolder fire/ vs no_fire/
# --------------------------------------------------------------------------- #
def has_boxes(label_path: Path) -> bool:
    """True, wenn das YOLO-Label mindestens eine gueltige Box enthaelt."""
    if not label_path.exists():
        return False
    with open(label_path, "r") as f:
        for line in f:
            if len(line.strip().split()) >= 5:
                return True
    return False


def process_split(images_dir: Path, labels_dir: Path, out_dir: Path, split_name: str):
    """Sortiert ganze Bilder nach fire/ (>=1 Box) bzw. no_fire/ (kein Label)."""
    out_path = out_dir / split_name
    for cls in ("fire", "no_fire"):
        (out_path / cls).mkdir(parents=True, exist_ok=True)

    stats = {"fire": 0, "no_fire": 0, "skipped": 0}
    for img_file in sorted(f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS):
        cls = "fire" if has_boxes(labels_dir / (img_file.stem + ".txt")) else "no_fire"
        try:
            Image.open(img_file).convert("RGB").save(
                out_path / cls / f"{img_file.stem}.jpg", quality=95
            )
            stats[cls] += 1
        except Exception as e:  # noqa: BLE001
            print(f"  Warnung: {img_file.name} nicht lesbar: {e}")
            stats["skipped"] += 1
    return stats


def preprocess_dfire(dfire_root: Path, out_dir: Path):
    if (out_dir / "train").exists() and (out_dir / "val").exists():
        print(f"Klassifikations-Datensatz existiert bereits: {out_dir} (ueberspringe).")
        return
    for split_name, out_name in (("train", "train"), ("test", "val")):
        img_dir = dfire_root / split_name / "images"
        lbl_dir = dfire_root / split_name / "labels"
        if not img_dir.exists():
            print(f"Ueberspringe {split_name}: {img_dir} nicht gefunden")
            continue
        print(f"Verarbeite {out_name}-Split ...")
        stats = process_split(img_dir, lbl_dir, out_dir, out_name)
        print(f"  Ergebnis: {stats}")


# --------------------------------------------------------------------------- #
# Dataset / Split (sauberer Ersatz fuer den val_set.dataset-Hack)
# --------------------------------------------------------------------------- #
class TransformSubset(Dataset):
    """Teilmenge eines ImageFolder-Datensatzes mit eigenem Transform."""

    def __init__(self, base: datasets.ImageFolder, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, target = self.base.samples[self.indices[i]]
        img = self.base.loader(path)
        return self.transform(img), target


def get_dataloaders(data_dir: str, batch_size: int, val_split: float, num_workers: int):
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    base = datasets.ImageFolder(data_dir)  # ein einziger Scan des Ordners
    class_names = base.classes
    n = len(base)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    n_val = int(n * val_split)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(
        TransformSubset(base, train_idx, train_tf), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        TransformSubset(base, val_idx, val_tf), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    print(f"Datensatz: {data_dir}")
    print(f"Klassen: {len(class_names)} {class_names}")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
    return train_loader, val_loader, class_names, len(class_names)


# --------------------------------------------------------------------------- #
# Training / Evaluation
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return running / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, fire_idx=None):
    model.eval()
    running = correct = total = 0
    probs_fire, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        running += criterion(outputs, labels).item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
        if fire_idx is not None:
            probs_fire.append(torch.softmax(outputs, 1)[:, fire_idx].cpu())
            all_labels.append(labels.cpu())
    loss, acc = running / total, correct / total
    if fire_idx is None:
        return loss, acc
    return loss, acc, torch.cat(probs_fire).numpy(), torch.cat(all_labels).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfire-root", type=str, default="./D-Fire",
                        help="Wurzel des D-Fire-YOLO-Datensatzes (train/ test/).")
    parser.add_argument("--work-dir", type=str, default=".",
                        help="Ablage fuer classification_dataset/ und best_model.pth.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Geraet: {device}")

    work_dir = Path(args.work_dir)
    out_dir = work_dir / "classification_dataset"
    model_path = work_dir / "best_model.pth"

    preprocess_dfire(Path(args.dfire_root), out_dir)

    train_loader, val_loader, class_names, n_classes = get_dataloaders(
        str(out_dir / "train"), args.batch_size, args.val_split, args.num_workers
    )

    model = resnet18(in_channels=3, n_classes=n_classes).to(device)
    with torch.no_grad():
        logits, feat = model(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device),
                             return_features=True)
    print(f"Modell: ResNet18 | Parameter: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Sanity-Check -> Logits {tuple(logits.shape)}, "
          f"Feature-Map f^l_k {tuple(feat.shape)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    print(f"\n{'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>9} "
          f"{'ValLoss':>10} {'ValAcc':>9} {'Time':>6}")
    print("-" * 56)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        marker = "*" if vl_acc > best_val_acc else ""
        print(f"{epoch:5d} {tr_loss:10.4f} {tr_acc:8.2%} "
              f"{vl_loss:10.4f} {vl_acc:8.2%} {time.time()-t0:5.1f}s {marker}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": vl_acc,
                "class_names": class_names,
            }, model_path)

    print(f"\nTraining fertig. Beste Val-Accuracy: {best_val_acc:.2%}")
    print(f"Modell gespeichert: {model_path}")

    # Abschliessende Auswertung auf dem separaten Val-Set (D-Fire test-Split).
    val_dir = out_dir / "val"
    if val_dir.exists():
        eval_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        eval_ds = datasets.ImageFolder(str(val_dir), transform=eval_tf)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
        fire_idx = eval_ds.classes.index("fire")
        loss, acc, probs, labels = validate(model, eval_loader, criterion, device, fire_idx)
        roc_auc = auc(*roc_curve((labels == fire_idx).astype(int), probs)[:2])
        print(f"\nHold-out (D-Fire test): Loss {loss:.4f} | "
              f"Acc {acc:.2%} | AUC {roc_auc:.4f}")
        print(f"Klassenverteilung Hold-out: {Counter(labels.tolist())}")


if __name__ == "__main__":
    main()
