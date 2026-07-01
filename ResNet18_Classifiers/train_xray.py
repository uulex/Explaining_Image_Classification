"""
ResNet-18 zur binaeren Klassifikation von Thorax-Roentgenbildern
(NIH Chest X-Ray): `No Finding` vs. `Infiltration`.

Bereinigte, lauffaehige Fassung des urspruenglichen Notebooks
`resnet18-binary-xray-cnn.ipynb`. Korrigiert wurde u. a.:

  * Die Markdown-Beschreibung behauptete "BCELoss mit pos_weight"; tatsaechlich
    wird `CrossEntropyLoss` mit Klassengewichten benutzt. Der Code hier ist
    ehrlich dazu -> CrossEntropyLoss (2 Logits, Integer-Labels).
  * BATCH_SIZE war auf 256 gesetzt, die DataLoader benutzten aber hart
    codierte 64 -> jetzt einheitlich ueber `--batch-size`.
  * Debug-Zelle `torch.randn(2, 3).cuda()` entfernt.
  * Hart codierte Kaggle-/`/kaggle/working`-Pfade -> ueber CLI konfigurierbar.

Wichtig fuer Mirror-CFE (siehe Uebergabeprotokoll GAP/CAM):
Die Architektur bleibt bewusst die selbst implementierte ResNet-18 mit der
Struktur `encoder.blocks[0..3]` + `decoder.decoder` (GAP + FC). Die Mirror-CFE-
Pipeline (mirrorcfe-xray / mirror-decoder-xray) haengt genau an diesen Namen:
sie hookt `encoder.blocks[0..3]` fuer die Skip-Features f1..f4 und liest
`decoder.decoder.weight/bias` fuer die CAM/CSP-Maske. Eine Standard-
`torchvision.models.resnet18` (mit `layer1..4`/`fc`) wuerde diese Pipeline
brechen. Statt eines Forward-Hooks (Option A des Protokolls) exponiert
`ResNet.forward(x, return_features=True)` die Feature-Map f^l_k [B,512,7,7]
direkt (Option B) -- rueckwaertskompatibel, da `model(x)` weiterhin nur die
Logits liefert.

Aufruf (lokal):
    .venv/bin/python train_xray.py --data-root "../NIH Chest X-Ray/data"
Aufruf (Kaggle-Standardpfad):
    python train_xray.py --data-root /kaggle/input/.../nih-chest-xrays/data \
                         --out-dir /kaggle/working
"""

import argparse
from collections import OrderedDict
from functools import partial
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# --------------------------------------------------------------------------- #
# Reproduzierbarkeit / Konstanten
# --------------------------------------------------------------------------- #
torch.manual_seed(2024)
np.random.seed(2024)

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = {0: "No Finding", 1: "Infiltration"}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Modell -- selbst implementiertes ResNet-18 (Encoder + GAP/FC-Decoder)
# Struktur bewusst identisch zum Original, damit trainierte .pth ohne Umbau
# von den Mirror-CFE-Notebooks geladen werden koennen.
# --------------------------------------------------------------------------- #
class Conv2dAuto(nn.Conv2d):
    """Conv2d mit automatischem 'same'-Padding anhand der Kernelgroesse."""

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
    """Gate (7x7 conv + pool) gefolgt von 4 Residual-Layern (blocks[0..3])."""

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
    """GAP (AdaptiveAvgPool2d) gefolgt von Fully-Connected-Klassifikator."""

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
    """Komplettes ResNet: Encoder + GAP/FC-Decoder."""

    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResNetDecoder(
            self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes
        )

    def forward(self, x, return_features=False):
        """
        return_features=False -> nur Logits [B, n_classes] (Standard, damit
        Trainings-/Eval-Loops und die vorhandenen Mirror-Hooks unveraendert
        funktionieren).
        return_features=True  -> (Logits, f^l_k) mit f^l_k = Encoder-Ausgabe
        vor dem GAP, Shape [B, 512, 7, 7]. Das ist die fuer CAM benoetigte
        Feature-Map (Option B aus dem Uebergabeprotokoll, ohne Forward-Hook).
        """
        feat = self.encoder(x)
        logits = self.decoder(feat)
        if return_features:
            return logits, feat
        return logits


def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, depths=[2, 2, 2, 2])


@torch.no_grad()
def compute_cam(model, x):
    """
    Class Activation Maps direkt aus der jetzt zugaenglichen Feature-Map.
    U[b, k] = sum_c W[k, c] * f^l_k[b, c]   (unnormalisierte CAM je Klasse k)
    Rueckgabe: logits [B, K], cams [B, K, 7, 7].
    """
    model.eval()
    logits, feat = model(x, return_features=True)      # feat: [B, 512, 7, 7]
    W = model.decoder.decoder.weight                   # [K, 512]
    cams = torch.einsum("kc,bchw->bkhw", W, feat)
    return logits, cams


# --------------------------------------------------------------------------- #
# Daten
# --------------------------------------------------------------------------- #
def assign_binary_label(finding: str):
    """0 = No Finding, 1 = Infiltration, None = verwerfen (Multi-Label etc.)."""
    if finding == "No Finding":
        return 0
    if "Infiltration" in finding:
        return 1
    return None


def build_dataframes(data_root: Path):
    csv_path = data_root / "Data_Entry_2017.csv"
    image_glob = str(data_root / "images*" / "*" / "*.png")

    df = pd.read_csv(csv_path)
    paths = {Path(p).name: p for p in glob(image_glob)}
    print(f"Scans gefunden: {len(paths)} | CSV-Zeilen: {len(df)}")

    df["path"] = df["Image Index"].map(paths.get)
    # Ehrliche Zuordnung auf 'Finding Labels' (nicht auf eine 'Infiltration'-Spalte).
    df["binary_label"] = df["Finding Labels"].map(assign_binary_label)

    binary_df = df.dropna(subset=["binary_label", "path"]).copy()
    binary_df["binary_label"] = binary_df["binary_label"].astype(int)

    # Klassen ausbalancieren (Infiltration ist die Minderheitsklasse).
    pos = binary_df[binary_df["binary_label"] == 1]
    neg = binary_df[binary_df["binary_label"] == 0].sample(
        min(len(pos), (binary_df["binary_label"] == 0).sum()), random_state=42
    )
    balanced = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Balancierter Datensatz: {len(balanced)} "
          f"(pos={len(pos)}, neg={len(neg)})")

    # Patienten-basierter Split (kein Leakage ueber Patient ID).
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(balanced, groups=balanced["Patient ID"]))
    train_df = balanced.iloc[train_idx].reset_index(drop=True)
    temp_df = balanced.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["Patient ID"]))
    valid_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    assert not (set(train_df["Patient ID"]) & set(valid_df["Patient ID"]))
    assert not (set(train_df["Patient ID"]) & set(test_df["Patient ID"]))
    print(f"Train: {len(train_df)} | Val: {len(valid_df)} | Test: {len(test_df)}")
    return train_df, valid_df, test_df


class XRayDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(int(row["binary_label"]), dtype=torch.long)
        try:
            image = Image.open(row["path"]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), label


def make_loaders(train_df, valid_df, test_df, batch_size, num_workers):
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(XRayDataset(train_df, train_tf), shuffle=True, **common)
    valid_loader = DataLoader(XRayDataset(valid_df, eval_tf), shuffle=False, **common)
    test_loader = DataLoader(XRayDataset(test_df, eval_tf), shuffle=False, **common)
    return train_loader, valid_loader, test_loader


# --------------------------------------------------------------------------- #
# Training / Evaluation
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
        all_probs.append(torch.softmax(outputs, dim=1)[:, 1].cpu())
        all_labels.append(labels.cpu())
    return (total_loss / total, correct / total,
            torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=str,
        default="../input/datasets/organizations/nih-chest-xrays/data",
        help="Ordner mit Data_Entry_2017.csv und images*/",
    )
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"Geraet: {device}")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "resnet18_xray_best.pth"
    final_ckpt = out_dir / "resnet18_xray_final.pth"

    train_df, valid_df, test_df = build_dataframes(data_root)
    train_loader, valid_loader, test_loader = make_loaders(
        train_df, valid_df, test_df, args.batch_size, args.num_workers
    )

    model = resnet18(in_channels=3, n_classes=2).to(device)
    with torch.no_grad():
        logits, feat = model(torch.zeros(2, 3, IMG_SIZE, IMG_SIZE, device=device),
                             return_features=True)
    print(f"Sanity-Check -> Logits {tuple(logits.shape)}, "
          f"Feature-Map f^l_k {tuple(feat.shape)}")  # [2,2] und [2,512,7,7]

    # CrossEntropyLoss (2 Logits) mit Klassengewichten gegen Restimbalanz.
    n_neg = int((train_df["binary_label"] == 0).sum())
    n_pos = int((train_df["binary_label"] == 1).sum())
    weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     factor=0.3, patience=2)

    best_val = float("inf")
    patience_ctr = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, valid_loader, criterion, device)
        scheduler.step(vl_loss)

        marker = ""
        if vl_loss < best_val:
            best_val, patience_ctr = vl_loss, 0
            torch.save({"model_state_dict": model.state_dict()}, best_ckpt)
            marker = " <- best"
        else:
            patience_ctr += 1
        print(f"Epoche {epoch:02d}/{args.epochs} | train_loss={tr_loss:.4f} "
              f"train_acc={tr_acc:.4f} | val_loss={vl_loss:.4f} "
              f"val_acc={vl_acc:.4f}{marker}")
        if patience_ctr >= args.patience:
            print(f"Early Stopping nach {args.patience} Epochen ohne Verbesserung.")
            break

    # Bestes Modell laden und auf dem Test-Set auswerten.
    model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state_dict"])
    test_loss, test_acc, probs, labels = evaluate(model, test_loader, criterion, device)
    preds = (probs >= 0.5).astype(int)
    roc_auc = auc(*roc_curve(labels, probs)[:2])
    print(f"\nTest-Loss {test_loss:.4f} | Test-Acc {test_acc:.4f} | AUC {roc_auc:.4f}\n")
    print(classification_report(labels, preds, target_names=list(CLASS_NAMES.values())))

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "img_size": IMG_SIZE,
        "n_classes": 2,
        "test_accuracy": round(test_acc, 4),
        "test_auc": round(float(roc_auc), 4),
    }, final_ckpt)
    print(f"Modell gespeichert: {final_ckpt}")


if __name__ == "__main__":
    main()
