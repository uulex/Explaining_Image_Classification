"""
ResNet-18 zur Unterscheidung der Fashion-MNIST-Klassen Trouser und Dress.

Laeuft lokal (CPU / Apple-MPS / CUDA wird automatisch erkannt).
Trainiert ein an 1-Kanal-Fashion-MNIST (28x28) angepasstes ResNet-18 und speichert
die Gewichte unter `fashionmnist_trouser_dress_resnet18.pth`.

Aufbau identisch zu `mnist_7v9_resnet18.py`; geaendert sind nur Datensatz
(FashionMNIST), Klassen (Trouser=1, Dress=3) und Normalisierung.

Aufruf:
    .venv/bin/python fashionmnist_trouser_dress_resnet18.py
    .venv/bin/python fashionmnist_trouser_dress_resnet18.py --epochs 5 --batch-size 128
"""

import argparse
import ssl
from pathlib import Path

# macOS-Python bringt oft keine CA-Zertifikate mit -> Download schlaegt
# mit CERTIFICATE_VERIFY_FAILED fehl. certifi-Bundle nutzen, sonst Pruefung lockern.
try:
    import certifi

    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
except ImportError:
    ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

# Fashion-MNIST: Trouser=1, Dress=3 -> auf die Klassen 0 und 1 abgebildet.
CLASSES = (1, 3)
CLASS_NAMES = {0: "Trouser", 1: "Dress"}
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
WEIGHTS_PATH = HERE / "fashionmnist_trouser_dress_resnet18.pth"
# Fashion-MNIST Standard-Statistik
MEAN, STD = 0.2860, 0.3530


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model() -> nn.Module:
    """ResNet-18, angepasst an 1-Kanal-Bilder mit 28x28 und 2 Klassen."""
    model = resnet18(weights=None)
    # 1 Eingangskanal statt 3; kleiner 3x3-Stem ohne aggressives Downsampling,
    # damit die 28x28-Bilder nicht zu frueh schrumpfen.
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def make_loader(train: bool, batch_size: int) -> DataLoader:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,)),
        ]
    )
    ds = datasets.FashionMNIST(root=str(DATA_DIR), train=train, download=True, transform=tfm)
    # Nur Trouser und Dress behalten und Labels auf 0/1 umschreiben.
    mask = (ds.targets == CLASSES[0]) | (ds.targets == CLASSES[1])
    idx = torch.where(mask)[0]
    ds.targets = (ds.targets == CLASSES[1]).long()  # Trouser -> 0, Dress -> 1
    subset = Subset(ds, idx.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=train, num_workers=2)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = get_device()
    print(f"Geraet: {device}")

    train_loader = make_loader(train=True, batch_size=args.batch_size)
    test_loader = make_loader(train=False, batch_size=args.batch_size)
    print(f"Train-Bilder: {len(train_loader.dataset)} | Test-Bilder: {len(test_loader.dataset)}")

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
        train_loss = running / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        print(f"Epoche {epoch}/{args.epochs} | Loss {train_loss:.4f} | Test-Acc {acc:.4f}")

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Gewichte gespeichert: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
