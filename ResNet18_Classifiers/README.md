# ResNet-18 Klassifikatoren (Fire & X-Ray)

Bereinigte, lauffaehige Trainings-Skripte fuer die beiden "grossen" Datensaetze,
deren Klassifikatoren spaeter von **Mirror-CFE** erklaert werden.

| Skript | Datensatz | Klassen |
|---|---|---|
| [`train_fire.py`](train_fire.py) | D-Fire (YOLO) | `fire` / `no_fire` |
| [`train_xray.py`](train_xray.py) | NIH Chest X-Ray | `No Finding` / `Infiltration` |

Beide Skripte sind eigenstaendig (kein gemeinsames Modul), sodass `train_xray.py`
unveraendert auf Kaggle laeuft.

```bash
# Fire (lokal, Apple-MPS/CUDA/CPU wird automatisch erkannt)
python train_fire.py --dfire-root "../Fire _Detection/fire_detection/D-Fire"

# X-Ray (Kaggle o. lokal)
python train_xray.py --data-root ".../nih-chest-xrays/data" --out-dir .
```

## Macht eine Standard-`torchvision.models.resnet18` Sinn? — Nein (hier).

Kurz: **Nein, fuer Fire/X-Ray nicht.** Die selbst implementierte Architektur wird
beibehalten. Begruendung:

Die Mirror-CFE-Pipeline (`Counterfactuals/Mirror/mirrorcfe-{fire,xray}.ipynb`
und `mirror-decoder-{fire,xray}.ipynb`) ist **fest an die Namensstruktur** dieses
Netzes gekoppelt:

* Sie hookt `model.encoder.blocks[0..3]` fuer die Multi-Scale-Skip-Features
  **f1..f4** `(B,64,56,56) → (B,128,28,28) → (B,256,14,14) → (B,512,7,7)`, die
  der `MirrorDecoder` (CSP-Modul) braucht.
* Sie liest die CAM/CSP-Gewichte aus `model.decoder.decoder.weight` `(2,512)`
  und `.bias`, und rechnet Logits mit `model.decoder.decoder(flat)` nach.

Eine `torchvision.models.resnet18` exponiert stattdessen `layer1..layer4`,
`avgpool` und `fc` — andere Namen und andere Verschachtelung. Ein Umstieg wuerde
die komplette Mirror-Pipeline brechen. Genau *diese* modulare Encoder/Decoder-
Struktur ist es, die die Feature-Maps und GAP-Gewichte sauber adressierbar
macht — also das, was CAM/CSP verlangt.

> Hinweis zur Konsistenz im Repo: Die MNIST/FashionMNIST-**Toy-Skripte**
> (`mnist_7v9_resnet18.py`, `fashionmnist_..._resnet18.py`) nutzen die
> Standard-`torchvision`-Variante — das ist dort ok, weil sie nur 1-Kanal-
> Spielzeugmodelle sind. Fuer die beiden Datensaetze, die real durch Mirror-CFE
> laufen, gibt die bestehende Pipeline die Custom-Architektur vor.

## CAM-Zugriff (Umsetzung des Uebergabeprotokolls GAP/CAM)

Das Protokoll beschrieb korrekt: Die GAP-Schicht ist vorhanden, aber die fuer CAM
noetige Feature-Map `f^l_k [B,512,7,7]` existierte nur als lokale Variable in
`forward()` und war von aussen nicht greifbar.

Statt eines nachgelagerten Forward-Hooks (Option A) setzen diese Skripte
**Option B sauber** um — die Modellklasse gibt die Feature-Map auf Wunsch mit
zurueck, ohne bestehenden Code zu brechen:

```python
logits            = model(x)                      # wie bisher: nur Logits
logits, f_l_k     = model(x, return_features=True) # f_l_k: [B, 512, 7, 7]
logits, cams      = compute_cam(model, x)          # cams:  [B, K, 7, 7]
```

`compute_cam` implementiert `U[b,k] = Σ_c W[k,c] · f^l_k[b,c]` (unnormalisierte
CAM je Klasse). Da `return_features` nur ein zusaetzliches Rueckgabeobjekt ist,
bleiben Trainings-/Eval-Loops **und** die vorhandenen Mirror-Hooks unveraendert
funktionsfaehig. Die erzeugten `.pth` (identischer `state_dict`) sind drop-in
kompatibel mit den vorhandenen Mirror-CFE-Notebooks.

## Was gegenueber den Original-Notebooks korrigiert wurde

**X-Ray (`resnet18-binary-xray-cnn.ipynb`):**
* Markdown behauptete "BCELoss mit `pos_weight`" — tatsaechlich lief
  `CrossEntropyLoss` mit Klassengewichten. Jetzt konsistent (CrossEntropyLoss).
* `BATCH_SIZE = 256` deklariert, aber DataLoader hart auf `64` — jetzt einheitlich
  ueber `--batch-size`.
* Tote Debug-Zelle `torch.randn(2, 3).cuda()` entfernt.
* Hart codierte `/kaggle/...`-Pfade → `--data-root` / `--out-dir`.

**Fire (`fire_detection.ipynb`):**
* Irrefuehrender toter Code entfernt: `CLASS_MAP={0:'fire',1:'smoke'}`,
  `MIN_CROP_SIZE`, `crop_from_yolo()` wurden nie benutzt. Das Preprocessing
  croppt keine Boxen und trennt kein "smoke", sondern sortiert **ganze Bilder**
  (≥1 Box → `fire`, kein Label → `no_fire`). Der Code sagt das jetzt auch.
* Fragiler Val-Transform-Hack (`val_set.dataset = ImageFolder(...)` auf einem
  `random_split`-Subset) → sauberer Split auf Sample-Ebene mit getrennten
  Transform-Wrappern (`TransformSubset`).
