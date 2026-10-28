from pathlib import Path
import os
from collections import defaultdict, Counter
from PIL import Image, ImageEnhance, ImageOps
import hashlib, random, time, copy, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("RETINAWATCH_DATA_DIR", str(ROOT / "data")))
OUT = ROOT / "backend" / "models"; OUT.mkdir(parents=True, exist_ok=True)
REPORTS = ROOT / "reports"; REPORTS.mkdir(parents=True, exist_ok=True)
CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

# Group exact duplicate images before the train/validation/test split. Remove any exact image
# hash that appears under conflicting labels, then retain one copy for consistent labels.
by_hash = defaultdict(list)
for label in CLASSES:
    class_dir = DATA / label
    if not class_dir.exists():
        raise FileNotFoundError(f"Missing class folder: {class_dir}")
    for path in sorted(class_dir.glob("*.png")):
        by_hash[hashlib.md5(path.read_bytes()).hexdigest()].append((path, label))
records = []
conflicts = same_label_extras = 0
for digest, group in by_hash.items():
    if len({label for _, label in group}) > 1:
        conflicts += 1
        continue
    if len(group) > 1:
        same_label_extras += len(group) - 1
    records.append(group[0])
if len(records) < 100:
    raise ValueError("Too few images found. Extract the five labeled folders directly into data/.")
print(f"Clean unique images: {len(records)}; conflicting groups removed: {conflicts}; duplicate copies removed: {same_label_extras}", flush=True)

paths = [str(path) for path, label in records]
labels = [CLASSES.index(label) for path, label in records]
train_paths, temp_paths, train_labels, temp_labels = train_test_split(paths, labels, test_size=.30, random_state=SEED, stratify=labels)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=.50, random_state=SEED, stratify=temp_labels)
SIZE = 192

class FundusDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths, self.labels, self.augment = paths, labels, augment
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        if self.augment:
            if random.random() < .5: image = ImageOps.mirror(image)
            if random.random() < .15: image = ImageOps.flip(image)
            if random.random() < .65: image = image.rotate(random.uniform(-12, 12), Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
            if random.random() < .5: image = ImageEnhance.Brightness(image).enhance(random.uniform(.9, 1.1))
            if random.random() < .5: image = ImageEnhance.Contrast(image).enhance(random.uniform(.9, 1.1))
            if random.random() < .25: image = ImageEnhance.Color(image).enhance(random.uniform(.9, 1.1))
        array = np.asarray(image, dtype=np.float32) / 255.
        array = (array - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return torch.from_numpy(array.transpose(2, 0, 1).copy()), int(self.labels[i])

class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, kernel=3, stride=1, groups=1):
        super().__init__(nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=False), nn.BatchNorm2d(cout), nn.Hardswish(inplace=True))

class DepthwiseSeparable(nn.Module):
    def __init__(self, cin, cout, stride=1, expansion=2):
        super().__init__()
        hidden = max(cin, int(cin * expansion))
        layers = []
        if hidden != cin: layers.append(ConvBNAct(cin, hidden, kernel=1))
        layers += [ConvBNAct(hidden, hidden, 3, stride, hidden), ConvBNAct(hidden, cout, 1)]
        self.block = nn.Sequential(*layers)
        self.skip = stride == 1 and cin == cout
    def forward(self, x):
        y = self.block(x)
        return y + x if self.skip else y

class RetinaWatchNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 24, 3, 2),
            DepthwiseSeparable(24, 32, 2, 2),
            DepthwiseSeparable(32, 48, 2, 2),
            DepthwiseSeparable(48, 64, 2, 2),
            DepthwiseSeparable(64, 96, 2, 2),
            DepthwiseSeparable(96, 128, 1, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(.30)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        return self.classifier(self.dropout(self.pool(self.features(x)).flatten(1)))

train_loader = DataLoader(FundusDataset(train_paths, train_labels, True), batch_size=24, shuffle=True, num_workers=0)
val_loader = DataLoader(FundusDataset(val_paths, val_labels), batch_size=48, shuffle=False, num_workers=0)
test_loader = DataLoader(FundusDataset(test_paths, test_labels), batch_size=48, shuffle=False, num_workers=0)
model = RetinaWatchNet()
counts = Counter(train_labels)
weights = np.array([len(train_labels) / (len(CLASSES) * counts[i]) for i in range(5)], np.float32)
weights = weights / weights.mean()
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights), label_smoothing=.025)
optimizer = torch.optim.AdamW(model.parameters(), lr=.00065, weight_decay=.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=14)

def evaluate(loader):
    model.eval(); truth=[]; pred=[]; scores=[]
    with torch.no_grad():
        for x, y in loader:
            probabilities = torch.softmax(model(x), dim=1)
            truth.extend(y.numpy()); pred.extend(probabilities.argmax(1).numpy()); scores.extend(probabilities.numpy())
    return np.array(truth), np.array(pred), np.array(scores)

best_score, best_state, best_epoch, waiting = -1, None, 0, 0
start = time.time()
for epoch in range(1, 15):
    model.train(); total_loss=0.; rows=0
    for x, y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        total_loss += loss.item() * len(y); rows += len(y)
    scheduler.step()
    truth, pred, _ = evaluate(val_loader)
    val_acc = accuracy_score(truth, pred)
    val_bal = balanced_accuracy_score(truth, pred)
    val_f1 = f1_score(truth, pred, average="macro", zero_division=0)
    score = val_bal + .15 * val_f1
    print(f"Epoch {epoch:02d}: loss={total_loss/rows:.4f}, val_accuracy={val_acc:.4f}, val_balanced_accuracy={val_bal:.4f}, val_macro_f1={val_f1:.4f}, elapsed={time.time()-start:.1f}s", flush=True)
    if score > best_score + 1e-4:
        best_score, best_state, best_epoch, waiting = score, copy.deepcopy(model.state_dict()), epoch, 0
    else:
        waiting += 1
    if waiting >= 4 and epoch >= 8:
        print("Early stopping based on validation metrics.", flush=True)
        break
if best_state is not None:
    model.load_state_dict(best_state)
truth, pred, _ = evaluate(test_loader)
metrics = {
    "total_images": sum(1 for label in CLASSES for _ in (DATA / label).glob("*.png")),
    "cleaned_unique_images": len(records), "excluded_conflicting_duplicate_groups": conflicts,
    "removed_same_label_duplicate_copies": same_label_extras, "train_images": len(train_paths),
    "validation_images": len(val_paths), "test_images": len(test_paths),
    "classes": {name: int(sum(label == i for label in labels)) for i, name in enumerate(CLASSES)},
    "test_accuracy": float(accuracy_score(truth, pred)),
    "test_balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
    "test_macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
    "test_quadratic_weighted_kappa": float(cohen_kappa_score(truth, pred, weights="quadratic")),
    "classification_report": classification_report(truth, pred, labels=list(range(5)), target_names=CLASSES, output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(truth, pred, labels=list(range(5))).tolist(),
    "class_names": CLASSES, "input_size": SIZE, "mean": [.485, .456, .406], "std": [.229, .224, .225],
    "architecture": "RetinaWatchCNN-ExpandedDS-v3 trained from scratch; no pretrained weights.",
    "best_epoch": best_epoch, "epochs_completed": epoch, "seed": SEED,
}
torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "input_size": SIZE, "mean": metrics["mean"], "std": metrics["std"], "architecture": metrics["architecture"], "best_epoch": best_epoch}, OUT / "retinawatch_model.pt")
(REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("FINAL TEST:", metrics["test_accuracy"], metrics["test_balanced_accuracy"], metrics["test_macro_f1"], metrics["test_quadratic_weighted_kappa"], flush=True)
print(classification_report(truth, pred, target_names=CLASSES, zero_division=0), flush=True)
print("Confusion matrix:", metrics["confusion_matrix"], flush=True)
