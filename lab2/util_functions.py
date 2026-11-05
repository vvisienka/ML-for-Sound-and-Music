# This file contains helper functions used for training CNNs

import copy
import re
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import model_helpers as MF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------
def image_to_filename(path):
    """'blues00012.png' -> 'blues.00012.wav' (the filename used in features_30_sec.csv)."""
    m = re.match(r"^([a-z]+)(\d{5})\.png$", Path(path).name)
    return f"{m.group(1)}.{m.group(2)}.wav"


def find_crop_box(images, white=0.97):
    """Rows/columns of the spectrogram area (the PNGs have a white margin around the plot)."""
    mean_img = images.float().mean(dim=(0, 1)) / 255
    dark = mean_img < white
    rows = torch.where(dark.float().mean(dim=1) > 0.5)[0]
    cols = torch.where(dark.float().mean(dim=0) > 0.5)[0]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def load_images(img_dir, drop_files=()):
    """
    Loads all spectrograms into memory (uint8) and crops the white margin.
    Returns a dict with the RGB images (N,3,H,W), grey-scale images (N,1,H,W), labels,
    filenames (Lab 1 format) and class names.
    """
    paths = sorted(Path(img_dir).glob("*/*.png"))
    paths = [p for p in paths if image_to_filename(p) not in set(drop_files)]
    rgb = np.stack([np.array(Image.open(p).convert("RGB")) for p in paths])
    rgb = torch.from_numpy(rgb).permute(0, 3, 1, 2)                      # (N, 3, H, W)

    top, bottom, left, right = find_crop_box(rgb)
    rgb = rgb[:, :, top:bottom, left:right].contiguous()
    gray = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).round().to(torch.uint8).unsqueeze(1)

    genres = [p.parent.name for p in paths]
    classes = sorted(set(genres))
    return {
        "rgb": rgb,
        "gray": gray,
        "labels": np.array([classes.index(g) for g in genres]),
        "filenames": np.array([image_to_filename(p) for p in paths]),
        "classes": classes,
        "crop_box": (top, bottom, left, right),
    }


def make_split(labels, seed=42, test_size=0.2, val_size=0.2):
    """
    Stratified, shuffled split: 80% train+val / 20% test, then the 80% is split again
    80/20 into train / val. Returns a dict of index arrays.
    """
    all_idx = np.arange(len(labels))
    trainval_idx, test_idx = train_test_split(all_idx, test_size=test_size, stratify=labels, random_state=seed)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=val_size, stratify=labels[trainval_idx],
                                          random_state=seed)
    return {"train": train_idx, "val": val_idx, "trainval": trainval_idx, "test": test_idx}


def train_subset(idx, labels, fraction, seed=42):
    """Stratified random subset with `fraction` of the songs in idx (for the learning curves)."""
    if fraction >= 1:
        return idx
    subset, _ = train_test_split(idx, train_size=fraction, stratify=labels[idx], random_state=seed)
    return subset


class SpectrogramDataset(Dataset):
    def __init__(self, images, labels, idx, mean, std):
        self.images = images[idx]
        self.labels = torch.as_tensor(labels[idx])
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i].float() / 255.0
        return (x - self.mean) / self.std, self.labels[i]


def make_loader(data, config, idx, batch_size=32, shuffle=False):
    """ResNet gets the RGB image with ImageNet normalisation, our CNNs the grey-scale image."""
    if config["model"] == "resnet18":
        ds = SpectrogramDataset(data["rgb"], data["labels"], idx, IMAGENET_MEAN, IMAGENET_STD)
    else:
        ds = SpectrogramDataset(data["gray"], data["labels"], idx, [data["gray_mean"]], [data["gray_std"]])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Accumulate exact sample-weighted loss and correct count
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total_samples += batch_size

    return running_loss / total_samples, correct / total_samples


def make_optimizer(name, params, lr, weight_decay=0.0):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(name)


def train_model(config, data, train_idx, val_idx, device, seed=42, verbose=False):
    """
    Builds the model described by `config` and trains it for config['epochs'] epochs
    (with config['optimizer'], config['lr'], config['batch_size']).
    If val_idx is given, the validation loss/accuracy is recorded after every epoch.
    Returns the trained model and the history.
    """
    set_seed(seed)
    model = MF.build_model(config, len(data["classes"]), tuple(data["gray"].shape[-2:])).to(device)
    train_loader = make_loader(data, config, train_idx, config["batch_size"], shuffle=True)
    val_loader = make_loader(data, config, val_idx, 64) if val_idx is not None else None

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(
        config["optimizer"],
        params,
        config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    scheduler_patience = config.get("lr_scheduler_patience")
    scheduler = None
    if val_loader is not None and scheduler_patience is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=scheduler_patience,
            min_lr=config.get("min_lr", 1e-5),
        )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    start = time.time()
    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    epochs_without_improvement = 0
    min_delta = config.get("min_delta", 1e-4)
    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_loss)
        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if scheduler is not None:
                scheduler.step(val_loss)
            if best_state is None or val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        if verbose:
            msg = f"epoch {epoch:2d} | train loss {train_loss:.3f}"
            if val_loader is not None:
                msg += f" | val loss {val_loss:.3f} acc {val_acc:.3f}"
            print(msg)
        early_stopping_patience = config.get("early_stopping_patience")
        if (
            val_loader is not None
            and early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            if verbose:
                print(f"Early stopping at epoch {epoch}; best validation loss at epoch {best_epoch}.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss
        history["best_val_acc"] = history["val_acc"][best_epoch - 1]
    history["epochs_ran"] = len(history["train_loss"])
    history["train_time"] = time.time() - start
    return model, history


def predict(model, data, config, idx, device):
    """Returns class probabilities (N, C) and predicted labels (N,)."""
    model.eval()
    loader = make_loader(data, config, idx, 64)
    probs = []
    with torch.no_grad():
        for inputs, _ in loader:
            probs.append(torch.softmax(model(inputs.to(device)), dim=1).cpu())
    probs = torch.cat(probs).numpy()
    return probs, probs.argmax(axis=1)


def get_embeddings(model, data, config, idx, device):
    """Hidden-layer activations of the model (CNN embeddings)."""
    model.eval()
    loader = make_loader(data, config, idx, 64)
    with torch.no_grad():
        return torch.cat([model.embed(x.to(device)).cpu() for x, _ in loader]).numpy()


# ---------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------
def accuracy_ci(y_true, y_pred, z=1.96):
    """Accuracy with a 95% Wilson confidence interval."""
    n = len(y_true)
    p = np.mean(np.asarray(y_true) == np.asarray(y_pred))
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return p, centre - half, centre + half


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test. Returns (#only A right, #only B right, p-value)."""
    from scipy.stats import binomtest
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    if only_a + only_b == 0:
        return only_a, only_b, 1.0
    return only_a, only_b, binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue


def plot_history(history: dict, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train')
    if history.get('val_loss'):
        ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('epoch')
    ax1.legend()

    if history.get('val_acc'):
        ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(predictions: dict, y_true, classes):
    """predictions: {model name: predicted labels}. Row-normalised confusion matrices side by side."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(6.5 * len(predictions), 6))
    for ax, (name, y_pred) in zip(np.atleast_1d(axes), predictions.items()):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False,
                                                                xticks_rotation=45, values_format=".2f")
        ax.set_title(f"{name} (acc {np.mean(np.asarray(y_pred) == np.asarray(y_true)):.3f})")
    plt.tight_layout()
    plt.show()
