import torch
import os
import random
import numpy as np
import torch.nn.functional as F
from typing import Sequence, Tuple, List, Optional
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from config import CFG

# Helper functions for training

def set_seed(seed: int):
    # Track seed for training reproducability
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def focal_loss_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """
    Binary focal loss on logits.
    logits/targets: [B,1,H,W], targets in [0,1]
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    return loss.mean()

def robust_normalize(img: np.ndarray, p: Tuple[float, float] = (1, 99)) -> np.ndarray:
    """Normalize to [0,1] using percentiles to clip outliers"""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, p)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo + 1e-6)
    return np.clip(out, 0, 1).astype(np.float32)

class FormDataset(Dataset):
    """
    imgs: list/array of (H,W) numpy arrays
    heatmaps: list/array of (H,W) numpy arrays in [0,1]
    """
    def __init__(self, imgs: Sequence[np.ndarray], heatmaps: Sequence[np.ndarray], normalize_inputs: bool = True):
        assert len(imgs) == len(heatmaps)
        self.imgs = imgs
        self.heatmaps = heatmaps
        self.normalize_inputs = normalize_inputs

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = self.imgs[idx]
        hm = self.heatmaps[idx]

        if self.normalize_inputs:
            img = robust_normalize(img)

        img = img.astype(np.float32)
        hm = hm.astype(np.float32)
        hm = np.clip(hm, 0.0, 1.0)

        x = torch.from_numpy(img)[None, ...]  # [1,H,W]
        y = torch.from_numpy(hm)[None, ...]   # [1,H,W]
        
        return x, y
    
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, cfg: CFG) -> float:
    model.train()
    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        loss = focal_loss_with_logits(logits, y, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
        loss.backward()

        # Prevents exploding gradients by clipping is total norm of grad is too big
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip) 

        optimizer.step()

        total += float(loss.item())
        n += 1

    return total / max(1, n)

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, cfg: CFG) -> float:
    model.eval()
    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        logits = model(x)
        loss = focal_loss_with_logits(logits, y, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)

        total += float(loss.item())
        n += 1

    return total / max(1, n)

def load_csv_image_pairs(
    data_dir: str,
    label_dir: str,
    val_frac: float = 0.2,
    label_suffix: str = "_label_heatmap",
    random_state: Optional[int] = None,
    shuffle: bool = True,
    strict: bool = True,  # if True, error on missing/duplicate keys
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Returns: train_imgs, train_hms, val_imgs, val_hms
    Each is a list of np.float32 arrays. Pairing is enforced by shared keys.
    """

    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be between 0 and 1 (exclusive).")

    def list_top_level_csv_files(folder: str) -> List[str]:
        out = []
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and name.lower().endswith(".csv"):
                out.append(path)
        return out

    # ---- Build data map: key = stem (filename without extension)
    data_map = {}
    for path in list_top_level_csv_files(data_dir):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in data_map:
            msg = f"Duplicate data key '{stem}':\n  {data_map[stem]}\n  {path}"
            if strict:
                raise ValueError(msg)
            else:
                print("Warning:", msg, "\nUsing last one.")
        data_map[stem] = path

    # ---- Build label map: key = stem with label_suffix stripped
    label_map = {}
    for path in list_top_level_csv_files(label_dir):
        stem = os.path.splitext(os.path.basename(path))[0]
        if not stem.endswith(label_suffix):
            # Not a label heatmap file by naming convention; skip
            continue
        key = stem[: -len(label_suffix)]
        if key in label_map:
            msg = f"Duplicate label key '{key}':\n  {label_map[key]}\n  {path}"
            if strict:
                raise ValueError(msg)
            else:
                print("Warning:", msg, "\nUsing last one.")
        label_map[key] = path

    data_keys = set(data_map.keys())
    label_keys = set(label_map.keys())
    common_keys = sorted(data_keys & label_keys)

    missing_labels = sorted(data_keys - label_keys)
    missing_data = sorted(label_keys - data_keys)

    if strict:
        if missing_labels:
            raise ValueError(
                f"{len(missing_labels)} data files have no matching label key. "
                f"Example: {missing_labels[0]}"
            )
        if missing_data:
            raise ValueError(
                f"{len(missing_data)} label files have no matching data key. "
                f"Example: {missing_data[0]}"
            )
    else:
        if missing_labels:
            print(f"Warning: {len(missing_labels)} data files missing labels. Example: {missing_labels[0]}")
        if missing_data:
            print(f"Warning: {len(missing_data)} labels missing data. Example: {missing_data[0]}")

    if not common_keys:
        raise RuntimeError("No matched (data,label) keys found. Check naming and suffix.")

    # ---- Optional shuffle (keys are shuffled, preserving alignment)
    if shuffle:
        rng = np.random.default_rng(random_state)
        common_keys = list(rng.permutation(common_keys))

    # ---- Load arrays in key order
    imgs: List[np.ndarray] = []
    hms: List[np.ndarray] = []
    for k in common_keys:
        img = np.loadtxt(data_map[k], delimiter=",", dtype=np.float32)
        hm = np.loadtxt(label_map[k], delimiter=",", dtype=np.float32)
        imgs.append(np.asarray(img, dtype=np.float32))
        hms.append(np.asarray(hm, dtype=np.float32))

    # ---- Split
    n_total = len(imgs)
    n_val = int(round(val_frac * n_total))
    n_train = n_total - n_val

    train_imgs = imgs[:n_train]
    train_hms  = hms[:n_train]
    val_imgs   = imgs[n_train:]
    val_hms    = hms[n_train:]

    # Sanity
    assert len(train_imgs) == len(train_hms)
    assert len(val_imgs) == len(val_hms)

    return train_imgs, train_hms, val_imgs, val_hms

def pad_collate(batch):
    # Handle inputs of different sizes by padding
    xs, ys = zip(*batch)  # x is inputs, y is labels. Seperate them. Each is [1,H,W]

    max_h = max(x.shape[-2] for x in xs)
    max_w = max(x.shape[-1] for x in xs)

    def pad(t):
        _, h, w = t.shape
        pad_h = max_h - h
        pad_w = max_w - w
        return torch.nn.functional.pad(t, (0, pad_w, 0, pad_h)) # Pad each image/label with 0s on the right and bottom to match the largest image/label

    xs = torch.stack([pad(x) for x in xs])
    ys = torch.stack([pad(y) for y in ys])

    return xs, ys