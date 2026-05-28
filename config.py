import torch
from dataclasses import dataclass

# Contains all metadata for a training run. Ensure a new name for each run to prevent overwrite

@dataclass
class CFG:
    # Info
    name: str|None = "Run Name"
    notes: str|None = "Put notes about the training run here"
    
    # Training 
    lr: float = 2e-4
    wd: float = 1e-4
    epochs: int = 30
    batch_size: int = 4 # May run out of memory if this is too high
    num_workers: int = 0
    grad_clip: float = 1.0
    val_frac: float = 0.2

    # Loss
    focal_alpha: float = 0.55
    focal_gamma: float = 1.0

    # Architecture
    base: int = 32
    kern: int = 3
    stride: int = 1
    pad: int = 1
    dil: int = 1

    # System
    seed: int = 0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

