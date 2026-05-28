from model import *
from functions import *
from typing import Sequence
from config import CFG
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Function defining the training process. Plotting can be removed if unwanted. Modify output location if needed.

def train_model(
    train_imgs: Sequence[np.ndarray],
    train_heatmaps: Sequence[np.ndarray],
    val_imgs: Sequence[np.ndarray],
    val_heatmaps: Sequence[np.ndarray],
    cfg: CFG,
) -> nn.Module:
    
    set_seed(cfg.seed)

    # Create Model
    model = HeatmapUNet(cfg.base, cfg.kern, cfg.stride, cfg.pad, cfg.dil).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    # Form Dataset
    train_ds = FormDataset(train_imgs, train_heatmaps, normalize_inputs=True)
    val_ds = FormDataset(val_imgs, val_heatmaps, normalize_inputs=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        collate_fn=pad_collate
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        collate_fn=pad_collate
    )

    # Train & Validate
    tl = np.zeros((cfg.epochs,2))
    vl = np.zeros((cfg.epochs,2))
    with tqdm(total=cfg.epochs, disable=False) as pbar:
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, train_dl, optimizer, cfg)
            val_loss = validate(model, val_dl, cfg)
            tl[epoch-1,:] = epoch, train_loss
            vl[epoch-1,:] = epoch, val_loss
            pbar.update(1)

    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": tl,
        "validation_loss": vl,
        "config": vars(cfg)}
    torch.save(ckpt, f"Models/{cfg.name}_model.pt") # Output location of the model pytorch file

    print("Training Loss:", tl[-1,1])
    print("Validation Loss:", vl[-1,1])

    plt.plot(tl[:,0],tl[:,1], label="Training Loss")
    plt.plot(vl[:,0],vl[:,1], label="Validation Loss")
    plt.title("Loss During Training")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()