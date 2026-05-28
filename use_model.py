import torch
import os
from model import *
from fitting import robust_normalize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Used for troubleshooting and fine tuning. Change "path" to the path of a csv. This will display the normalized image and the output heatmap of the CNN

def Model(name, model_ctor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join("Models", f"{name}_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = model_ctor(ckpt["config"])
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval()

model = Model(
    name="Test7",
    model_ctor=lambda cfg: HeatmapUNet(cfg["base"], cfg["kern"], cfg["stride"], cfg["pad"], cfg["dil"]),
)

device = next(model.parameters()).device
path = r"Path to csv file"
img = np.array(pd.DataFrame.to_numpy(pd.read_csv(path,header=None)), dtype=np.float32)
img = robust_normalize(img)
img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(img)
    heatmap = (torch.sigmoid((logits-torch.median(logits))) - 0.5)*2
    #heatmap = torch.sigmoid(logits)

img = img.squeeze().detach().cpu().numpy() 
heatmap = heatmap.squeeze().detach().cpu().numpy() 

fig, ax = plt.subplots(2, 1, figsize=(3,5), sharey=True)
im0 = ax[0].imshow(img, cmap = "jet", aspect = "equal", origin="lower")
im1 = ax[1].imshow(heatmap, cmap = "jet", vmin=0, vmax=1, aspect = "equal", origin="lower")
fig.colorbar(im1, ax=ax.ravel().tolist())
plt.show()

