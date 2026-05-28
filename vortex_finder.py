import torch
import os
from model import *
from fitting import robust_normalize
import pandas as pd
import numpy as np
import skimage

# Step 1/2 of analysis pipeline. This script runs all csv images in the given folder through the CNN and 
# produces a new corresponding csv file for each containing the predicted coordinates of the vortices. 
# After this script, only the first three columns will be populated. 

# This script uses the blob_log function. Its parameters may need to be tuned a bit. The idea is that, once found,
# the same parameters can be used for many experiment types rather than having to find new parameters for every FOV. 
# As of 4/26/2026, it can struggle with dim vortices, high vortiex densities, and certain moat structures. 

folder_path = r"Folder Path" # Path to the folder containing the csv images
ckpt_path = r"Path to the model ending in .pt" # Path to the .pt file containing the CNN to be used
header = "row,column,radius, xc, yc, z, xc std, yc std, z std"

# Blob parameters (blob_log)
min_r = 5
max_r = 10
n_r = 10
thr_rel = 0.6

def Model(ckpt_path, model_ctor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = model_ctor(ckpt["config"])
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval()

model = Model(
    ckpt_path=ckpt_path,
    model_ctor=lambda cfg: HeatmapUNet(cfg["base"], cfg["kern"], cfg["stride"], cfg["pad"], cfg["dil"]),
)

device = next(model.parameters()).device

for img_name in os.listdir(folder_path):
    if img_name[-13:] == "_vortices.csv":
        continue
    path = os.path.join(folder_path, img_name)

    img = np.array(pd.DataFrame.to_numpy(pd.read_csv(path,header=None)), dtype=np.float32)
    img = robust_normalize(img)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        heatmap = (torch.sigmoid((logits-torch.median(logits))) - 0.5)*2

    heatmap = heatmap.squeeze().detach().cpu().numpy()
    blobs = skimage.feature.blob_log(heatmap, min_sigma=min_r/np.sqrt(2), max_sigma=max_r/np.sqrt(2), num_sigma=n_r, threshold_rel=thr_rel)
    blobs[:,-1] = np.sqrt(2) * blobs[:,-1]
    
    np.savetxt(f"{path[:-4]}_vortices.csv", blobs, delimiter=",", header=header, comments='')

