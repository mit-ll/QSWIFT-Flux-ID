from functions import *
from training import *
from config import CFG


# Set input parameters. Run this file to start training. Change paths as needed

if __name__ == "__main__":
    cfg = CFG()

    train_imgs, train_hms, val_imgs, val_hms = load_csv_image_pairs(r"Labelling/Cropped Data", r"Labelling/Label Heatmaps", val_frac=cfg.val_frac, random_state=cfg.seed)

    model = train_model(train_imgs, train_hms, val_imgs, val_hms, cfg)