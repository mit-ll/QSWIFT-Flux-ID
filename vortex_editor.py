from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import napari
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel
from fitting import *

# Step 2/2 in the analysis pipeline. This script is where any corrections to the vortex finder are made, along with fitting.
# It produces a gui that will show each image, along with an overlay of the identified vortices. Use the controls to add vortices
# or select vortices to remove. Groups of false positives can be selected by dragging for removal in select mode. The remaining  
# columns in the _vortices csv will be populated with the fit information 

# Note: Fitting/updates will not occur until saved. Be sure to use the space bar to finalize changes. 

# Config
DATA_FOLDER = Path("Folder Path")
VORTEX_SUFFIX = "_vortices.csv"
HEADER = "row,column,radius, xc, yc, z, xc std, yc std, z std"
DEFAULT_RADIUS = 5.0  # px

# Robust normalization
def robust_normalize(img: np.ndarray, p: Tuple[float, float] = (1, 99)) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, p)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo + 1e-6)
    return np.clip(out, 0, 1).astype(np.float32)

# CSV helpers
def load_image_csv(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim != 2:
        raise ValueError(f"{path} must be a 2D CSV image. Got shape {arr.shape}")
    return arr

def vortices_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}{VORTEX_SUFFIX}")

def load_vortices(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 3), dtype=float)
    txt = path.read_text().strip()
    if txt == "":
        return np.empty((0, 3), dtype=float)

    arr = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    if arr.size == 0:
        return np.empty((0, 3), dtype=float)
    arr = np.atleast_2d(arr)

    if arr.shape[1] < 3:
        raise ValueError(f"{path} must have at least 3 columns: row,column,radius")

    return arr[:, :3].astype(float)

def save_vortices(path: Path, pts: np.ndarray, img: np.ndarray):
    pts = np.asarray(pts, dtype=float)
    pts = pts.reshape(-1, 3) if pts.size else np.empty((0, 3), dtype=float)
    if pts.shape[0] == 0:
        path.write_text(HEADER + "\n")
    else:
        params = np.zeros((np.shape(pts)[0],3))
        pstds = np.zeros((np.shape(pts)[0],3))
        for i, (yc, xc, r) in enumerate(pts):

            #MK edits 3/10/2026; make sure that if a fit fails, the CSV will still save. We will just assign failed fits to NaNs
            try: 
                param, pstd = fit_monopole(img, xc, yc, 2*r, plot=False)
            except:
                param = np.full(3, np.nan)
                pstd = np.full(3, np.nan)
                print('Failed to fit')
            #end of MK edits  
            params[i,:] = param
            pstds[i,:] = pstd
        file = np.column_stack([pts, params, pstds])   # shape (N, 9)
        np.savetxt(path, file, delimiter=",", header=HEADER, comments="")

# UI Panel
class InfoPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.title = QLabel("")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)

        self.help = QLabel(
            "Keys:\n"
            "  Update CSV: Space\n"
            "  Next Image: → Right Arrow\n"
            "  Previous Image: ← Left Arrow\n"
            "  Add Points: 2 or p\n"
            "  Select Points: 3 or s\n\n"
        )
        self.help.setWordWrap(True)
        layout.addWidget(self.help)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def set_image(self, idx: int, total: int, name: str):
        self.title.setText(f"Image {idx+1} / {total}\n{name}")

    def set_status(self, msg: str):
        self.status.setText(msg)

# Main
def run_editor(folder: Path = DATA_FOLDER):
    folder = Path(folder)
    image_paths = sorted([p for p in folder.glob("*.csv") if not p.name.endswith(VORTEX_SUFFIX)])
    if not image_paths:
        print(f"No image CSVs found in: {folder}")
        return

    images = [robust_normalize(load_image_csv(p)) for p in image_paths]
    vortex_paths = [vortices_path(p) for p in image_paths]
    vortices = [load_vortices(vp) for vp in vortex_paths]

    viewer = napari.Viewer()
    idx = 0

    image_layer = viewer.add_image(images[0], name=image_paths[0].name, colormap="viridis")

    # Points layer holds centers; radius stored as 'size' (native slider controls it)
    points_layer = viewer.add_points(
        np.empty((0, 2), dtype=float),
        name="Vortices",
        size=DEFAULT_RADIUS,
        symbol="disc",
        face_color="transparent",
        border_color="red",
        border_width=0.1,
    )

    panel = InfoPanel()
    viewer.window.add_dock_widget(panel, area="right", name="Vortex Editor")

    def load_current(i: int):
        nonlocal idx
        idx = i
        image_layer.data = images[idx]
        image_layer.name = image_paths[idx].name
        panel.set_image(idx, len(images), image_paths[idx].name)

        pts = vortices[idx]
        if pts.shape[0] == 0:
            centers = np.empty((0, 2), dtype=float)
            radii = np.array([], dtype=float)
        else:
            centers = pts[:, :2].astype(float)  # row, col
            radii = pts[:, 2].astype(float)

        points_layer.data = centers
        # IMPORTANT: in napari, points_layer.size can be scalar or per-point array.
        # We'll store radius directly in size as a per-point array.
        points_layer.size = radii if radii.size else DEFAULT_RADIUS

        points_layer.mode = "select"
        viewer.layers.selection.active = points_layer
        panel.set_status("")

    def extract_current() -> np.ndarray:
        centers = np.asarray(points_layer.data, dtype=float)
        if centers.size == 0:
            return np.empty((0, 3), dtype=float)

        # points_layer.size can be scalar or array; normalize to per-point array
        sz = points_layer.size
        if np.isscalar(sz):
            radii = np.full((centers.shape[0],), float(sz), dtype=float)
        else:
            radii = np.asarray(sz, dtype=float)
            if radii.size != centers.shape[0]:
                radii = np.full((centers.shape[0],), DEFAULT_RADIUS, dtype=float)

        return np.column_stack([centers[:, 0], centers[:, 1], radii])

    def save_current():
        vortices[idx] = extract_current()
        save_vortices(vortex_paths[idx], vortices[idx], images[idx])
        msg = f"Saved: {vortex_paths[idx].name}"
        panel.set_status(msg)
        viewer.status = msg

    # Key bindings
    @viewer.bind_key("Space")
    def _save(v):
        save_current()

    @viewer.bind_key("Right")
    def _next(v):
        load_current(min(idx + 1, len(images) - 1))

    @viewer.bind_key("Left")
    def _prev(v):
        load_current(max(idx - 1, 0))

    load_current(0)
    napari.run()

if __name__ == "__main__":
    run_editor()