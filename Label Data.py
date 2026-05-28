from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import napari
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel

# Now that data is cropped and the ROIs are selected, label the vortices using this gui. A number of hotkeys
# are listed in the panel for convenience. A csv file with a list of coordinates of each vortex is stored. 
# Unwanted ROIs can be moved to a trash folder in Cropped Data. 
# **Every image must be labelled before training**

# Cropped Data: Folder containing the csv ROIs selected by the Crop Data script. Now will contain a trash subfolder for unwanted ROIs
# Label Coors: The output folder where the coordinates for each ROI will be stored in a corresponding csv

# Hotkey note: No coordinate csv file will be output until saving happens. If there are no vortices in the ROI, save with no points selected. 
# The empty csv serves to provide training when no vortices are present. 

def load_image_as_array(path: Path) -> np.ndarray:
    """Load a CSV as a 2D 'image'."""
    arr = pd.read_csv(path, header=None).to_numpy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D CSV image, got shape {arr.shape} for {path}")
    return arr

def move_to_trash(path: Path, trash_dir: Path) -> Path:
    """Move a file to a trash directory (same filename)."""
    trash_dir.mkdir(parents=True, exist_ok=True)
    dest = trash_dir / path.name
    return path.rename(dest)

def out_csv_for_image(out_dir: Path, image_path: Path) -> Path:
    """Output file path for an image."""
    return out_dir / f"{image_path.stem}_label_coors.csv"

def is_image_processed(out_dir: Path, image_path: Path) -> bool:
    """Done iff the output coords CSV exists."""
    return out_csv_for_image(out_dir, image_path).exists()

def save_points(points: np.ndarray, out_path: Path) -> None:
    """Save napari points as y,x with header; allow 0 rows."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if points is None or len(points) == 0:
        df = pd.DataFrame(columns=["y", "x"])
    else:
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError(f"Unexpected points shape: {pts.shape}")
        df = pd.DataFrame(pts[:, :2], columns=["y", "x"])
    df.to_csv(out_path, index=False)

class PointCsvExporter(QWidget):
    def __init__(self, viewer, input_files, out_dir, trash_dir):
        super().__init__()
        self.viewer = viewer
        self.input_files = input_files
        self.out_dir = out_dir
        self.trash_dir = trash_dir

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trash_dir.mkdir(parents=True, exist_ok=True)

        self.idx = self._find_first_unprocessed(0)
        if self.idx is None:
            raise RuntimeError("All images already have label_coors CSVs.")

        self._suspend = False

        self.image_layer = None
        self.points_layer = None

        # UI
        self.label = QLabel("")
        self.btn_save = QPushButton("Save & Next (w)")
        self.btn_skip = QPushButton("Skip (s)")
        self.btn_delete = QPushButton("Delete → Trash (d)")
        self.btn_undo = QPushButton("Undo Last (z)")
        self.btn_prev = QPushButton("Prev (p)")
        self.btn_next = QPushButton("Next (n)")

        layout = QVBoxLayout()
        for w in (
            self.label,
            self.btn_save,
            self.btn_skip,
            self.btn_delete,
            self.btn_undo,
            self.btn_prev,
            self.btn_next,
        ):
            layout.addWidget(w)
        self.setLayout(layout)

        # Buttons
        self.btn_save.clicked.connect(self.save_and_next)
        self.btn_skip.clicked.connect(self.skip_and_next)
        self.btn_delete.clicked.connect(self.delete_and_next)
        self.btn_undo.clicked.connect(self.undo_last)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)

        # Hotkeys
        viewer.bind_key("w", lambda v: self.save_and_next(), overwrite=True)
        viewer.bind_key("s", lambda v: self.skip_and_next(), overwrite=True)
        viewer.bind_key("d", lambda v: self.delete_and_next(), overwrite=True)
        viewer.bind_key("z", lambda v: self.undo_last(), overwrite=True)
        viewer.bind_key("n", lambda v: self.next_image(), overwrite=True)
        viewer.bind_key("p", lambda v: self.prev_image(), overwrite=True)
        viewer.bind_key("Control-S", lambda v: self.save_and_next(), overwrite=True)
        viewer.bind_key("Command-S", lambda v: self.save_and_next(), overwrite=True)

        self._load_current_image()
        self._setup_points_layer()

    def _find_first_unprocessed(self, start):
        for i in range(start, len(self.input_files)):
            if not is_image_processed(self.out_dir, self.input_files[i]):
                return i
        return None

    def _find_prev_unprocessed(self, start):
        for i in range(start, -1, -1):
            if not is_image_processed(self.out_dir, self.input_files[i]):
                return i
        return None

    @property
    def current_path(self):
        return self.input_files[self.idx]

    def _update_label(self):
        self.label.setText(
            f"{self.idx + 1}/{len(self.input_files)} | {self.current_path.name}"
        )

    def _load_current_image(self):
        self._suspend = True
        arr = load_image_as_array(self.current_path)

        if self.image_layer is None:
            self.image_layer = self.viewer.add_image(
                arr,
                name="image",
                colormap="plasma",
                contrast_limits=[np.percentile(arr, 1), np.percentile(arr, 99.9)],
            )
        else:
            self.image_layer.data = arr
            self.image_layer.contrast_limits = [
                np.percentile(arr, 1),
                np.percentile(arr, 99.9),
            ]

        self._update_label()
        self._suspend = False

    def _setup_points_layer(self):
        self._suspend = True

        if self.points_layer is None:
            self.points_layer = self.viewer.add_points(
                name="labels",
                size=7,
                face_color="green",
                opacity=0.4
            )

        self.points_layer.data = np.empty((0, 2), dtype=float)
        self.points_layer.mode = "add"
        self._suspend = False

    def undo_last(self):
        data = np.asarray(self.points_layer.data)
        if data.shape[0] > 0:
            self.points_layer.data = data[:-1, :]

    def save_and_next(self):
        out_csv = out_csv_for_image(self.out_dir, self.current_path)
        save_points(self.points_layer.data, out_csv)
        show_info(f"Saved {out_csv.name}")
        self.next_image()

    def skip_and_next(self):
        show_info(f"Skipped {self.current_path.name}")
        self.next_image()

    def delete_and_next(self):
        p = self.current_path
        dest = self.trash_dir / p.name
        try:
            p.rename(dest)
            show_info(f"Moved {p.name} → trash/")
        except Exception as e:
            show_info(f"Failed to move to trash: {e}")
        self.next_image()

    def next_image(self):
        nxt = self._find_first_unprocessed(self.idx + 1)
        if nxt is None:
            show_info("No more unprocessed images.")
            return
        self.idx = nxt
        self._load_current_image()
        self._setup_points_layer()

    def prev_image(self):
        prv = self._find_prev_unprocessed(self.idx - 1)
        if prv is None:
            show_info("No previous unprocessed images.")
            return
        self.idx = prv
        self._load_current_image()
        self._setup_points_layer()

def main(input_dir, output_dir, pattern="*.csv"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    trash_dir = input_dir / "Trash"

    files = sorted(
        p for p in input_dir.glob(pattern)
        if p.parent == input_dir
    )
    if not files:
        raise RuntimeError("No input files found.")

    viewer = napari.Viewer()
    widget = PointCsvExporter(viewer, files, output_dir, trash_dir)
    viewer.window.add_dock_widget(widget, area="right")

    show_info(
        "Click to add points (y,x).\n"
        "Output: <stem>_label_coors.csv (one per image).\n"
        "Images with existing outputs are skipped.\n"
        "Hotkeys: w=save&next, s=skip, d=delete input, z=undo, n=next, p=prev"
    )

    napari.run()


if __name__ == "__main__":
    main("Cropped Data", "Label Coors", "*.csv")
    