from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import napari
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel

# A helper file to prepare images for training. A gui will be launched. Each image in Full Data will be shown one at a time. Drag 
# ROIs which will make up the training set. You may use many or few regions per image. It is often easier to label and train smaller
# images rather than whole FOVs. The folder names can be changed at the "main" call at the bottom. 

# Full Data: A folder that should contain csv files of experiment images before running this script
# Cropped Data: The output folder in which the cropped subimages will be stored. 

def load_image_as_array(path: Path) -> np.ndarray:
    """Load a CSV as a 2D 'image'."""
    return pd.read_csv(path, header=None).to_numpy()

def rect_to_slices(rect_vertices: np.ndarray):
    rows = rect_vertices[:, 0]
    cols = rect_vertices[:, 1]
    r0, r1 = int(np.floor(rows.min())), int(np.ceil(rows.max()))
    c0, c1 = int(np.floor(cols.min())), int(np.ceil(cols.max()))
    return slice(r0, r1), slice(c0, c1)

def roi_files_for_image(out_dir: Path, image_stem: str) -> list[Path]:
    pat = re.compile(rf"^{re.escape(image_stem)}__roi_\d+__.*\.csv$")
    return [p for p in out_dir.iterdir() if p.is_file() and pat.match(p.name)]

def is_image_processed(out_dir: Path, image_path: Path) -> bool:
    return len(roi_files_for_image(out_dir, image_path.stem)) > 0

class RoiCsvExporter(QWidget):
    def __init__(self, viewer, input_files, out_dir):
        super().__init__()
        self.viewer = viewer
        self.input_files = input_files
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.idx = self._find_first_unprocessed(0)
        if self.idx is None:
            raise RuntimeError("All images already have ROI CSVs.")

        self.roi_counter_for_image = 0
        self._last_n_shapes = 0

        # Suspend flag prevents exports during navigation / programmatic edits
        self._suspend_export = False
        self._current_stem = self.current_path.stem

        self.image_layer = None
        self.shapes_layer = None

        self.label = QLabel("")
        self.btn_prev = QPushButton("Prev")
        self.btn_next = QPushButton("Next")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_prev)
        layout.addWidget(self.btn_next)
        self.setLayout(layout)

        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)

        viewer.bind_key("n", lambda v: self.next_image(), overwrite=True)
        viewer.bind_key("p", lambda v: self.prev_image(), overwrite=True)

        self._load_current_image()
        self._setup_shapes_layer()

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

    def _load_current_image(self):
        self._suspend_export = True

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
            self.image_layer.contrast_limits = [np.percentile(arr, 1), np.percentile(arr, 99.9)]

        # Track which image we are on
        self._current_stem = self.current_path.stem

        # If you ever revisit an image, continue numbering; with skipping it will usually be 0
        self.roi_counter_for_image = len(roi_files_for_image(self.out_dir, self._current_stem))

        self._update_label()

        self._suspend_export = False

    def _setup_shapes_layer(self):
        self._suspend_export = True

        if self.shapes_layer is None:
            self.shapes_layer = self.viewer.add_shapes(
                name="rois",
                shape_type="rectangle",
                edge_width=2,
                opacity=0.25,
            )
            self.shapes_layer.events.data.connect(self._on_shapes_changed)

        # Always clear shapes when moving images
        self.shapes_layer.data = []
        self.shapes_layer.mode = "add_rectangle"

        # Reset the "already exported" count to match cleared layer
        self._last_n_shapes = 0

        self._suspend_export = False

    def _update_label(self):
        self.label.setText(
            f"{self.idx + 1}/{len(self.input_files)}  |  "
            f"{self.current_path.name}  |  "
            f"ROIs saved: {self.roi_counter_for_image}"
        )

    def _on_shapes_changed(self, event=None):
        # If we're navigating or clearing shapes programmatically, do not export
        if self._suspend_export:
            self._last_n_shapes = len(self.shapes_layer.data)
            return

        n = len(self.shapes_layer.data)
        if n <= self._last_n_shapes:
            self._last_n_shapes = n
            return

        img = np.asarray(self.image_layer.data)
        H, W = img.shape[:2]

        for i in range(self._last_n_shapes, n):
            rect = np.asarray(self.shapes_layer.data[i])
            rsl, csl = rect_to_slices(rect)

            r0, r1 = max(0, rsl.start), min(H, rsl.stop)
            c0, c1 = max(0, csl.start), min(W, csl.stop)

            if r1 <= r0 or c1 <= c0:
                continue

            roi = img[r0:r1, c0:c1]

            stem = self._current_stem  # safer than self.current_path.stem during transitions
            out_name = (
                f"{stem}__roi_{self.roi_counter_for_image:03d}"
                f"__r{r0}-{r1}_c{c0}-{c1}.csv"
            )

            pd.DataFrame(roi).to_csv(
                self.out_dir / out_name,
                index=False,
                header=False,
            )

            self.roi_counter_for_image += 1
            show_info(f"Saved {out_name}")

        self._last_n_shapes = n
        self._update_label()

    def next_image(self):
        nxt = self._find_first_unprocessed(self.idx + 1)
        if nxt is None:
            show_info("No more unprocessed images.")
            return

        # Suspend exports during the switch to prevent re-saving old ROIs
        self._suspend_export = True
        if self.shapes_layer is not None:
            self.shapes_layer.data = []
        self._last_n_shapes = 0

        self.idx = nxt

        self._suspend_export = False
        self._load_current_image()
        self._setup_shapes_layer()

    def prev_image(self):
        prv = self._find_prev_unprocessed(self.idx - 1)
        if prv is None:
            show_info("No previous unprocessed images.")
            return

        # Suspend exports during the switch to prevent re-saving old ROIs
        self._suspend_export = True
        if self.shapes_layer is not None:
            self.shapes_layer.data = []
        self._last_n_shapes = 0

        self.idx = prv

        self._suspend_export = False
        self._load_current_image()
        self._setup_shapes_layer()

def main(input_dir, output_dir, pattern="*.csv"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = sorted(input_dir.glob(pattern))
    if not files:
        raise RuntimeError("No input files found.")

    viewer = napari.Viewer()
    widget = RoiCsvExporter(viewer, files, output_dir)
    viewer.window.add_dock_widget(widget, area="right")

    show_info(
        "Draw rectangles to save ROIs.\n"
        "Each rectangle → one CSV.\n"
        "Images with existing ROI CSVs are skipped.\n"
        "Hotkeys: n=next, p=prev"
    )

    napari.run()


if __name__ == "__main__":
    main("Full Data", "Cropped Data", "*.csv")
