from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple
import numpy as np

# The "label" for the NN training is a heatmap where the center of the vortex is "hot". This script
# takes the cropped ROIs and their corresponding vortex labels and make a heatmap with gaussian spots 
# for each vortex. The folder names can be changed at the bottom

# Cropped Data: The same folder as before that contains all the selected ROI csv files
# Label Coors: The same folder that contains the csv files with each ROIs vortex coordinates
# Label Heatmaps: The output folder containing a csv file for each ROI serving as the heatmap label used during training
# sigma: Determines how wide the gaussians will be
# radius: determins how far from the center of a vortex the gaussian will be applied


def generate_label_heatmaps_from_csv_images(
    cropped_data_dir: str | os.PathLike,
    label_coors_dir: str | os.PathLike,
    label_heatmaps_dir: str | os.PathLike,
    *,
    sigma: float = 1.0,
    radius: int = 3,
    coordinate_columns: Tuple[str, str] = ("x", "y"),
    allow_header_guess: bool = True,
    verbose: bool = True,
) -> None:
    """
    Generate per-image heatmaps (CSV) from label coordinate CSVs.

    Directory / naming assumptions:
      - cropped_data_dir contains image CSV files (2D arrays), possibly in subfolders.
      - A subfolder named "Trash" anywhere under cropped_data_dir is ignored.
      - label_coors_dir contains coordinate CSVs named:
            "<image_stem>_label_coors.csv"
        where image_stem is the image csv filename without extension.
      - Outputs are written to label_heatmaps_dir as:
            "<image_stem>_label_heatmap.csv"

    Heatmap:
      - same height/width as the image CSV
      - values on [0, 1]
      - for each coordinate, writes a Gaussian centered at (x, y)
      - overlapping Gaussians are combined via max() (not sum)

    Parameters
    ----------
    sigma:
        Gaussian standard deviation in pixels.
        A common choice is sigma=1.0 with radius=3 for a compact bump.
        If you want a wider bump, increase sigma (e.g., 1.5, 2.0).
    radius:
        Window radius around each point to evaluate (speeds up computation).
        The window is (2*radius+1)^2.
    coordinate_columns:
        If the label CSV has headers, these are the column names to use.
        If not found, function falls back to first two columns.
    allow_header_guess:
        If True, tries to interpret first row as header; otherwise assumes no header.
    """

    cropped_data_dir = Path(cropped_data_dir)
    label_coors_dir = Path(label_coors_dir)
    label_heatmaps_dir = Path(label_heatmaps_dir)
    label_heatmaps_dir.mkdir(parents=True, exist_ok=True)

    # Precompute Gaussian kernel for integer offsets in [-radius, radius]
    # Kernel peak is 1 at (0,0) and decays with sigma.
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    # Ensure exact 1.0 at center in case of floating rounding
    kernel[radius, radius] = 1.0

    def _is_in_trash(path: Path) -> bool:
        return any(part.lower() == "trash" for part in path.parts)

    def _load_image_shape(csv_path: Path) -> Tuple[int, int]:
        # Load as numeric; we only need shape.
        arr = np.loadtxt(csv_path, delimiter=",")
        if arr.ndim != 2:
            raise ValueError(f"Image CSV must be 2D, got shape {arr.shape} for {csv_path}")
        return int(arr.shape[0]), int(arr.shape[1])

    def _load_coords(label_csv: Path) -> np.ndarray:
        """
        Returns Nx2 array of (x, y) float coords.
        Supports:
          - header with columns named per coordinate_columns
          - no header: first two columns are used
        """
        # Try header-based parsing first (without pandas)
        # We'll do a small sniff: read first line and see if it contains non-numeric tokens.
        with label_csv.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        def _looks_like_header(line: str) -> bool:
            # If any token fails float conversion, assume header.
            toks = [t.strip() for t in line.split(",")]
            for t in toks:
                if t == "":
                    continue
                try:
                    float(t)
                except ValueError:
                    return True
            return False

        has_header = allow_header_guess and _looks_like_header(first_line)

        if has_header:
            # Parse header row to locate x/y columns
            header = [h.strip() for h in first_line.split(",")]
            x_name, y_name = coordinate_columns
            try:
                x_idx = header.index(x_name)
                y_idx = header.index(y_name)
            except ValueError:
                # Fall back to first two columns
                x_idx, y_idx = 0, 1

            data = np.loadtxt(label_csv, delimiter=",", skiprows=1)
            if data.size == 0:
                return np.zeros((0, 2), dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            coords = data[:, [x_idx, y_idx]].astype(np.float32)
        else:
            data = np.loadtxt(label_csv, delimiter=",")
            if data.size == 0:
                return np.zeros((0, 2), dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            coords = data[:, :2].astype(np.float32)

        return coords

    def _write_heatmap(out_csv: Path, heatmap: np.ndarray) -> None:
        np.savetxt(out_csv, heatmap, delimiter=",", fmt="%.6f")

    # Walk cropped data dir recursively, skipping Trash
    image_csvs: list[Path] = []
    for p in cropped_data_dir.rglob("*.csv"):
        if _is_in_trash(p.relative_to(cropped_data_dir)):
            continue
        image_csvs.append(p)

    if verbose:
        print(f"Found {len(image_csvs)} image CSV(s) under {cropped_data_dir} (excluding Trash).")

    for img_csv in image_csvs:
        image_stem = img_csv.stem  # filename without extension
        label_csv = label_coors_dir / f"{image_stem}_label_coors.csv"

        if not label_csv.exists():
            if verbose:
                print(f"Skipping (no label file): {img_csv.name} -> expected {label_csv.name}")
            continue

        H, W = _load_image_shape(img_csv)
        coords = _load_coords(label_csv)

        heatmap = np.zeros((H, W), dtype=np.float32)

        # Place gaussians with max-compositing
        for (x_f, y_f) in coords:
            # Interpret coords as pixel indices (x=col, y=row)
            x0 = int(round(float(x_f)))
            y0 = int(round(float(y_f)))

            # Skip coords fully out of bounds
            if x0 < 0 or x0 >= W or y0 < 0 or y0 >= H:
                continue

            # Window bounds in image
            x1 = max(0, x0 - radius)
            x2 = min(W, x0 + radius + 1)
            y1 = max(0, y0 - radius)
            y2 = min(H, y0 + radius + 1)

            # Corresponding window bounds in kernel
            kx1 = x1 - (x0 - radius)
            kx2 = kx1 + (x2 - x1)
            ky1 = y1 - (y0 - radius)
            ky2 = ky1 + (y2 - y1)

            patch = kernel[ky1:ky2, kx1:kx2]
            # Max-composite (do not sum)
            heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], patch)

        # Already in [0,1] because kernel peak is 1 and we max() patches, but clip defensively
        heatmap = np.clip(heatmap, 0.0, 1.0)

        out_csv = label_heatmaps_dir / f"{image_stem}_label_heatmap.csv"
        _write_heatmap(out_csv, heatmap)

        if verbose:
            print(f"Wrote: {out_csv.name}  (shape={H}x{W}, points={len(coords)})")

generate_label_heatmaps_from_csv_images(
    cropped_data_dir="Cropped Data",
    label_coors_dir="Label Coors",
    label_heatmaps_dir="Label Heatmaps",
    sigma=1.5,
    radius=4,
)
