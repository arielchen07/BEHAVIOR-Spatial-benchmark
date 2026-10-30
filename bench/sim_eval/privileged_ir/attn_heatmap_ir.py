"""Render the attention-map IR image from a target center (the paper's heatmap input).

A single isotropic 2D Gaussian (sigma = 8 px at 224x224) at the target's
center, colored with an analytic jet colormap: background dark blue
(0, 0, 128), peak red (255, 0, 0). Unlike the dot IR there is no camera image
underneath. A target outside the view (center None) gives a uniform dark-blue
frame. Only the first center is used: training renders the target alone. The
constants match those the training data was rendered with.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

# ── TRAIN-LOCKED constants (the training-data heatmap generator) ──────────────────
HEATMAP_SIGMA: float = 8.0  # Gaussian sigma in 224px space (~2.5-3 SigLIP patches)
JET_MAX: float = 0.875      # jet domain scale: intensity 1.0 -> PURE red
IR_RESOLUTION: int = 224


def _jet_lut() -> np.ndarray:
    """uint8[256, 3] jet LUT over v in [0, JET_MAX]. Index 0 (background) is
    dark blue (0, 0, 128); index 255 (Gaussian peak) is pure red (255, 0, 0).

    Byte-identical to the training-data heatmap generator's _jet_lut.
    """
    v = np.linspace(0.0, JET_MAX, 256, dtype=np.float64)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return np.round(np.stack([r, g, b], axis=1) * 255.0).astype(np.uint8)


_JET_LUT = _jet_lut()


def render_gaussian_heatmap(
    center: Optional[Tuple[int, int]],
    *,
    sigma: float = HEATMAP_SIGMA,
    res: int = IR_RESOLUTION,
) -> np.ndarray:
    """Standalone attention-map frame: uint8[res, res, 3].

    Byte-identical to the training-data heatmap generator's render_gaussian_heatmap with
    its default colormap="jet" (the only mode used for training data).
    """
    if center is None:
        return np.broadcast_to(_JET_LUT[0], (res, res, 3)).copy()
    cx, cy = float(center[0]), float(center[1])
    ys = np.arange(res, dtype=np.float32)[:, None]
    xs = np.arange(res, dtype=np.float32)[None, :]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma))
    idx = np.round(255.0 * g).astype(np.uint8)
    return _JET_LUT[idx]


def render_attn_heatmap_ir(
    centers_224: Iterable[Optional[Tuple[int, int]]],
) -> np.ndarray:
    """Render the attn-map IR image expected by the attnmap policy.

    Public eval-time entry point (mirrors object_center_ir.
    render_object_center_ir, minus the head frame — the heatmap has no RGB
    underlay). Takes the same target-first center list the dot path uses and
    renders the FIRST entry only, matching the training data (one Gaussian).
    """
    centers = list(centers_224)
    first = centers[0] if centers else None
    return render_gaussian_heatmap(first)
