"""Render the dot IR image from a target center (the paper's dot-IR input).

`render_object_center_ir` draws a red dot with a white outline at the target's
center (224x224 head-image pixels) over the head-camera frame. The constants
below (DOT_RADIUS, OUTLINE_WIDTH, COLOURS) are the ones the training data was
rendered with, so a policy sees identical pixels at train and eval time. Each
center you pass is drawn in the next color of COLOURS; the training data passed
the target alone (for placement tasks, the destination), so pass one center.

Centers come from privileged_ir/object_center_ir_omnigibson.py (oracle) or
from your own predictor; the dataset's meta/ir_centers.jsonl holds the
training centers.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np


# Tuned for 224x224 + SigLIP 16-px patches. Match training data generation.
DOT_RADIUS: int = 4
OUTLINE_WIDTH: int = 1

# Positional palette: index 0 = target (manipulated), index 1 = reference, etc.
# pi05 IR data so far renders only the target, so only COLOURS[0] is exercised
# in training. We expose the full palette for symmetry / future use.
COLOURS: List[str] = ["#FF3333", "#3399FF", "#33FF33", "#FF9900", "#CC33FF"]

# Native vs. policy-input resolution. Helper provided for callers that want to
# project at native resolution and scale here.
NATIVE_RESOLUTION: int = 720
IR_RESOLUTION: int = 224


def scale_center(
    center: Tuple[int, int],
    src_res: int = NATIVE_RESOLUTION,
    dst_res: int = IR_RESOLUTION,
) -> Tuple[int, int]:
    """Scale (cx, cy) from a src_res x src_res grid to dst_res x dst_res.

    Identical to the training-data generator's scale_center.
    """
    s = dst_res / src_res
    return (int(round(center[0] * s)), int(round(center[1] * s)))


def render_centers_on_frame(
    rgb: np.ndarray,
    centers: Iterable[Optional[Tuple[int, int]]],
    dot_radius: int = DOT_RADIUS,
    outline_width: int = OUTLINE_WIDTH,
    colours: Optional[List[str]] = None,
) -> np.ndarray:
    """Draw colored dots at object centers on an RGB frame.

    Byte-identical to the training-data generator's render_centers_on_frame (training-time
    renderer). Accepts either a list of (cx, cy)|None or any iterable of those
    — the training-time version accepts a dict but only uses values in order,
    so an ordered iterable suffices.

    Args:
        rgb: np.uint8[H, W, 3] — RGB image (designed for 224x224).
        centers: iterable of (cx, cy) or None, in target-first order.
        dot_radius: dot radius in pixels (default 4).
        outline_width: white outline thickness (default 1).
        colours: list of hex colors keyed by position; default COLOURS.

    Returns:
        np.uint8[H, W, 3] — annotated copy of the input. `rgb` itself is not
        mutated.
    """
    # Import lazily — PIL is a soft dep we don't want to pull in until first IR
    # render. Matches training-side lazy import in the training-data IR generator.
    from PIL import Image, ImageDraw

    if colours is None:
        colours = COLOURS

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb).convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    for i, center in enumerate(centers):
        if center is None:
            continue
        colour = colours[i % len(colours)]
        cx, cy = int(center[0]), int(center[1])
        r_out = dot_radius + outline_width
        draw.ellipse(
            [cx - r_out, cy - r_out, cx + r_out, cy + r_out],
            fill="white",
        )
        draw.ellipse(
            [cx - dot_radius, cy - dot_radius,
             cx + dot_radius, cy + dot_radius],
            fill=colour,
        )

    return np.asarray(img, dtype=np.uint8)


def render_object_center_ir(
    head_rgb_224: np.ndarray,
    centers_224: Iterable[Optional[Tuple[int, int]]],
    *,
    dot_radius: int = DOT_RADIUS,
    outline_width: int = OUTLINE_WIDTH,
    colours=None,
) -> np.ndarray:
    """Render the IR image expected by `BehaviorIRInputs`.

    This is the public eval-time entry point. Wraps `render_centers_on_frame`
    with shape/dtype validation that matches the training data contract.

    Args:
        head_rgb_224: np.uint8[224, 224, 3] — head camera RGB, already resized
            to the IR_RESOLUTION used during training. If your eval pipeline
            still has the head image at native resolution (e.g. 720), resize
            BEFORE calling this; the dots are drawn in pixel-space coordinates
            and won't survive a post-render resize cleanly.
        centers_224: iterable of (cx, cy) integer pixel coords in 224x224
            space. None entries are skipped (rendered as no dot). To match
            training, the FIRST element should be the target object's center;
            extra entries get COLOURS[1+] in order.

    Returns:
        np.uint8[224, 224, 3] — the IR image. Pass straight into the obs dict
        under key `observation/object_center_image`.
    """
    arr = np.asarray(head_rgb_224)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"head_rgb_224 must be [H,W,3]; got shape {arr.shape}"
        )
    if arr.shape[0] != IR_RESOLUTION or arr.shape[1] != IR_RESOLUTION:
        # Don't silently resize — the training generator renders dots in 224
        # pixel-space and any rescale would warp them. Caller must resize.
        raise ValueError(
            f"head_rgb_224 must be {IR_RESOLUTION}x{IR_RESOLUTION}; got "
            f"{arr.shape[0]}x{arr.shape[1]}. Resize the head camera frame "
            f"BEFORE calling render_object_center_ir."
        )
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return render_centers_on_frame(
        arr,
        list(centers_224),
        dot_radius=dot_radius,
        outline_width=outline_width,
        colours=colours,
    )
