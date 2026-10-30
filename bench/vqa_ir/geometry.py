"""Camera + scene geometry for the dot-IR eval. numpy-only (no torch/sim).

Conventions mirror the training-data projection (world_to_image_pinhole):
the stored 4x4 cameraViewTransform gets its first row (rotation AND
translation) sign-flipped before use, then pixels = K @ (M @ X_world) with a
divide by the camera-frame z. Unprojection inverts exactly that chain. The
per-camera z sign of "in front" is calibrated in build_gt from a known-visible
3D point and stored as ray_z_sign.

Direction / surface predicates re-implement the sim-eval spec
(bench/sim_eval/goal_checker.py + summarize_eval_results.py + METRICS.md);
the sim harness is deliberately NOT imported.
"""

import numpy as np

# summarize_eval_results.py: _ON_REF_HORIZ_M — carving "on_ref" vs correct/wrong
ON_REF_HORIZ_M = 0.17
# goal_checker.py check_in_layer: LAYER_BAND_TOL default
LAYER_BAND_TOL = 0.05

# Fridge (fridge_petcxr_0) compartment z-bands per layer, verified against the
# sim_eval held_out in_layer goal conditions (band_kind="compartment").
# NOTE: for the general (multi-container) path, compartment bands are derived
# per task from the demo layer bands via compartments_from_demo_bands(), which
# reproduces these fridge numbers exactly at the interior boundaries.
COMPARTMENT_BANDS = {
    4: (0.8635, 1.2088),
    5: (1.2088, 1.5541),
    6: (1.5541, 1.8994),
}

# Depth (m) behind a container's front face at which demos actually rest the
# placed object. Measured for the fridge over 42 demos (mean 0.275, sd 0.019).
# The dot must be lifted at this plane, NOT the front face: the head camera
# looks ~25 deg down, so a front-face entry point sits ~0.15 m too high and can
# read a full layer off. For containers without a measured value, ~0.29x the
# container depth reproduces the fridge ratio (0.275 / 0.935).
PLACEMENT_DEPTH_M = 0.275  # fridge default (kept for back-compat)
PLACEMENT_DEPTH_FRAC = 0.29


def compartments_from_demo_bands(layer_bands: dict) -> dict:
    """Contiguous compartment bands {layer: (z_lo, z_hi)} from per-layer demo
    bands {layer: (lo, hi)}. Interior boundaries are the midpoints between
    adjacent demo bands; the outer ends extend by one band width. Reproduces
    the fridge sim-eval compartment boundaries (1.2088, 1.5541) exactly.
    """
    layers = sorted(layer_bands)
    out = {}
    for i, L in enumerate(layers):
        lo, hi = layer_bands[L]
        b_lo = (layer_bands[layers[i - 1]][1] + lo) / 2 if i > 0 else lo - (hi - lo)
        b_hi = (hi + layer_bands[layers[i + 1]][0]) / 2 if i < len(layers) - 1 else hi + (hi - lo)
        out[L] = (b_lo, b_hi)
    return out


# ── camera ───────────────────────────────────────────────────────────────────

def fix_view_matrix(view_flat) -> np.ndarray:
    """4x4 world→camera matrix with the row-0 sign fix applied."""
    view = np.array(view_flat, dtype=np.float64).reshape(4, 4).copy()
    view[0, :3] = -view[0, :3]
    view[0, 3] = -view[0, 3]
    return view


def camera_center(view_fixed: np.ndarray) -> np.ndarray:
    R, t = view_fixed[:3, :3], view_fixed[:3, 3]
    return -np.linalg.inv(R) @ t


def project_point(view_fixed: np.ndarray, K: np.ndarray, p_world) -> tuple:
    """Return ((u, v), cam_z). cam_z's sign tells front/back for this rig."""
    cam = view_fixed[:3, :3] @ np.asarray(p_world, dtype=np.float64) + view_fixed[:3, 3]
    uvw = np.asarray(K, dtype=np.float64) @ cam
    return (uvw[0] / (uvw[2] + 1e-12), uvw[1] / (uvw[2] + 1e-12)), cam[2]


def pixel_ray(view_fixed: np.ndarray, K: np.ndarray, u: float, v: float,
              z_sign: float) -> tuple:
    """Return (origin, unit direction) of the world ray through pixel (u, v)."""
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d_cam = np.array([(u - cx) / fx * z_sign, (v - cy) / fy * z_sign, z_sign])
    R = view_fixed[:3, :3]
    d_world = np.linalg.inv(R) @ d_cam
    d_world = d_world / (np.linalg.norm(d_world) + 1e-12)
    return camera_center(view_fixed), d_world


def omni_pixel_ray(V, P, u, v, res=720):
    """World-space (origin, unit direction) for pixel (u, v) from an OmniGibson
    VisionSensor's cameraViewTransform V and cameraProjection P (both 4x4, the
    row-vector/column-major convention: world->cam = V.T, clip = P.T @ cam).
    Validated 0.0mm ray-to-point on the constructed goal frames.
    """
    V = np.asarray(V, dtype=np.float64).reshape(4, 4)
    P = np.asarray(P, dtype=np.float64).reshape(4, 4)
    M = V.T
    R, t = M[:3, :3], M[:3, 3]
    Rinv = np.linalg.inv(R)
    center = -Rinv @ t
    ndc_x = (u / res) * 2 - 1
    ndc_y = 1 - (v / res) * 2
    cam_dir = np.array([ndc_x / P.T[0, 0], ndc_y / P.T[1, 1], -1.0])
    d = Rinv @ cam_dir
    d = d / (np.linalg.norm(d) + 1e-12)
    return center, d


def omni_project(V, P, p, res=720):
    """Project world point p to pixel (u, v) for an OmniGibson camera (V, P)."""
    V = np.asarray(V, dtype=np.float64).reshape(4, 4)
    P = np.asarray(P, dtype=np.float64).reshape(4, 4)
    clip = P.T @ (V.T @ np.array([p[0], p[1], p[2], 1.0]))
    ndc = clip[:3] / clip[3]
    return (ndc[0] * 0.5 + 0.5) * res, (1 - (ndc[1] * 0.5 + 0.5)) * res


def ray_point_distance(origin, direction, point) -> float:
    delta = np.asarray(point, dtype=np.float64) - origin
    return float(np.linalg.norm(delta - np.dot(delta, direction) * direction))


# ── intersections ────────────────────────────────────────────────────────────

def ray_plane_z(origin, direction, z: float):
    """Intersect ray with horizontal plane at height z. None if behind/parallel."""
    if abs(direction[2]) < 1e-9:
        return None
    t = (z - origin[2]) / direction[2]
    if t <= 0:
        return None
    return origin + t * direction


def ray_plane_y(origin, direction, y: float):
    """Intersect ray with the vertical plane at depth y. None if behind/parallel."""
    if abs(direction[1]) < 1e-9:
        return None
    t = (y - origin[1]) / direction[1]
    if t <= 0:
        return None
    return origin + t * direction


def ray_aabb_entry(origin, direction, lo, hi):
    """Slab-method first intersection point of ray with AABB, or None."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    t_near, t_far = -np.inf, np.inf
    for ax in range(3):
        if abs(direction[ax]) < 1e-12:
            if origin[ax] < lo[ax] or origin[ax] > hi[ax]:
                return None
            continue
        t1 = (lo[ax] - origin[ax]) / direction[ax]
        t2 = (hi[ax] - origin[ax]) / direction[ax]
        t1, t2 = min(t1, t2), max(t1, t2)
        t_near, t_far = max(t_near, t1), min(t_far, t2)
    if t_near > t_far or t_far <= 0:
        return None
    t_hit = t_near if t_near > 0 else 0.0
    return origin + t_hit * direction


# ── boxes ────────────────────────────────────────────────────────────────────

def rotated_box_envelope(center, size, yaw: float) -> tuple:
    """World-aligned AABB (lo, hi) of a yaw-rotated box (center, w/d/h size)."""
    center = np.asarray(center, dtype=np.float64)
    w, d, h = [float(s) for s in size]
    c, s = np.cos(yaw), np.sin(yaw)
    half_x = abs(c) * w / 2 + abs(s) * d / 2
    half_y = abs(s) * w / 2 + abs(c) * d / 2
    lo = center - np.array([half_x, half_y, h / 2])
    hi = center + np.array([half_x, half_y, h / 2])
    return lo, hi


def point_in_footprint(p_xy, lo, hi, margin: float = 0.0) -> bool:
    return (lo[0] - margin <= p_xy[0] <= hi[0] + margin
            and lo[1] - margin <= p_xy[1] <= hi[1] + margin)


def point_in_bbox2d(u: float, v: float, bbox) -> bool:
    return bbox is not None and bbox[0] <= u <= bbox[2] and bbox[1] <= v <= bbox[3]


# ── robot-frame direction (goal_checker.py get_robot_forward_right /
#    check_direction, re-implemented) ─────────────────────────────────────────

def robot_forward_right(ori_cos: float, ori_sin: float) -> tuple:
    """2D forward/right unit vectors from the base yaw cos/sin.
    right = forward rotated -90 deg: [fy, -fx]."""
    fwd = np.array([ori_cos, ori_sin], dtype=np.float64)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    right = np.array([fwd[1], -fwd[0]])
    return fwd, right


def proj_right(p_xy, ref_xy, right) -> float:
    delta = np.asarray(p_xy[:2], dtype=np.float64) - np.asarray(ref_xy[:2], dtype=np.float64)
    return float(np.dot(delta, right))
