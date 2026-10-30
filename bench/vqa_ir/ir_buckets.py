"""Bucket one predicted point against an IR ground-truth record (numpy-only).

The taxonomy mirrors the sim eval (bench/sim_eval/METRICS.md):
  reach            correct (point on the instructed object's mask; projected
                   box where no mask exists) / wrong (on another candidate) /
                   none (background or unparseable)
  place_direction  lift the point onto the counter plane -> correct / wrong
                   (side, or off the reference's surface) / on_ref (<0.17 m
                   from the reference) / none
  place_container  point on the container mask, lifted onto the placement
                   plane -> correct (instructed compartment +/-0.05 m) /
                   wrong (another compartment) / none
Items whose ground truth cannot be answered from the frame are `invalid_gt`
and excluded from rates. Use score_ir.py to score a predictions file.
"""

import json
import re
from pathlib import Path

import numpy as np

import geometry as geo


def _cam(gt_entry):
    view = np.array(gt_entry["camera"]["view_fixed"]).reshape(4, 4)
    K = np.array(gt_entry["camera"]["K"]).reshape(3, 3)
    z_sign = gt_entry["camera"]["ray_z_sign"] or 1.0
    return view, K, z_sign


_MASK_CACHE = {}


def _load_mask(gt):
    """(label array, names) for an instance, or (None, None)."""
    path = gt.get("mask_path")
    if not path or not gt.get("mask_names"):
        return None, None
    if path not in _MASK_CACHE:
        from PIL import Image as _I
        _MASK_CACHE[path] = np.array(_I.open(path))
    return _MASK_CACHE[path], gt["mask_names"]


_HULL_CACHE = {}


def _in_hull(mask_path, label, idx, u, v) -> bool:
    """Is (u, v) inside the filled convex hull of instance `idx`'s mask?"""
    import cv2
    key = (mask_path, idx)
    hull = _HULL_CACHE.get(key)
    if hull is None:
        ys, xs = np.nonzero(label == idx)
        if len(xs) < 3:
            _HULL_CACHE[key] = False
            return False
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        hull = cv2.convexHull(pts)
        _HULL_CACHE[key] = hull
    if hull is False:
        return False
    return cv2.pointPolygonTest(hull, (float(u), float(v)), False) >= 0


def _dist_to_mask(label, idx, u, v, cap=80):
    """Approx distance in px from (u,v) to the nearest pixel of label==idx."""
    h, w = label.shape
    ui, vi = int(round(u)), int(round(v))
    lo_v, hi_v = max(0, vi - cap), min(h, vi + cap + 1)
    lo_u, hi_u = max(0, ui - cap), min(w, ui + cap + 1)
    sub = label[lo_v:hi_v, lo_u:hi_u]
    ys, xs = np.nonzero(sub == idx)
    if len(ys) == 0:
        return float("inf")
    return float(np.min(np.hypot(xs + lo_u - u, ys + lo_v - v)))


_ORDINAL_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)_")


def _ordinal_rank(relation) -> int:
    """Rank N of an `Nth_largest`-style relation; 1 for `largest`/`closest`/..."""
    m = _ORDINAL_RE.match(str(relation or ""))
    return int(m.group(1)) if m else 1


def eval_reach(gt, res) -> dict:
    """Dot-in-target test.

    Uses the exact segmentation mask when the demo has a seg stream; falls back
    to the projected 3D AABB otherwise (paperback). The masks matter: projected
    envelopes of neighbouring objects overlap, so a dot on the correct object
    could be charged as a hit on a distractor.
    """
    out = {"bucket": "none", "sub": None}

    # Tie acceptance: a dot on ANY instance tied with the target (e.g. two
    # identical half-apples both "the smallest") counts as correct.
    accept = set(gt.get("accept_instances") or [])

    label, mask_names = _load_mask(gt)

    # GT validity is a property of the frame, not of the prediction, so it is
    # settled before parse handling -- otherwise an unanswerable item lands in
    # `none` for a model that failed to parse and in `invalid_gt` for one that
    # did, and the excluded set stops being the same across models.
    if label is not None:
        if gt["gt_target"] not in mask_names:
            return {"bucket": "invalid_gt", "sub": "target_not_in_mask",
                    "gt_source": "seg_mask"}
        # An Nth-ordinal instruction ("reach the 2nd largest plate") is only
        # answerable from the image when every ranked candidate is visible: a
        # hidden one shifts the ordinal, so the correct on-screen pick differs
        # from gt and the model is charged for a frame it cannot solve. Rank-1
        # relations stay valid -- gt naming a visible target means nothing
        # hidden outranks it.
        # `gt_reranked_visible` means the ground-truth build already re-ranked this record
        # over the candidates visible in the frame, so gt answers what the image
        # shows and the leftover hidden names in `candidates` are harmless.
        if (_ordinal_rank(gt.get("spatial_relation")) >= 2
                and not gt.get("gt_reranked_visible")):
            hidden = sorted(set(gt.get("candidates") or []) - set(mask_names))
            if hidden:
                return {"bucket": "invalid_gt",
                        "sub": "ranked_candidate_not_in_mask:" + ",".join(hidden),
                        "gt_source": "seg_mask"}

    if not res["parse_ok"]:
        out["sub"] = "parse_fail"
        return out
    u, v = res["x_px"], res["y_px"]

    if label is not None:
        out["gt_source"] = "seg_mask"
        h, w = label.shape
        ui, vi = int(round(u)), int(round(v))
        hit_idx = int(label[vi, ui]) if (0 <= ui < w and 0 <= vi < h) else 0
        tgt_idx = mask_names.index(gt["gt_target"]) + 1
        accept_idx = {mask_names.index(n) + 1 for n in ({gt["gt_target"]} | accept)
                      if n in mask_names}
        hit_name = mask_names[hit_idx - 1] if hit_idx > 0 else None
        if hit_idx in accept_idx:
            out["bucket"] = "correct"
        elif hit_idx > 0:
            out["bucket"], out["sub"] = "wrong", f"hit:{hit_name}"
        else:
            # Background by exact mask — try the FILLED CONVEX HULL of each
            # candidate. Objects resting ON the target occlude part of it (pizza
            # on the pizza plate, toys in the storage box): those pixels belong
            # to the target region but aren't in its visible seg mask. Exact
            # masks take precedence (tested above), so hulls only ever claim
            # background pixels — attribution between candidates stays exact.
            hull_hit = None
            for idx in sorted(accept_idx):
                if _in_hull(gt["mask_path"], label, idx, u, v):
                    hull_hit = ("correct", None)
                    break
            if hull_hit is None:
                for i, n in enumerate(mask_names):
                    if (i + 1) in accept_idx:
                        continue
                    if _in_hull(gt["mask_path"], label, i + 1, u, v):
                        hull_hit = ("wrong", f"hull:{n}")
                        break
            if hull_hit:
                out["bucket"], out["sub"] = hull_hit[0], hull_hit[1] or "hull"
            else:
                out["sub"] = "background"
        out["dist_to_box_px"] = min(
            [_dist_to_mask(label, i + 1, u, v)
             for i, n in enumerate(mask_names) if n in ({gt["gt_target"]} | accept)]
            or [float("inf")])
        d_other = [
            _dist_to_mask(label, i + 1, u, v)
            for i, n in enumerate(mask_names) if n not in ({gt["gt_target"]} | accept)
        ]
        out["dist_to_distractor_box_px"] = min(d_other) if d_other else float("inf")
        # Distances above are to the VISIBLE mask, but a hull hit means the dot
        # is inside the object's filled region while the visible pixels sit far
        # away (pizza covers the plate, so the plate mask is a thin ring). Left
        # as-is the tolerant read demotes a strictly-correct dot to "none" —
        # tolerant must stay a superset of strict.
        if out["bucket"] == "correct":
            out["dist_to_box_px"] = 0.0
        elif out["bucket"] == "wrong":
            out["dist_to_distractor_box_px"] = 0.0
        bb = (gt.get("bboxes_2d_seg") or {}).get(gt["gt_target"])
        if bb:
            out["dist_to_target_px"] = float(np.hypot(u - (bb[0]+bb[2])/2,
                                                      v - (bb[1]+bb[3])/2))
        return out

    out["gt_source"] = "projected_bbox"
    gt_bbox = gt["bboxes_2d"].get(gt["gt_target"])
    if gt_bbox is None:
        return {"bucket": "invalid_gt", "sub": "no_target_bbox"}
    if geo.point_in_bbox2d(u, v, gt_bbox):
        out["bucket"] = "correct"
    else:
        hit = None
        for name in gt["candidates"]:
            if name == gt["gt_target"]:
                continue
            if geo.point_in_bbox2d(u, v, gt["bboxes_2d"].get(name)):
                hit = name
                break
        if hit:
            out["bucket"], out["sub"] = "wrong", f"hit:{hit}"
        else:
            out["sub"] = "background"
    cx, cy = (gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2
    out["dist_to_target_px"] = float(np.hypot(u - cx, v - cy))
    # Distance to the box itself (0 inside). Targets here can be ~20 px across,
    # so a strict in-box test conflates "picked the wrong object" with "was two
    # pixels short"; this feeds the tolerant secondary metric.
    out["dist_to_box_px"] = _dist_to_box(u, v, gt_bbox)
    out["dist_to_distractor_box_px"] = min(
        [_dist_to_box(u, v, gt["bboxes_2d"].get(n))
         for n in gt["candidates"] if n != gt["gt_target"]
         and gt["bboxes_2d"].get(n) is not None] or [float("inf")])
    return out


def _dist_to_box(u, v, box) -> float:
    """Euclidean distance from a point to a rectangle (0 if inside)."""
    if box is None:
        return float("inf")
    dx = max(box[0] - u, 0.0, u - box[2])
    dy = max(box[1] - v, 0.0, v - box[3])
    return float(np.hypot(dx, dy))


def tolerant_bucket(item: dict, tol_px: float) -> str:
    """Reach bucket allowing a tol_px slack around the boxes."""
    if item["bucket"] in ("invalid_gt",) or item.get("sub") == "parse_fail":
        return item["bucket"] if item["bucket"] == "invalid_gt" else "none"
    if item.get("dist_to_box_px", float("inf")) <= tol_px:
        return "correct"
    if item.get("dist_to_distractor_box_px", float("inf")) <= tol_px:
        return "wrong"
    return "none"


def eval_carving(gt, res) -> dict:
    out = {"bucket": "none", "sub": None}
    if not res["parse_ok"]:
        out["sub"] = "parse_fail"
        return out
    view, K, z_sign = _cam(gt)
    o, d = geo.pixel_ray(view, K, res["x_px"], res["y_px"], z_sign)

    surface = gt.get("surface")
    ref = np.array(gt["ref_center_3d"])
    plane_z = surface["top_z"] if surface else float(ref[2])
    p = geo.ray_plane_z(o, d, plane_z)
    if p is None:
        out["sub"] = "ray_miss_plane"
        return out
    out["lifted_point"] = p.tolist()

    horiz = float(np.hypot(p[0] - ref[0], p[1] - ref[1]))
    out["horiz_to_ref_m"] = horiz
    fwd, right = geo.robot_forward_right(gt["robot"]["ori_cos"], gt["robot"]["ori_sin"])
    pr = geo.proj_right(p, ref, right)
    out["proj_right"] = pr

    if surface and not geo.point_in_footprint(
            p[:2], surface["footprint_lo"], surface["footprint_hi"], margin=0.05):
        out["bucket"], out["sub"] = "wrong", "off_counter"
        return out
    if horiz < geo.ON_REF_HORIZ_M:
        out["bucket"], out["sub"] = "on_ref", "near_ref"
        return out
    expected_right = gt["spatial_relation"] == "right"
    actual_right = pr > 0
    out["bucket"] = "correct" if actual_right == expected_right else "wrong"
    return out


def eval_container(gt, res) -> dict:
    """Dot-in-container-layer test (fridge, cabinet, ...).

    In-container is gated on the container SEGMENTATION mask (exact 2D), not the
    projected 3D AABB — the appliance is large and often has its door open, so
    the AABB badly under-covers it and rejects dots that plainly land on it.
    Only the layer (z) check needs the 3D lift, which uses the per-container
    placement-depth plane and the per-task derived compartment bands (both
    stored in the GT entry).
    """
    out = {"bucket": "none", "sub": None}
    if not res["parse_ok"]:
        out["sub"] = "parse_fail"
        return out
    u, v = res["x_px"], res["y_px"]

    # ── in-container gate: seg mask if available, else projected AABB ───────
    label, _names = _load_mask(gt)
    ref_idx = gt.get("reference_mask_idx")
    # Constructed goal frames carry an OmniGibson camera (cameraViewTransform +
    # cameraProjection); use its exact ray. Demo frames use the reconstructed
    # pinhole (view_fixed + K).
    com = gt.get("camera_omni")
    if com is not None:
        o, d = geo.omni_pixel_ray(com["V"], com["P"], u, v,
                                  res=gt.get("image_size", [720])[0])
    else:
        view, K, z_sign = _cam(gt)
        o, d = geo.pixel_ray(view, K, u, v, z_sign)
    aabb = gt.get("container_aabb") or gt.get("fridge_aabb")
    lo, hi = [np.array(x) for x in aabb]

    if label is not None and ref_idx is not None:
        out["gt_source"] = "seg_mask"
        h, w = label.shape
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h and int(label[vi, ui]) == ref_idx):
            out["sub"] = "not_on_container"
            return out
    else:
        out["gt_source"] = "projected_bbox"
        if geo.ray_aabb_entry(o, d, lo, hi) is None:
            out["sub"] = "not_in_container"
            return out

    # ── layer (z) via the placement-depth plane lift (validated by oracle) ──
    depth = gt.get("placement_depth", geo.PLACEMENT_DEPTH_M)
    p = geo.ray_plane_y(o, d, float(hi[1]) - depth)
    if p is None:
        out["sub"] = "no_placement_plane_hit"
        return out
    out["lifted_point"] = p.tolist()
    z = float(p[2])
    out["lift_z"] = z
    z0, z1 = gt["compartment_band"]
    if z0 - geo.LAYER_BAND_TOL <= z <= z1 + geo.LAYER_BAND_TOL:
        out["bucket"] = "correct"
    else:
        bands = gt.get("all_compartment_bands") or {
            str(k): v for k, v in geo.COMPARTMENT_BANDS.items()}
        hit = next((L for L, (a, b) in bands.items() if a <= z <= b), None)
        out["bucket"], out["sub"] = "wrong", f"in_layer:{hit}"
    return out


# Family -> scorer. eval_carving handles place_direction; eval_container handles
# place_container. The registry maps every task via its family (_FAMILY above).
_FAMILY = {
    **{t: "reach" for t in (
        "reach_food_item", "reach_dice", "reach_paperback_book", "reach_item",
        "reach_pizza_plate", "reach_plate", "reach_soda_can", "reach_storage_box",
        "reach_tool", "reach_toys", "reach_hardback_from_bookcase")},
    "place_carving_knife": "place_direction",
    "place_tupperware_in_fridge": "place_container",
    "place_bowl_in_cabinet": "place_container",
}

_FAMILY_EVAL = {
    "reach": eval_reach,
    "place_direction": eval_carving,
    "place_container": eval_container,
}
EVALUATORS = {t: _FAMILY_EVAL[f] for t, f in _FAMILY.items()}


def score_item(task, gt, res):
    """Pick the scorer for one item. A GT entry may override its family scorer
    via `score_as` (bowl goal frames are scored dot-on-bowl = eval_reach)."""
    override = gt.get("score_as")
    scorer = _FAMILY_EVAL.get(override) if override else EVALUATORS[task]
    return scorer(gt, res)


def draw_viz(gt, res, item, viz_path: Path):
    from PIL import Image, ImageDraw
    img = Image.open(gt["image_path"]).convert("RGB")

    # Place tasks: the GT is a REGION (correct side / layer), not the in-hand
    # object — shade the pixels that would score `correct` (green) instead of
    # boxing the held object, which misleads.
    task = gt.get("vla_task", "")
    if gt.get("mode") == "destination" and not gt.get("score_as"):
        arr = np.array(img).astype(float)
        stride = 14
        h_, w_ = arr.shape[:2]
        for vv in range(0, h_, stride):
            for uu in range(0, w_, stride):
                b = score_item(task, gt, {"parse_ok": True,
                                          "x_px": float(uu), "y_px": float(vv)})["bucket"]
                if b == "correct":
                    arr[vv:vv+stride, uu:uu+stride] = (
                        0.6 * arr[vv:vv+stride, uu:uu+stride] + 0.4 * np.array([0, 255, 0]))
        img = Image.fromarray(arr.astype("uint8"))
    dr = ImageDraw.Draw(img)
    if gt.get("mode") != "destination":
        for name, bb in (gt.get("bboxes_2d_seg") or gt.get("bboxes_2d") or {}).items():
            if bb is None:
                continue
            color = "lime" if name == gt["gt_target"] else "yellow"
            dr.rectangle(bb, outline=color, width=3)
            dr.text((bb[0] + 2, bb[1] + 2), name, fill=color)
    if res["parse_ok"]:
        u, v = res["x_px"], res["y_px"]
        r = 7
        dr.ellipse([u - r, v - r, u + r, v + r], fill="red", outline="white", width=2)
    dr.text((6, 6), f"{item['bucket']}" + (f" ({item['sub']})" if item["sub"] else ""),
            fill="red")
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(viz_path, quality=90)


