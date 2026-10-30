"""Oracle target centers from live OmniGibson state (used only with --with-ir).

Projects the instructed target's 3D bounding box into the head camera (and,
with --with-wrist-ir, the wrist cameras) and adds the center, in 224x224 image
pixels, to each observation under `policy_extras::object_center_xy` (see
docs/protocol.md). Results that use these keys are privileged.

The projection reproduces the one used to label the training data: the
camera's cameraViewTransform has its first row sign-flipped to reconcile
OmniGibson's OpenGL camera convention with OpenCV pinhole math, and the center
is that of the projected box, clamped to the image. Only the target (for
placement tasks, the destination) is sent, matching training.

API
---
- `IRObsInjector(robot, scope)`: holds the robot and task object scope;
  `update_target_instances([...])` is called per episode and `inject(obs)`
  adds the center keys to an observation in place.
- `wrap_policy_forward(policy, injector)`: makes the BEHAVIOR-1K websocket
  policy call `injector.inject(obs)` before sending each observation.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Sensor key for the R1Pro head camera. Sourced from BEHAVIOR-1K's
# eval_utils.ROBOT_CAMERA_NAMES["R1Pro"]["head"] = "robot_r1::robot_r1:zed_link:Camera:0".
# `robot.sensors[...]` is keyed by the suffix after the first "::".
HEAD_SENSOR_NAME: str = "robot_r1:zed_link:Camera:0"

# Wrist cameras, same source (eval_utils.ROBOT_CAMERA_NAMES["R1Pro"]):
#   left_wrist  = "robot_r1::robot_r1:left_realsense_link:Camera:0"
#   right_wrist = "robot_r1::robot_r1:right_realsense_link:Camera:0"
# NOTE we do NOT hardcode a wrist intrinsic here the way the training-time
# generator had to (WRIST_K = 388.6639 @ 480, because
# the training-data get_camera_params returns a HEAD K for every camera_id).
# At eval the K comes straight off the live sensor (sensor.intrinsic_matrix),
# so it is correct per-camera by construction and stays correct if the eval
# wrapper changes the render resolution (the wrappers set image_width/height
# to 224 for ALL cameras; fx and cx scale together, so the projection is
# resolution-independent).
WRIST_SENSOR_NAMES: dict = {
    "left": "robot_r1:left_realsense_link:Camera:0",
    "right": "robot_r1:right_realsense_link:Camera:0",
}

# Resolution conventions — match training data generation.
NATIVE_RESOLUTION_DEFAULT: int = 720
IR_RESOLUTION: int = 224

# Observation keys of the privileged IR inputs (docs/protocol.md).
OBJECT_CENTER_XY_KEY: str = "policy_extras::object_center_xy"
# Wrist IR centers (--with-wrist-ir). Same wire format as the head
# center: an int32 (N, 2) array in 224-space, or the (-1, -1) sentinel when the
# target is not visible from that wrist. The server renders the heatmaps.
OBJECT_CENTER_LEFT_WRIST_XY_KEY: str = "policy_extras::object_center_left_wrist_xy"
OBJECT_CENTER_RIGHT_WRIST_XY_KEY: str = "policy_extras::object_center_right_wrist_xy"
# The constant frame-0 IR anchor image (head + dot at episode start), sent with
# --with-first-frame.
OBJECT_CENTER_FIRST_IMAGE_KEY: str = "policy_extras::object_center_first_image"
# Concrete referring expression
# for the VLM IR predictor (only shipped when set — i.e. the ir-abstract policy,
# whose own prompt is abstract). The server feeds this to the VLM instead of the
# policy prompt.
VLM_TARGET_DESC_KEY: str = "policy_extras::vlm_target_desc"


# ── Camera ───────────────────────────────────────────────────────────────────

def _get_head_sensor(robot, sensor_name: str = HEAD_SENSOR_NAME):
    """Resolve the head-camera VisionSensor handle from a R1Pro robot.

    Mirrors BEHAVIOR-1K eval.py:312:
        camera = self.robot.sensors[camera_name.split("::")[1]]
    """
    sensors = getattr(robot, "sensors", None)
    if sensors is None:
        raise AttributeError(
            "robot has no `.sensors` attribute; cannot resolve head camera"
        )
    if sensor_name not in sensors:
        available = list(sensors.keys())
        raise KeyError(
            f"head sensor {sensor_name!r} not found on robot. "
            f"Available sensors: {available}"
        )
    return sensors[sensor_name]


def get_head_camera_params(
    robot, sensor_name: str = HEAD_SENSOR_NAME,
) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
    """Read K, view_mat (world→camera), and native image size from OmniGibson.

    Returns
    -------
    (K, view_mat, native_W, native_H) where K and view_mat are 3×3 and 4×4
    numpy float32 arrays, or None if camera_parameters returns zeros (the
    well-known first-render-step gotcha; see eval.py:305-310).
    """
    sensor = _get_head_sensor(robot, sensor_name)

    # ── Intrinsic ────────────────────────────────────────────────────────────
    K_tensor = sensor.intrinsic_matrix  # torch.Tensor (3, 3); see vision_sensor.py:883-896
    K = np.asarray(K_tensor.detach().cpu().numpy(), dtype=np.float32)

    # ── Native render resolution ─────────────────────────────────────────────
    native_W = int(sensor.image_width)
    native_H = int(sensor.image_height)

    # ── Extrinsic (cameraViewTransform from sim) ─────────────────────────────
    # OmniGibson stores cameraViewTransform as a column-major flatten of the
    # world-to-camera matrix — i.e. raw_view.reshape(4,4).T gives W2C in
    # standard row-major form (see BEHAVIOR-1K eval.py:319 where they do
    # `np.linalg.inv(direct_cam_pose.reshape(4,4).T)` to recover C2W).
    #
    # The TRAINING-TIME pipeline (the training-data world_to_image_pinhole)
    # used a different format: their `reconstruct_camera_view_transform_from_rel_pose`
    # produces a ROW-major flatten of W2C, so they could `.reshape(4,4)` directly
    # without `.T`. We must transpose at eval to match the math the training
    # code intended (W2C @ world_homo).
    raw_view = sensor.camera_parameters.get("cameraViewTransform", None)
    if raw_view is None:
        logger.warning(
            "[ir_omnigibson] cameraViewTransform missing on sensor %s", sensor_name,
        )
        return None
    view_mat = np.asarray(raw_view, dtype=np.float32).reshape(4, 4).T
    if np.allclose(view_mat, 0.0):
        # First-render-step zero matrix — see eval.py:305-310 comment. Caller
        # should retry next step (or fall back to no IR detection on this frame).
        return None

    # Sign-flip first row of W2C — flips the camera's X axis. The training-time
    # pinhole math applied this to compensate for the OmniGibson camera frame
    # convention (X axis orientation differs from the standard OpenCV pinhole
    # K assumes). Identical fix needed on live cameraViewTransform.
    view_mat[0, :3] = -view_mat[0, :3]
    view_mat[0, 3] = -view_mat[0, 3]
    return K, view_mat, native_W, native_H


# ── Object AABB → world-frame corners ────────────────────────────────────────

def _quat_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion → 3×3 rotation matrix. Same convention as
    omnigibson.utils.transform_utils.quat2mat (we avoid the torch import to
    keep this module light).
    """
    x, y, z, w = quat_xyzw
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, yy, zz = x * xs, y * ys, z * zs
    xy, xz, yz = x * ys, x * zs, y * zs
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz,         xz + wy        ],
            [xy + wz,         1.0 - (xx + zz), yz - wx        ],
            [xz - wy,         yz + wx,         1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def _to_np(x) -> np.ndarray:
    """Convert a torch tensor or numpy array to numpy.ndarray (float32)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def get_target_world_corners(obj) -> Optional[np.ndarray]:
    """Return (8, 3) world-frame AABB corners for an OmniGibson object, or None.

    Uses `obj.get_base_aligned_bbox()` (object_base.py:376-458), which returns
    `(center_world, orn_world_xyzw, extent, center_local)`. Mirrors the
    training-time `get_rotated_bbox_3d_corners(position, size, R)` formula
   , so the 8 corners produced here are
    byte-equivalent to those projected during training-data generation.
    """
    actual = obj.wrapped_obj if hasattr(obj, "wrapped_obj") else obj
    try:
        center, orn, extent, _ = actual.get_base_aligned_bbox()
    except Exception as e:
        logger.warning("[ir_omnigibson] get_base_aligned_bbox failed: %s", e)
        return None

    center_np = _to_np(center).reshape(3)
    orn_np = _to_np(orn).reshape(4)
    extent_np = _to_np(extent).reshape(3)
    R = _quat_to_rotmat(orn_np)

    w, d, h = extent_np / 2.0
    local = np.array(
        [
            [-w, -d, -h], [+w, -d, -h],
            [-w, +d, -h], [+w, +d, -h],
            [-w, -d, +h], [+w, -d, +h],
            [-w, +d, +h], [+w, +d, +h],
        ],
        dtype=np.float32,
    )
    world = local @ R.T + center_np  # (8, 3)
    return world


# ── Projection ───────────────────────────────────────────────────────────────

def _project_with_depth(
    world_points: np.ndarray, K: np.ndarray, view_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pinhole projection returning (pixels (N,2), camera-space depth z (N,)).

    `z` is the camera-frame z of each point. In the OmniGibson view-transform
    convention used here, IN-FRONT points have z < 0; z >= 0 means on/behind the
    image plane (the divide below then mirrors the point to a *finite, bogus*
    in-frame pixel, so callers must guard on z — `np.isfinite` does NOT catch it).
    """
    N = world_points.shape[0]
    homo = np.concatenate([world_points, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N,4)
    cam = (view_mat @ homo.T).T[:, :3]                                                # (N,3)
    z = cam[:, 2].copy()                                                              # (N,) depth
    img_homo = (K @ cam.T).T                                                          # (N,3)
    pix = img_homo[:, :2] / (img_homo[:, 2:3] + 1e-8)                                 # (N,2)
    return pix, z


def project_world_to_pixel(
    world_points: np.ndarray,
    K: np.ndarray,
    view_mat: np.ndarray,
) -> np.ndarray:
    """Standard pinhole projection. Mirrors the training-data world_to_image_pinhole.

    Returns (N, 2) float32 pixel coords. NOTE: points behind the camera get a
    finite (mirrored) pixel, not NaN/Inf — use `_project_with_depth` + a z>0
    guard to reject them.
    """
    pix, _ = _project_with_depth(world_points, K, view_mat)
    return pix


def _aabb_corners_to_native_bbox(
    corners_2d: np.ndarray, native_W: int, native_H: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Take min/max over the 8 projected corners, clamp to image, drop empty.

    Mirrors _project_bbox_3d_to_2d. Returns
    (x_min, y_min, x_max, y_max) at NATIVE resolution, or None.
    """
    if not np.all(np.isfinite(corners_2d)):
        return None
    x_min_raw = int(np.floor(corners_2d[:, 0].min()))
    y_min_raw = int(np.floor(corners_2d[:, 1].min()))
    x_max_raw = int(np.ceil(corners_2d[:, 0].max()))
    y_max_raw = int(np.ceil(corners_2d[:, 1].max()))
    x_min = max(0, min(x_min_raw, native_W - 1))
    y_min = max(0, min(y_min_raw, native_H - 1))
    x_max = max(0, min(x_max_raw, native_W - 1))
    y_max = max(0, min(y_max_raw, native_H - 1))
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def _bbox_to_center_224(
    bbox_native: Tuple[int, int, int, int], native_W: int, native_H: int,
) -> Tuple[int, int]:
    """((x_min + x_max)//2, ...) at native, then scale to 224.

    Matches the training-data generator's bbox_to_center + scale_center (the training-data IR generator
    :158-167). Assumes the native head image is SQUARE (W == H); if not, we
    scale each axis independently with its own (W or H)→224 ratio. The
    training pipeline uses 720×720 so both ratios are identical there.
    """
    x_min, y_min, x_max, y_max = bbox_native
    cx_native = (x_min + x_max) // 2
    cy_native = (y_min + y_max) // 2
    sx = IR_RESOLUTION / float(native_W)
    sy = IR_RESOLUTION / float(native_H)
    cx_224 = int(round(cx_native * sx))
    cy_224 = int(round(cy_native * sy))
    # Clamp to [0, IR_RESOLUTION) so downstream sentinel-check doesn't drop it.
    cx_224 = max(0, min(IR_RESOLUTION - 1, cx_224))
    cy_224 = max(0, min(IR_RESOLUTION - 1, cy_224))
    return cx_224, cy_224


def _center_px_to_224(
    cx_native: float, cy_native: float, native_W: int, native_H: int,
) -> Tuple[int, int]:
    """Scale a single native-resolution pixel to 224, clamped to [0, 224)."""
    cx = int(round(cx_native * IR_RESOLUTION / float(native_W)))
    cy = int(round(cy_native * IR_RESOLUTION / float(native_H)))
    return max(0, min(IR_RESOLUTION - 1, cx)), max(0, min(IR_RESOLUTION - 1, cy))


_DEBUG_LOGGED_ONCE = False
_WRIST_DEBUG_LOGGED = set()


def compute_wrist_center_xy_224(
    robot, corners_3d: np.ndarray, side: str,
) -> Tuple[Optional[Tuple[int, int]], str]:
    """Project 8 world-frame bbox corners through the LIVE **wrist** camera.

    TRAIN/EVAL CONTRACT — this must mirror the training-time generator
    (the training-data wrist IR generator
    :project_bbox_center_wrist), which differs from the HEAD path above in two
    deliberate ways. Getting either wrong silently shifts the blob:

    1. CENTER, NOT CORNER HULL. The head path takes min/max over the 8
       projected corners and uses the bbox midpoint. On a wrist the arm gets
       close enough that the bbox STRADDLES the image plane (measured at the
       grasp frame: z spanning [-0.208, +0.021]), and the hull then explodes
       (raw x over [-604, -137]) even though the object is plainly in view.
       Those near-grasp frames are the most informative in the episode, so the
       training renderer projects the bbox CENTRE and uses that. Same here.

    2. STRICT OFF-FRUSTUM -> None. The head path clamps to the image bounds;
       for a wrist that would pin the blob to a border for the large fraction
       of every episode where the target is out of view, teaching a confident
       WRONG location. Training returns None (-> uniform "not visible" frame)
       and so does this.

    Depth sign: in this view-transform convention IN-FRONT is z < 0 (see
    _project_with_depth), i.e. depth = -z. The training code carries the same
    note; a `z > 0` test blanks every frame.

    Returns ((cx, cy) | None, reason_tag) with the same tag vocabulary as
    compute_object_center_xy_224.
    """
    try:
        sensor_name = WRIST_SENSOR_NAMES.get(side)
        if sensor_name is None:
            return None, f"bad_side:{side}"
        cam = get_head_camera_params(robot, sensor_name)  # generic: reads any sensor
        if cam is None:
            return None, "no_camera"
        K, view_mat, native_W, native_H = cam

        corners_3d = np.asarray(corners_3d, dtype=np.float32).reshape(8, 3)
        center_3d = corners_3d.mean(axis=0, keepdims=True)  # (1, 3)
        pix, z = _project_with_depth(center_3d, K, view_mat)
        cx_native, cy_native = float(pix[0, 0]), float(pix[0, 1])
        depth = -float(z[0])

        if side not in _WRIST_DEBUG_LOGGED:
            _WRIST_DEBUG_LOGGED.add(side)
            logger.info(
                "[ir_omnigibson] one-shot WRIST(%s) projection debug:\n"
                "  sensor = %s\n  K = %s\n  native = (%d, %d)\n"
                "  center_native = (%.1f, %.1f)  depth(-z) = %.3f",
                side, sensor_name, K.tolist(), native_W, native_H,
                cx_native, cy_native, depth,
            )

        if not np.isfinite([cx_native, cy_native]).all():
            return None, "off_screen"
        if depth <= 1e-6:
            return None, "behind_camera"
        if not (0.0 <= cx_native < native_W and 0.0 <= cy_native < native_H):
            return None, "off_screen"
        return _center_px_to_224(cx_native, cy_native, native_W, native_H), "ok"
    except Exception as e:  # noqa: BLE001
        logger.exception("[ir_omnigibson] wrist(%s) center computation failed", side)
        return None, f"exception:{type(e).__name__}:{e}"


def compute_object_center_xy_224(
    robot, scope: dict, target_name: str,
) -> Tuple[Optional[Tuple[int, int]], str]:
    """End-to-end: live sim state → ((cx, cy) | None, reason_tag).

    `reason_tag` is one of:
        "ok"            — center successfully computed
        "no_camera"     — cameraViewTransform was zero / unreadable
        "not_in_scope"  — target_name not present in object_scope
        "no_aabb"       — get_base_aligned_bbox failed
        "behind_camera" — target centroid is behind the image plane (no dot)
        "off_screen"    — target centroid projects outside the frame (no dot;
                          NOT clamped to the edge — a border dot misleads policy)
        "exception:<msg>" — unexpected exception (caught)

    Callers should encode (None, _) as the sentinel (-1, -1) before shipping
    over the wire (the server-side adapter maps that to a no-dot render).
    """
    try:
        if target_name not in scope:
            return None, "not_in_scope"
        corners_3d = get_target_world_corners(scope[target_name])
        if corners_3d is None:
            return None, "no_aabb"
        return compute_corners_center_xy_224(robot, corners_3d, tag=target_name)
    except Exception as e:
        logger.exception(
            "[ir_omnigibson] center computation failed for %r", target_name,
        )
        return None, f"exception:{type(e).__name__}:{e}"


def compute_corners_center_xy_224(
    robot, corners_3d: np.ndarray, *, tag: str = "static",
) -> Tuple[Optional[Tuple[int, int]], str]:
    """Project a FIXED set of 8 world-frame bbox corners through the LIVE head
    camera → ((cx, cy) | None, reason_tag).

    This is the shared projection core: compute_object_center_xy_224 feeds it
    the live object's AABB corners (reach: dot tracks the object), while the
    place destination mode feeds it the target's bbox corners FROZEN at the
    demo's place-end frame (`ir_destination_corners_3d` in the instance cfg) —
    matching the training-data generator's destination-center mode, where the
    dot pins the goal location instead of riding the carried object.
    Same reason_tags as compute_object_center_xy_224 (minus scope/aabb ones).
    """
    try:
        cam = get_head_camera_params(robot)
        if cam is None:
            return None, "no_camera"
        K, view_mat, native_W, native_H = cam

        corners_3d = np.asarray(corners_3d, dtype=np.float32).reshape(8, 3)

        # Project the 8 AABB corners AND the AABB centroid, with camera-space
        # depth. The centroid is the stable signal for the visibility guards;
        # the corner bbox gives the training-consistent dot position.
        center_3d = corners_3d.mean(axis=0, keepdims=True)  # (1, 3)
        all_2d, all_z = _project_with_depth(
            np.concatenate([corners_3d, center_3d], axis=0), K, view_mat,
        )
        corners_2d, corners_z = all_2d[:8], all_z[:8]
        center_2d, center_z = all_2d[8], float(all_z[8])

        global _DEBUG_LOGGED_ONCE
        if not _DEBUG_LOGGED_ONCE:
            _DEBUG_LOGGED_ONCE = True
            logger.info(
                "[ir_omnigibson] one-shot projection debug:\n"
                "  K = %s\n  view_mat = %s\n  native = (%d, %d)\n"
                "  target = %r\n  center_2d = %s  center_z = %.3f\n"
                "  corners_2d = %s",
                K.tolist(), view_mat.tolist(), native_W, native_H,
                tag, center_2d.tolist(), center_z, corners_2d.tolist(),
            )

        # ── Visibility guards (off-frame spurious-dot fix) ───────────────────
        # Depth sign: in the OmniGibson cameraViewTransform convention used
        # here, IN-FRONT points have camera-space z < 0 (verified live: visible
        # target centroids project with center_z ≈ -1.0). So z >= 0 ⇒ on/behind
        # the image plane, where the divide-by-z mirrors the centroid to a bogus
        # in-frame pixel — reject on DEPTH (isfinite does NOT catch it).
        if center_z >= 0.0:
            return None, "behind_camera"
        cx_f, cy_f = float(center_2d[0]), float(center_2d[1])
        # If the object straddles the image plane (any corner on/behind the
        # camera, z >= 0), the corner bbox is unreliable → centroid px, but only
        # when the centroid itself is in-frame.
        if float(corners_z.max()) >= 0.0:
            if not (0.0 <= cx_f < native_W and 0.0 <= cy_f < native_H):
                return None, "off_screen"
            return _center_px_to_224(cx_f, cy_f, native_W, native_H), "ok"
        # TRAIN-CONSISTENT partial-visibility handling (fixed 2026-08-08): the
        # training generator (_project_bbox_3d_to_2d) CLAMPS the
        # projected corner bbox to the frame and draws the dot at the clamped
        # bbox center whenever the clamp is non-degenerate — so a PARTIALLY
        # visible object gets a dot at its visible portion. The old behavior here
        # suppressed the dot whenever the CENTROID was off-frame ("a border dot
        # misleads policy"), which was a train/eval mismatch: it blanked the IR
        # on 151/448 held_out instances (reach_plate had fully dotless episodes)
        # for targets the policy was trained to see dotted. off_screen now means
        # what it means in training: the ENTIRE clamped bbox is out of frame.
        bbox_native = _aabb_corners_to_native_bbox(corners_2d, native_W, native_H)
        if bbox_native is not None:
            return _bbox_to_center_224(bbox_native, native_W, native_H), "ok"
        if not (0.0 <= cx_f < native_W and 0.0 <= cy_f < native_H):
            return None, "off_screen"
        return _center_px_to_224(cx_f, cy_f, native_W, native_H), "ok"
    except Exception as e:
        logger.exception(
            "[ir_omnigibson] center computation failed for %r", tag,
        )
        return None, f"exception:{type(e).__name__}:{e}"


# ── Obs injector ─────────────────────────────────────────────────────────────

class IRObsInjector:
    """Adds OBJECT_CENTER_XY_KEY to obs dicts using live OmniGibson state.

    Held by the eval harness; one instance for the whole job. `update_target_instances`
    is called once per subtask (before the rollout loop) to point at the new
    target object.

    Example
    -------
        injector = IRObsInjector(evaluator.robot, evaluator.env.task.object_scope)
        injector.update_target_instances(subtask_cfg["target_instances"])
        wrap_policy_forward(evaluator.policy, injector)
        # ... evaluator.step() runs as usual; obs dicts get the IR center added.
    """

    def __init__(
        self,
        robot,
        scope: dict,
        *,
        sensor_name: str = HEAD_SENSOR_NAME,
        debug_video_dir: Optional[pathlib.Path] = None,
        debug_video_fps: int = 30,
        with_first_frame: bool = False,
    ):
        self.robot = robot
        self.scope = scope
        self.sensor_name = sensor_name
        self._target_instances: List[str] = []
        # DESTINATION mode (place tasks): fixed 8x3 world corners to project
        # instead of the live object AABB. None = live-object (reach) mode.
        self._static_world_corners: Optional[np.ndarray] = None
        # First-frame anchor (5th camera). When on, the rendered frame-0 IR image
        # is captured once per episode and shipped under OBJECT_CENTER_FIRST_IMAGE_KEY
        # every step; the server passes it through to object_center_first_image.
        self.with_first_frame = bool(with_first_frame)
        self._first_frame_image: Optional[np.ndarray] = None
        # Diagnostics
        self.n_calls: int = 0
        self.n_skipped_no_target: int = 0
        self.n_missing_detection: int = 0  # camera not ready / off-screen / bad AABB
        self._failure_reasons: dict = {}  # reason_tag -> count

        # Optional debug-video capture. When set, every inject() also renders
        # the IR image locally (byte-identical to what the server renders) and
        # buffers it; flush_video() writes the buffered frames as an mp4 under
        # this dir. Use to visually confirm the IR dot lands on the target.
        self._debug_video_dir: Optional[pathlib.Path] = (
            pathlib.Path(debug_video_dir) if debug_video_dir is not None else None
        )
        self._debug_video_fps = int(debug_video_fps)
        self._frame_buffer: List[np.ndarray] = []
        self._current_instance_name: Optional[str] = None
        self._vlm_target_desc: Optional[str] = None
        if self._debug_video_dir is not None:
            self._debug_video_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "[ir_omnigibson] debug IR video dir: %s (fps=%d)",
                self._debug_video_dir, self._debug_video_fps,
            )

    def update_target_instances(
        self,
        target_instances: Iterable[str],
        *,
        instance_name: Optional[str] = None,
        static_world_corners: Optional[np.ndarray] = None,
        vlm_target_desc: Optional[str] = None,
    ) -> None:
        """Set the active target list for the current subtask. Index 0 is the
        only one rendered (matches the training-data IR generator).

        `static_world_corners` (8x3 world frame) switches the injector to
        DESTINATION mode: the dot is the projection of these FIXED corners
        (the target's bbox frozen at the demo's place-end frame — see
        `ir_destination_corners_3d` in place instance cfgs) instead of the live
        object AABB. Matches training's place --center-mode destination, where
        the dot pins the goal location, not the carried object. Pass None
        (default) for the live-object mode used by reach tasks.

        If `instance_name` is given and debug video logging is on, any pending
        frames from the previous instance are flushed to disk under that
        name before the buffer resets.
        """
        # Flush any pending video from the previous instance first.
        if self._debug_video_dir is not None and self._frame_buffer:
            self.flush_video()
        self._target_instances = list(target_instances or [])
        self._static_world_corners = (
            np.asarray(static_world_corners, dtype=np.float32).reshape(8, 3)
            if static_world_corners is not None else None
        )
        # New instance → recapture the frame-0 anchor on the next inject().
        self._first_frame_image = None
        # Concrete referring expression for the VLM IR predictor (shipped per
        # step to the server as policy_extras::vlm_target_desc). Only needed when
        # the policy prompt is abstract (ir-abstract); None leaves the VLM to
        # fall back to the policy prompt.
        self._vlm_target_desc = vlm_target_desc
        if instance_name is not None:
            self._current_instance_name = instance_name
        logger.info(
            "[ir_omnigibson] target instances for IR: %s (instance=%r, mode=%s)",
            self._target_instances, self._current_instance_name,
            "DESTINATION(static corners)" if self._static_world_corners is not None
            else "track(live object)",
        )

    def _dump_wrist_pair(
        self, obs: dict, side: str, center: Optional[Tuple[int, int]], dump_dir: str,
    ) -> None:
        """Write a [wrist RGB | wrist heatmap] PNG for visual verification.

        The blob MUST sit on the target as seen in the wrist RGB column. A
        projection that is wrong (bad sign, swapped axis, wrong extrinsic) still
        yields in-frame coordinates and a clean log line — this is the only
        check that catches it. Same review method used on the training data.
        """
        import pathlib

        from PIL import Image

        from privileged_ir.attn_heatmap_ir import render_gaussian_heatmap

        rgb_key = f"robot_r1::robot_r1:{side}_realsense_link:Camera:0::rgb"
        wrist_rgb = obs.get(rgb_key)
        if wrist_rgb is None:
            return
        wrist_rgb = np.asarray(wrist_rgb)[..., :3].astype(np.uint8)
        heat = render_gaussian_heatmap(center)
        if wrist_rgb.shape[:2] != heat.shape[:2]:
            from PIL import Image as _I
            wrist_rgb = np.asarray(
                _I.fromarray(wrist_rgb).resize(
                    (heat.shape[1], heat.shape[0]), _I.BILINEAR
                )
            )
        sep = np.full((heat.shape[0], 2, 3), 128, dtype=np.uint8)
        out = np.concatenate([wrist_rgb, sep, heat], axis=1)
        d = pathlib.Path(dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        tag = "blank" if center is None else f"{center[0]}_{center[1]}"
        Image.fromarray(out).save(d / f"{side}_{self.n_calls:05d}_{tag}.png")

    def _render_debug_frame(
        self, obs: dict, center: Optional[Tuple[int, int]],
    ) -> Optional[np.ndarray]:
        """Render the IR image locally for the debug video — matches what the
        server will produce. Returns None if the head image isn't usable
        (wrong shape / missing); the missing frame is silently dropped from
        the video buffer.
        """
        # Lazy import — keeps PIL out of the import path for non-debug runs.
        from privileged_ir.object_center_ir import (
            IR_RESOLUTION,
            render_object_center_ir,
        )

        # OmniGibson obs key for the head camera RGB.
        HEAD_RGB_KEY = "robot_r1::robot_r1:zed_link:Camera:0::rgb"
        if HEAD_RGB_KEY not in obs:
            return None

        head = obs[HEAD_RGB_KEY]
        if hasattr(head, "detach"):
            head = head.detach().cpu().numpy()
        head = np.asarray(head)
        if head.ndim == 3 and head.shape[-1] == 4:  # drop alpha
            head = head[..., :3]
        if head.dtype != np.uint8:
            if np.issubdtype(head.dtype, np.floating):
                head = (np.clip(head, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                head = head.astype(np.uint8)

        # IR renderer requires 224×224; the server will reject anything else.
        if head.shape[0] != IR_RESOLUTION or head.shape[1] != IR_RESOLUTION:
            # Resize for the debug video so we see what the server will see.
            try:
                import cv2  # type: ignore[import]
                head = cv2.resize(
                    head, (IR_RESOLUTION, IR_RESOLUTION),
                    interpolation=cv2.INTER_LINEAR,
                )
            except Exception:
                logger.exception(
                    "[ir_omnigibson] debug resize failed; dropping frame "
                    "(head shape=%s, want %d×%d)",
                    head.shape, IR_RESOLUTION, IR_RESOLUTION,
                )
                return None

        centers_iter = [center] if center is not None else [None]
        return render_object_center_ir(head, centers_iter)

    def inject(self, obs: dict) -> dict:
        """Add OBJECT_CENTER_XY_KEY to `obs` (in place) and return it.

        Always writes the key — uses (-1, -1) sentinel if no detection — so the
        server always receives the key and can render its IR image.
        """
        self.n_calls += 1
        center_xy: Optional[Tuple[int, int]] = None
        reason = "no_target"
        if self._static_world_corners is not None:
            # DESTINATION mode (place): project the frozen goal bbox corners
            # through the live camera — the dot pins the goal location.
            center_xy, reason = compute_corners_center_xy_224(
                self.robot, self._static_world_corners,
                tag=f"destination:{self._target_instances[0] if self._target_instances else '?'}",
            )
            if center_xy is None:
                self.n_missing_detection += 1
                self._failure_reasons[reason] = self._failure_reasons.get(reason, 0) + 1
                obs[OBJECT_CENTER_XY_KEY] = np.array([[-1, -1]], dtype=np.int32)
            else:
                obs[OBJECT_CENTER_XY_KEY] = np.array([list(center_xy)], dtype=np.int32)
        elif not self._target_instances:
            self.n_skipped_no_target += 1
            obs[OBJECT_CENTER_XY_KEY] = np.array([[-1, -1]], dtype=np.int32)
        else:
            # TIE-AWARE (2026-08-09): _target_instances may be a GT TIE SET
            # (e.g. two same-size bags of chips — the instruction is satisfied
            # by either, and scoring accepts either via gt_targets). The
            # constructed primary can be the OFF-FRAME tie member while its
            # sibling is visible; dotting only [0] left the policy guideless
            # for entire episodes (reach_item 18/20, reach_plate: user report).
            # Try members IN ORDER and dot the first with a computable visible
            # center; the primary still wins whenever it is visible.
            # ALL-TIES (2026-08-09, user decision): every member of the GT tie
            # set that is visible gets a dot — both tied objects are correct
            # targets, so both are marked. NOTE: training IR always contained a
            # SINGLE red dot; multi-dot tie frames are OOD for the policy (all
            # rendered RED, never the blue reference color). center_xy keeps the
            # first visible center for logging/back-compat.
            centers_all = []
            reasons = []
            target = self._target_instances[0]
            for cand in self._target_instances:
                c_xy, c_reason = compute_object_center_xy_224(
                    self.robot, self.scope, cand,
                )
                reasons.append(c_reason)
                if c_xy is not None:
                    centers_all.append(list(c_xy))
            if centers_all:
                center_xy = tuple(centers_all[0])
                reason = "ok" if reasons[0] == "ok" else f"ok_tie:{len(centers_all)}"
            else:
                center_xy, reason = None, reasons[0] if reasons else "no_target"
            if center_xy is None:
                self.n_missing_detection += 1
                self._failure_reasons[reason] = self._failure_reasons.get(reason, 0) + 1
                obs[OBJECT_CENTER_XY_KEY] = np.array([[-1, -1]], dtype=np.int32)
            else:
                obs[OBJECT_CENTER_XY_KEY] = np.array(centers_all, dtype=np.int32)

        # Verbose diagnostics: every call for the first 5, then sample every 60
        # to keep log volume bounded during a long rollout. This is the only
        # signal we have to tell if/why IR detection is failing at eval.
        if self.n_calls <= 5 or self.n_calls % 60 == 0:
            logger.info(
                "[ir_omnigibson] inject #%d target=%r center=%s reason=%s "
                "(missing=%d, no_target=%d, by_reason=%s)",
                self.n_calls,
                self._target_instances[0] if self._target_instances else None,
                center_xy, reason,
                self.n_missing_detection, self.n_skipped_no_target,
                dict(self._failure_reasons),
            )

        # WRIST IR (WITH_WRIST_IR=1): the SAME target, projected into each wrist
        # camera. Shipped every step, always both keys — same "any key the
        # policy requires must be producible" rule the head path learned the
        # hard way; the *_attnmap_wrist policy's input transform HARD-REQUIRES
        # observation/object_center_{left,right}_wrist_image, so a missing key
        # would KeyError on the first infer and wedge every client at warm-up.
        # A (-1,-1) sentinel renders the train-consistent uniform "not visible"
        # frame, which is the CORRECT signal here, not a degraded fallback:
        # measured on the training data, the target is outside the left wrist
        # 58.9% / right wrist 47.9% of all frames.
        if os.environ.get("WITH_WRIST_IR") == "1":
            wrist_corners = self._static_world_corners
            if wrist_corners is None and self._target_instances:
                # Track mode: use the live AABB of the same instance the head
                # path dotted (first target; ties resolved as above).
                try:
                    obj = self.scope.get(self._target_instances[0])
                    wrist_corners = get_target_world_corners(obj) if obj is not None else None
                except Exception:  # noqa: BLE001
                    wrist_corners = None
            for side, key in (
                ("left", OBJECT_CENTER_LEFT_WRIST_XY_KEY),
                ("right", OBJECT_CENTER_RIGHT_WRIST_XY_KEY),
            ):
                w_xy, w_reason = (
                    compute_wrist_center_xy_224(self.robot, wrist_corners, side)
                    if wrist_corners is not None else (None, "no_target")
                )
                obs[key] = np.array(
                    [list(w_xy) if w_xy is not None else [-1, -1]], dtype=np.int32
                )
                if self.n_calls <= 5 or self.n_calls % 60 == 0:
                    logger.info(
                        "[ir_omnigibson] wrist(%s) center=%s reason=%s",
                        side, w_xy, w_reason,
                    )
                # WRIST_IR_DUMP_DIR: the ONLY check that proves the eval wrist
                # blob is GEOMETRICALLY correct rather than merely plausible.
                # Blank-rate statistics cannot distinguish "right projection"
                # from "wrong projection that still lands in-frame" — only
                # seeing the blob sit on the target in the WRIST RGB can. Writes
                # [wrist RGB | wrist heatmap] PNGs every WRIST_IR_DUMP_EVERY
                # steps. Off unless the env var is set (zero cost in real runs).
                dump_dir = os.environ.get("WRIST_IR_DUMP_DIR")
                if dump_dir:
                    every = int(os.environ.get("WRIST_IR_DUMP_EVERY", "40"))
                    if self.n_calls % every == 0:
                        try:
                            self._dump_wrist_pair(obs, side, w_xy, dump_dir)
                        except Exception as e:  # noqa: BLE001
                            logger.warning("[ir_omnigibson] wrist dump failed: %s", e)

        # First-frame anchor: capture the rendered frame-0 IR image once (the
        # first inject with a usable head), then ship that same image every step.
        # The server passes it through to observation/object_center_first_image.
        # IR_STYLE-aware (matches the server's 4th-camera renderer): "dot"
        # (default) anchors head-RGB+dot (object_center_first_image-trained
        # policies); "heatmap" anchors the standalone frame-0 Gaussian jet
        # heatmap (attention-map first-frame policies). The heatmap
        # needs no head image, so it captures on the first inject — a frame-0
        # off-view target yields the constant dark-blue zero-attention field,
        # matching the training convention.
        if self.with_first_frame:
            if self._first_frame_image is None:
                if os.environ.get("IR_STYLE", "dot") == "heatmap":
                    # numpy-only import — safe inside the OmniGibson process.
                    from privileged_ir.attn_heatmap_ir import render_gaussian_heatmap

                    self._first_frame_image = render_gaussian_heatmap(center_xy)
                else:
                    rendered = self._render_debug_frame(obs, center_xy)
                    if rendered is not None:
                        self._first_frame_image = rendered
            if self._first_frame_image is not None:
                obs[OBJECT_CENTER_FIRST_IMAGE_KEY] = self._first_frame_image

        # Concrete VLM target description (ir-abstract policy): ship every step
        # so the server can prompt the VLM with the real object, not the abstract
        # policy prompt. Absent → server falls back to the policy prompt.
        if self._vlm_target_desc:
            obs[VLM_TARGET_DESC_KEY] = self._vlm_target_desc

        if self._debug_video_dir is not None:
            ir_frame = self._render_debug_frame(obs, center_xy)
            if ir_frame is not None:
                self._frame_buffer.append(ir_frame)
        return obs

    def flush_video(self) -> Optional[pathlib.Path]:
        """Write the buffered IR frames to mp4 and clear the buffer.

        Filename: `{debug_video_dir}/{instance_name}_ir.mp4`. If no
        `instance_name` was provided via update_target_instances, falls back
        to `episode_NNN_ir.mp4` with a monotonic counter.

        Returns the written path, or None if there were no frames / no debug
        dir configured. Safe to call repeatedly (no-op when buffer empty).
        """
        if self._debug_video_dir is None or not self._frame_buffer:
            self._frame_buffer.clear()
            return None
        name = self._current_instance_name or f"episode_{self.n_calls:06d}"
        out_path = self._debug_video_dir / f"{name}_ir.mp4"
        try:
            self._encode_mp4(self._frame_buffer, out_path, fps=self._debug_video_fps)
            logger.info(
                "[ir_omnigibson] wrote IR debug video: %s (%d frames)",
                out_path, len(self._frame_buffer),
            )
        except Exception:
            logger.exception(
                "[ir_omnigibson] failed to write IR debug video to %s", out_path,
            )
            out_path = None
        self._frame_buffer.clear()
        return out_path

    @staticmethod
    def _encode_mp4(frames: List[np.ndarray], out_path: pathlib.Path, *, fps: int) -> None:
        """Write a list of uint8 [H,W,3] RGB frames to mp4 via PyAV (h264).

        Uses h264 (not libsvtav1 like the training data) — quicker to encode
        and universally viewable in browsers. This is only the debug video,
        not the policy input.
        """
        import av

        if not frames:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        H, W, _ = frames[0].shape
        container = av.open(str(out_path), mode="w")
        try:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = W
            stream.height = H
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "20", "preset": "veryfast"}
            for f in frames:
                frame = av.VideoFrame.from_ndarray(f, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()


# ── Policy wrapping ──────────────────────────────────────────────────────────

def wrap_policy_forward(policy, injector: IRObsInjector) -> None:
    """Monkey-patch `policy.forward(obs, ...)` to call `injector.inject(obs)`
    first.

    Designed for BEHAVIOR-1K's `WebsocketPolicy` (policies.py:42-76). Idempotent:
    if `policy.forward` is already wrapped, calling again is a no-op.

    Note: the injector mutates `obs` in place; the BEHAVIOR-1K caller has
    already finished using `obs` before .forward() is called, so this is safe.
    """
    if getattr(policy.forward, "_privileged_ir_wrapped", False):
        return
    original = policy.forward

    def forward_with_ir(obs, *args, **kwargs):
        injector.inject(obs)
        return original(obs, *args, **kwargs)

    forward_with_ir._privileged_ir_wrapped = True  # type: ignore[attr-defined]
    forward_with_ir._original_forward = original  # type: ignore[attr-defined]
    policy.forward = forward_with_ir
