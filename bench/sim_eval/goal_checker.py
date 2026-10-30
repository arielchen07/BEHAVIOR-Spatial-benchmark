"""Spatial goal checker for subtask evaluation.

Checks conditions like `ontop`, `left_of`, `right_of`,
`in_front_of`, `behind`, `grasped` at the end of a rollout using the sim's
object positions + the robot's egocentric frame.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ── geometry helpers ─────────────────────────────────────────────────────

def get_robot_forward_right(robot):
    """Return the robot's 2D forward + right unit vectors in the world frame."""
    pos, quat = robot.get_position_orientation()
    pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.array(pos)
    quat = quat.cpu().numpy() if hasattr(quat, "cpu") else np.array(quat)
    r = Rotation.from_quat(quat)  # scipy [x, y, z, w]
    forward_3d = r.apply([1, 0, 0])
    forward_2d = forward_3d[:2]
    forward_2d = forward_2d / (np.linalg.norm(forward_2d) + 1e-8)
    # Right = rotate forward -90 deg in 2D.
    right_2d = np.array([forward_2d[1], -forward_2d[0]])
    return forward_2d, right_2d


def get_object_pos(obj) -> np.ndarray:
    pos, _ = obj.get_position_orientation()
    return pos.cpu().numpy() if hasattr(pos, "cpu") else np.array(pos)


def check_direction(obj_pos, ref_pos, robot_forward, robot_right, direction: str) -> bool:
    delta = np.array([obj_pos[0] - ref_pos[0], obj_pos[1] - ref_pos[1]])
    proj_forward = float(np.dot(delta, robot_forward))
    proj_right = float(np.dot(delta, robot_right))
    if direction == "left_of":
        return proj_right < 0
    if direction == "right_of":
        return proj_right > 0
    if direction == "in_front_of":
        return proj_forward > 0
    if direction == "behind":
        return proj_forward < 0
    raise ValueError(f"Unknown direction: {direction}")


# ── nav-success tracker ──────────────────────────────────────────────────

class NavSuccessTracker:
    """Per-step "did the robot REACH the goal room?" tracker for move_to tasks.

    Success (recorded at ANY step — robust to the policy overshooting/wandering
    after arrival, which a final-frame-only check misses) is either:
      (a) the base maps INTO a goal room (exact seg-map membership), OR
      (b) the base comes within ``door_tol`` of an opened goal DOOR.
    We anchor on the DOOR, not the room mask: the segmentation mask is the room
    INTERIOR (for the garage it starts ~0.85 m past the door) and its whole
    entrance edge is wall, so distance-to-mask mislabels both door-arrivals
    (false fail) and outside-wall grazes (false pass); the door object is the
    actual gate. Also records an audit trail (closest door approach, first
    reached step/pose, downsampled base trajectory) for offline re-scoring.

    Usage: call ``update(base_xy, step)`` each rollout step, then read
    ``entered_rooms`` (consumed by SubtaskGoalChecker.check_in_room) and
    ``to_metrics(...)`` for the saved nav block.
    """

    def __init__(self, goal_rooms, goal_door_xys=None, seg_map=None,
                 door_tol=1.3, traj_every=5):
        self.goal_rooms = set(goal_rooms or [])
        self.goal_doors = [list(d) for d in (goal_door_xys or [])]
        self.seg_map = seg_map
        self.door_tol = float(door_tol)
        self.traj_every = max(1, int(traj_every))
        self.entered_rooms: set = set()
        self.min_door_dist = float("inf")
        self.reached_step = None
        self.reached_xy = None
        self.trajectory: list = []
        self._i = 0

    @property
    def active(self) -> bool:
        """True iff this is a nav task (an in_room goal is present)."""
        return bool(self.goal_rooms)

    def update(self, base_xy, step) -> None:
        """base_xy: world (x, y) of the robot base (torch tensor or array)."""
        if not self.goal_rooms:
            return
        try:
            bx, by = float(base_xy[0]), float(base_xy[1])
        except Exception:
            return
        if self._i % self.traj_every == 0:
            self.trajectory.append([round(bx, 3), round(by, 3)])
        self._i += 1
        cur_door = float("inf")
        for dx, dy in self.goal_doors:
            cur_door = min(cur_door, ((bx - dx) ** 2 + (by - dy) ** 2) ** 0.5)
        self.min_door_dist = min(self.min_door_dist, cur_door)
        if self.entered_rooms == self.goal_rooms:
            return  # already satisfied; keep accumulating the trajectory only
        hit = False
        if self.seg_map is not None:
            try:
                rm = self.seg_map.get_room_instance_by_point(base_xy)
            except Exception:
                rm = None
            if rm in self.goal_rooms:
                self.entered_rooms.add(rm)
                hit = True
        if not hit and cur_door <= self.door_tol:
            self.entered_rooms |= self.goal_rooms
            hit = True
        if hit and self.reached_step is None:
            self.reached_step = int(step)
            self.reached_xy = [round(bx, 3), round(by, 3)]

    def to_metrics(self, final_room=None, final_base_xyz=None) -> dict:
        """The ``metrics['nav']`` block: outcome + door audit + trajectory."""
        return {
            "final_room": final_room,
            "final_base_xyz": final_base_xyz,
            "reached_goal_rooms": sorted(self.entered_rooms),
            "entered_goal_room": bool(self.entered_rooms),
            "goal_rooms": sorted(self.goal_rooms),
            "goal_door_xys": self.goal_doors,
            "door_tol_m": self.door_tol,
            "min_dist_to_goal_door_m": (round(self.min_door_dist, 3)
                                        if self.min_door_dist != float("inf") else None),
            "reached_step": self.reached_step,
            "reached_xy": self.reached_xy,
            "base_trajectory": self.trajectory,
        }


def xy_dist_to_aabb(pxy, lo, hi) -> float:
    """Planar distance from a point to an axis-aligned box footprint (0 inside).

    Centroid distance mis-scores LARGE targets. The diner bar is 2.28 x 5.76 m,
    so a robot standing at one end of the counter reads 3.43 m from its centroid
    and fails a 1.5 m centroid tolerance despite being right at the bar. Measuring
    to the FOOTPRINT lets a tight tolerance (0.8 m) accept the whole object on
    every side, which is what "move to the bar" actually means.
    """
    import numpy as _np
    lo = _np.asarray(lo, dtype=float)
    hi = _np.asarray(hi, dtype=float)
    dx = max(lo[0] - pxy[0], 0.0, pxy[0] - hi[0])
    dy = max(lo[1] - pxy[1], 0.0, pxy[1] - hi[1])
    return float(_np.hypot(dx, dy))


def obj_xy_aabb(obj, links=None):
    """-> ((lo_x, lo_y), (hi_x, hi_y)) for an OmniGibson object, or None.

    `links`: restrict the box to these link names (union of their AABBs) instead
    of the whole entity. `obj.aabb` spans EVERY link, so an articulated part that
    is currently swung open inflates the footprint and, with distance_mode
    "footprint", silently enlarges the success region.

    Concretely: the diner bar has `base_link` (2.28 x 5.76 m counter) plus a
    `door` link (1.21 x 0.54 m flip-up pass-through). Eval opens that flap for
    traversability, which pushed the whole-entity AABB out far enough that the
    train_sanity start scored 0.349 m from the "bar" instead of 0.901 m — i.e.
    inside the 0.8 m tolerance before the policy moved at all. Measuring to
    `base_link` alone keeps the goal on the counter and makes the metric
    independent of whether the flap is open.
    """
    try:
        if links:
            boxes = []
            for ln in links:
                lk = (obj.links or {}).get(ln)
                if lk is None:
                    continue
                l, h = lk.aabb
                l = l.cpu().numpy() if hasattr(l, "cpu") else l
                h = h.cpu().numpy() if hasattr(h, "cpu") else h
                boxes.append((l, h))
            if boxes:
                return ((float(min(b[0][0] for b in boxes)), float(min(b[0][1] for b in boxes))),
                        (float(max(b[1][0] for b in boxes)), float(max(b[1][1] for b in boxes))))
            return None
        lo, hi = obj.aabb
        lo = lo.cpu().numpy() if hasattr(lo, "cpu") else lo
        hi = hi.cpu().numpy() if hasattr(hi, "cpu") else hi
        return (float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1]))
    except Exception:
        return None


class ObjectNavSuccessTracker:
    """Per-step "did the base REACH the target OBJECT?" for move_to-<object> tasks
    (e.g. "Move to the fridge." / "Move to the table in the kitchen.").

    Success (recorded at ANY step, like NavSuccessTracker) = the base comes within
    ``tol_m`` of the target object's xy. The target is fixed furniture; its world
    xy is captured once at load and passed in (located by sim id in the scene
    registry, since it may NOT be a BDDL task object).
    """

    def __init__(self, target_xy, tol_m=1.5, target_name=None, traj_every=5,
                 require_room=None, seg_map=None, target_aabb=None,
                 distance_mode="centroid", alt_targets=None):
        self.tx, self.ty = float(target_xy[0]), float(target_xy[1])
        self.tol = float(tol_m)
        self.target_name = target_name
        self.traj_every = max(1, int(traj_every))
        # Optional SAME-STEP room qualifier: count "reached" only while the base is
        # inside this room instance. Needed when the target sits close to a wall
        # shared with another room (hsf bookcase: corridor cells 0.74 m away through
        # the wall — distance alone can't tell a real approach from a robot parked
        # at the closed door). None → pure-distance behavior (all older configs).
        self.require_room = require_room
        self.seg_map = seg_map
        # distance_mode="footprint" measures to target_aabb instead of the
        # centroid — required for extended targets (see xy_dist_to_aabb).
        # Falls back to centroid if the AABB could not be read.
        self.distance_mode = distance_mode if target_aabb is not None else "centroid"
        self.target_aabb = target_aabb
        # alt_targets: [(name, xy, aabb)] additionally accepted (held_out only,
        # e.g. the OTHER corner booth). reached_candidate records which one.
        self.alt_targets = list(alt_targets or [])
        self.reached_candidate = None
        self.reached = False
        # True when the tolerance was already met on the FIRST update -- i.e. the
        # instance spawned inside the goal region and no navigation was required.
        # Not a success: it measures the test-set spawn, not the policy.
        self.spawn_satisfied = False
        self.min_dist = float("inf")
        self.reached_step = None
        self.reached_xy = None
        self.trajectory: list = []
        self._i = 0

    @property
    def active(self) -> bool:
        return True

    def update(self, base_xy, step) -> None:
        try:
            bx, by = float(base_xy[0]), float(base_xy[1])
        except Exception:
            return
        if self._i % self.traj_every == 0:
            self.trajectory.append([round(bx, 3), round(by, 3)])
        self._i += 1
        if self.distance_mode == "footprint":
            d = xy_dist_to_aabb((bx, by), *self.target_aabb)
        else:
            d = ((bx - self.tx) ** 2 + (by - self.ty) ** 2) ** 0.5
        hit_name = self.target_name
        for _n, _xy, _ab in self.alt_targets:
            _d = (xy_dist_to_aabb((bx, by), *_ab) if (self.distance_mode == "footprint" and _ab)
                  else ((bx - _xy[0]) ** 2 + (by - _xy[1]) ** 2) ** 0.5)
            if _d < d:
                d, hit_name = _d, _n
        self.min_dist = min(self.min_dist, d)
        if self._i == 1 and d <= self.tol:
            self.spawn_satisfied = True
        if not self.reached and d <= self.tol:
            if self.require_room is not None and self.seg_map is not None:
                try:
                    import numpy as _np
                    room = self.seg_map.get_room_instance_by_point(
                        _np.asarray([bx, by], dtype=float))
                except Exception:
                    room = None
                if room != self.require_room:
                    return
            self.reached = True
            self.reached_step = int(step)
            self.reached_xy = [round(bx, 3), round(by, 3)]
            self.reached_candidate = hit_name

    def to_metrics(self, final_room=None, final_base_xyz=None) -> dict:
        return {
            "goal_type": "near_object",
            "distance_mode": self.distance_mode,
            "target_object": self.target_name,
            # which candidate was actually reached — distinguishes a held_out pass
            # via an ALTERNATIVE accepted target (the other corner booth) from the
            # trained one, so a relaxed accept set can't hide route memorisation.
            "reached_candidate": self.reached_candidate,
            "target_xy": [round(self.tx, 3), round(self.ty, 3)],
            "tol_m": self.tol,
            "require_room": self.require_room,
            # A spawn-satisfied instance is NOT a navigation success: the base was
            # already inside tol on the first update, so the episode measures the
            # test-set spawn rather than the policy. Report it as not-reached and
            # flag it, instead of crediting a "success" the robot never earned.
            "reached_object": bool(self.reached) and not self.spawn_satisfied,
            "entered_goal_room": bool(self.reached) and not self.spawn_satisfied,  # alias: existing summaries read success
            "spawn_satisfied": bool(self.spawn_satisfied),
            "reached_object_raw": bool(self.reached),   # pre-guard value, for audit
            "min_dist_to_object_m": (round(self.min_dist, 3) if self.min_dist != float("inf") else None),
            "reached_step": self.reached_step,
            "reached_xy": self.reached_xy,
            "final_room": final_room,
            "final_base_xyz": final_base_xyz,
            "base_trajectory": self.trajectory,
        }


class LayerReachTracker:
    """Per-step SOFT signal for locate (tupperware) tasks: did each same-category
    object EVER pass through the target layer band while INSIDE the fridge during
    the rollout? Paired at analysis time with the FINAL in_layer outcome, this
    separates 'reached the layer but fell / slid out' from 'never got there'.
    Mirrors NavSuccessTracker: construct before the loop, .update(step) each step,
    .to_dict() after. All-best-effort; any failure just leaves active=False.
    """

    def __init__(self, env, cond, band_tol=0.05):
        self.active = False
        self.reached = {}        # bddl_ref -> bool (ever in-band inside fridge)
        self.reached_step = {}   # bddl_ref -> first step it reached
        self.peak_z = {}         # bddl_ref -> max z while inside fridge footprint (None if never inside)
        self.min_dist = {}       # bddl_ref -> min |z - band| while inside fridge (None if never inside)
        self._zlo = self._zhi = None
        self._cands = []         # [(bddl_ref, obj)]
        self._fridge = None
        if not cond or cond.get("type") != "in_layer":
            return
        zmin, zmax = cond.get("z_min"), cond.get("z_max")
        if zmin is None or zmax is None:
            return
        self._zlo, self._zhi = zmin - band_tol, zmax + band_tol
        try:
            scope = env.task.object_scope
        except Exception:
            return
        tgt_cat = str(cond["obj"]).rsplit("_", 1)[0]
        for k in list(scope.keys()):
            if str(k).rsplit("_", 1)[0] != tgt_cat:
                continue
            o = scope[k]
            self._cands.append((k, getattr(o, "wrapped_obj", o)))
            self.reached[k] = False
            self.peak_z[k] = None
            self.min_dist[k] = None
        if cond.get("ref") and cond["ref"] in scope:
            fo = scope[cond["ref"]]
            self._fridge = getattr(fo, "wrapped_obj", fo)
        self.active = len(self._cands) > 0

    def update(self, step):
        if not self.active:
            return
        aabb = None
        if self._fridge is not None:
            try:
                _lo, _hi = self._fridge.aabb
                aabb = (np.asarray(_lo, dtype=float), np.asarray(_hi, dtype=float))
            except Exception:
                aabb = None
        for k, o in self._cands:
            try:
                p = np.asarray(get_object_pos(o), dtype=float)
            except Exception:
                continue
            z = float(p[2])
            # Only count height while the tup is INSIDE the fridge x,y footprint, so
            # transit over the counter doesn't inflate the attempted-layer height.
            in_xy = True
            if aabb is not None:
                _lo, _hi = aabb
                in_xy = bool(_lo[0] <= p[0] <= _hi[0] and _lo[1] <= p[1] <= _hi[1])
            if not in_xy:
                continue
            # Peak height + closest approach, tracked EVERY step (not stopped at first
            # reach) so an overshoot's true peak layer is captured — an aim-higher pass
            # THROUGH the target band must not read as "reached".
            if self.peak_z[k] is None or z > self.peak_z[k]:
                self.peak_z[k] = z
            d = 0.0 if (self._zlo <= z <= self._zhi) else min(abs(z - self._zlo), abs(z - self._zhi))
            if self.min_dist[k] is None or d < self.min_dist[k]:
                self.min_dist[k] = d
            if self._zlo <= z <= self._zhi and not self.reached[k]:
                self.reached[k] = True
                self.reached_step[k] = int(step)

    def to_dict(self):
        return {
            "any_reached": bool(any(self.reached.values())),
            "per_candidate": {
                k: {"reached_band": bool(self.reached[k]),
                    "reached_step": self.reached_step.get(k),
                    "peak_z": (round(self.peak_z[k], 4) if self.peak_z[k] is not None else None),
                    "min_dist_to_band": (round(self.min_dist[k], 4) if self.min_dist[k] is not None else None)}
                for k in self.reached
            },
        }


# ── checker class ────────────────────────────────────────────────────────

class SubtaskGoalChecker:
    """Evaluate a list of goal-condition dicts against the sim state."""

    def __init__(self, env, robot, goal_conditions, initial_positions=None,
                 entered_rooms=None, reached_object=None):
        self.env = env
        self.robot = robot
        self.goal_conditions = goal_conditions
        self.initial_positions = initial_positions or {}
        # Set of room instances the robot base ENTERED during the rollout
        # (tracked per-step by the runner). When provided, check_in_room treats
        # nav success as "reached the room at any point", not "in the room at the
        # final frame" (the policy keeps acting to max_steps and wanders off).
        self.entered_rooms = entered_rooms
        # bool: did the base reach within tol of the target OBJECT at any step
        # (tracked per-step by the runner via ObjectNavSuccessTracker). Consumed
        # by check_near_object. None → fall back to a final-frame distance check.
        self.reached_object = reached_object

    def _get_obj(self, name: str):
        scope = self.env.task.object_scope
        if name not in scope:
            raise KeyError(
                f"Object {name!r} not in task scope; available={list(scope.keys())}"
            )
        obj = scope[name]
        return obj.wrapped_obj if hasattr(obj, "wrapped_obj") else obj

    def _get_obj_or_sim(self, bddl_ref, sim_id):
        """Resolve a reference object by BDDL ref (object_scope) first, else by
        SIM-ID (scene object name). Direction references can be real scene objects
        that aren't declared BDDL task objects — e.g. carving v1's `microwave`
        (microwave_abzvij_0) — so fall back to the scene registry by sim_id."""
        scope = self.env.task.object_scope
        if bddl_ref and bddl_ref in scope:
            o = scope[bddl_ref]
            return o.wrapped_obj if hasattr(o, "wrapped_obj") else o
        if sim_id:
            o = self.env.scene.object_registry("name", sim_id)
            if o is not None:
                return o.wrapped_obj if hasattr(o, "wrapped_obj") else o
        raise KeyError(f"ref bddl={bddl_ref!r} sim_id={sim_id!r} not resolvable")

    def check_ontop(self, cond) -> bool:
        obj = self._get_obj(cond["obj"])
        target = self._get_obj(cond["target"])
        obj_pos = get_object_pos(obj)
        tgt_pos = get_object_pos(target)
        z_diff = obj_pos[2] - tgt_pos[2]
        horiz_dist = float(np.linalg.norm(obj_pos[:2] - tgt_pos[:2]))
        passed = z_diff > -0.05 and horiz_dist < 2.0
        logger.info(
            f"  ontop({cond['obj']}, {cond['target']}): z_diff={z_diff:.3f} "
            f"horiz_dist={horiz_dist:.3f} -> {'PASS' if passed else 'FAIL'}"
        )
        return passed

    # ── quantities shared by the end-of-episode checks and start_state() ──
    def _eef_dist_to_candidates(self, candidate_refs) -> dict:
        """Candidate -> distance to whichever end effector is closer to it."""
        eef_ps = [self.robot.eef_links[a].get_position_orientation()[0]
                  for a in self.robot.arm_names]
        out = {}
        for c in candidate_refs:
            cp = np.asarray(get_object_pos(self._get_obj(c)))
            out[c] = min(float(np.linalg.norm(cp - np.asarray(ep))) for ep in eef_ps)
        return out

    def _base_dist_to_candidates(self, candidate_refs, log_prefix="base_near_target") -> dict:
        """Candidate -> horizontal distance from the robot base."""
        base_p = np.asarray(self.robot.get_position_orientation()[0])[:2]
        out = {}
        for c in candidate_refs:
            try:
                cp = np.asarray(get_object_pos(self._get_obj(c)))[:2]
                out[c] = float(np.linalg.norm(cp - base_p))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  {log_prefix}: pos failed for {c}: {e}")
        return out

    def _proj_right(self, cond) -> float:
        """Signed offset of the object from the reference along the robot's right axis."""
        obj = self._get_obj(cond["obj"])
        ref = self._get_obj_or_sim(cond.get("ref"), cond.get("ref_sim_id"))
        _, right = get_robot_forward_right(self.robot)
        return float(np.dot((np.asarray(get_object_pos(obj)) - np.asarray(get_object_pos(ref)))[:2],
                            np.asarray(right)[:2]))

    def start_state(self) -> dict:
        """Snapshot, taken right before the first policy action, of the quantities
        the soft lean compares at the end: per-candidate end-effector distance
        (touched_target), per-candidate base distance (base_near_target) and the
        object's offset from the reference (left_of / right_of). The summarizer
        credits a lean only if the episode made progress from these values."""
        out = {}
        for cond in self.goal_conditions:
            ctype = cond.get("type")
            try:
                if ctype == "touched_target":
                    d = self._eef_dist_to_candidates(list(cond.get("candidates") or [cond["obj"]]))
                    out["eef_dist_to_candidate"] = {c: round(v, 4) for c, v in d.items()}
                elif ctype == "base_near_target":
                    d = self._base_dist_to_candidates(list(cond.get("candidates") or [cond["obj"]]))
                    out["base_dist_to_candidate"] = {c: round(v, 4) for c, v in d.items()}
                elif ctype in ("left_of", "right_of"):
                    out["proj_right"] = round(self._proj_right(cond), 4)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  start_state: {ctype} failed: {e}")
                out.setdefault("errors", []).append(f"{ctype}: {e!r}")
        return out

    def check_direction_condition(self, cond) -> bool:
        from omnigibson.object_states import OnTop
        obj = self._get_obj(cond["obj"])
        ref = self._get_obj_or_sim(cond.get("ref"), cond.get("ref_sim_id"))
        obj_pos = get_object_pos(obj)
        ref_pos = get_object_pos(ref)
        fwd, right = get_robot_forward_right(self.robot)
        direction = cond["type"]
        direction_ok = check_direction(obj_pos, ref_pos, fwd, right, direction)
        # Signed actual side -> records which side it REALLY landed (soft signal).
        proj_right = float(np.dot((np.asarray(obj_pos) - np.asarray(ref_pos))[:2],
                                  np.asarray(right)[:2]))
        actual_side = "right" if proj_right > 0 else "left"
        # Which surface the knife rests on, via the OnTop physics state (not a
        # geometric guess). On a surface + correct side = correct; on a surface but
        # wrong side = wrong (placed, wrong dir); off every surface = none (dropped).
        # The STOVE/cooktop sits IN the countertop at counter height, and in some
        # scene constructions the intended placement target IS the burner region,
        # so a knife resting on the stove counts as "on the countertop" (2026-07-11;
        # was scored 'none' before, since only countertop/board were recognized).
        # `support_object` records the actual scope object for transparency.
        on_surface = None
        support_object = None
        for sname in self.env.task.object_scope.keys():
            low = str(sname).lower()
            is_board = ("chopping_board" in low or "cutting_board" in low)
            is_counter = (low.startswith("countertop") or low.startswith("stove")
                          or "cooktop" in low or "burner" in low or low.startswith("range"))
            if not (is_board or is_counter):
                continue
            try:
                if obj.states[OnTop].get_value(self._get_obj(sname)):
                    support_object = sname
                    on_surface = ("cutting_board" if is_board
                                  else ("countertop" if low.startswith("countertop")
                                        else "stove"))
                    break
            except Exception:
                continue

        proximity_threshold = cond.get("proximity_threshold", 1.0)
        horiz_dist = float(np.linalg.norm(obj_pos[:2] - ref_pos[:2]))
        proximity_ok = horiz_dist < proximity_threshold
        ref_name = cond.get("ref") or cond.get("ref_sim_id")
        ref_disp = 0.0
        ref_stable = True
        if ref_name in self.initial_positions:
            init_ref = np.array(self.initial_positions[ref_name])
            ref_disp = float(np.linalg.norm(ref_pos[:2] - init_ref[:2]))
            ref_stable = ref_disp < 0.3

        if on_surface is None:
            outcome = "none"
        elif direction_ok and proximity_ok and ref_stable:
            outcome = "correct"
        else:
            outcome = "wrong"
        passed = (outcome == "correct")
        self.direction_diag = {
            "actual_side": actual_side, "expected": direction,
            "direction_ok": bool(direction_ok), "on_surface": on_surface,
            "support_object": support_object,
            "horiz_to_ref": round(horiz_dist, 4), "proximity_ok": bool(proximity_ok),
            "ref_stable": bool(ref_stable), "proj_right": round(proj_right, 4),
            "outcome": outcome,
        }
        logger.info(
            f"  {direction}({cond['obj']}, {ref_name}): side={actual_side} "
            f"on={on_surface} dir_ok={direction_ok} prox={proximity_ok} "
            f"ref_stable={ref_stable} -> {outcome.upper()}"
        )
        return passed

    def check_in_layer(self, cond) -> bool:
        """Locate-task check: did the placed object end inside layer N's z-band?

        The band (`z_min`,`z_max`) is the empirically-derived world-z range for
        that fridge layer — computed from where demonstrators actually placed the
        object on train demos (no geometric layer metadata exists) — and embedded
        in the goal condition by build_slice_held_out_locate. We also require the
        object to be near the fridge in the floor plane so a correct height far
        from the fridge isn't counted.
        """
        z_min, z_max = cond.get("z_min"), cond.get("z_max")
        if z_min is None or z_max is None:
            logger.warning("  in_layer: cond has no z_min/z_max; cannot score -> FAIL")
            return False
        # Band tolerance (band runs tight, pad a few cm). PLACE_DISPLACE_MIN is the
        # min init->final move to CREDIT an object: a tupperware ALREADY in the
        # fridge at the start (never moved by the policy) must NOT count, so we
        # require the object to have been displaced ("in hand initially" proxy).
        _band_tol = float(os.environ.get("LAYER_BAND_TOL", "0.05"))
        _disp_min = float(os.environ.get("PLACE_DISPLACE_MIN", "0.10"))
        _prox = cond.get("proximity_threshold", 1.5)
        # "inside the fridge" = xy within the fridge's AABB footprint (tight), NOT
        # merely within _prox metres -- a tupperware on a nearby counter at the
        # right height must NOT count. Fall back to the radius if AABB unavailable.
        ref_pos = None
        ref_aabb = None
        if cond.get("ref"):
            try:
                _ref_obj = self._get_obj(cond["ref"])
                ref_pos = np.asarray(get_object_pos(_ref_obj))
                _ru = getattr(_ref_obj, "wrapped_obj", _ref_obj)
                _alo, _ahi = _ru.aabb
                ref_aabb = (np.asarray(_alo, dtype=float), np.asarray(_ahi, dtype=float))
            except Exception:
                ref_aabb = None
        # Candidates: EVERY instance of the target's category (e.g. all tupperwares).
        # The robot may hold >1 and place either (interchangeable), so PASS if ANY
        # one the robot MOVED ends in the target band near the fridge.
        tgt_cat = str(cond["obj"]).rsplit("_", 1)[0]
        cand_names = [k for k in self.env.task.object_scope.keys()
                      if str(k).rsplit("_", 1)[0] == tgt_cat]
        # HELD-ONLY RESTRICTION (2026-08-21, user-found). Scoring every instance
        # of the category credits an object the robot never had to place: these
        # episodes resume MID-TASK, so a sibling tupperware/bowl is often already
        # sitting in the fridge/cabinet from earlier in the demo. `moved` alone
        # does not save us — nudging that sibling >disp_min while inserting the
        # held one scores it as the success (observed: prompt_probe
        # v0_ep00250410 passed on tupperware.n.01_2 while the goal object was
        # tupperware.n.01_1). When the instance declares which objects were in
        # hand at t=0 (`held_objs`, stamped from the TRO grasps), restrict the
        # candidate set to those — the robot may legitimately hold TWO and place
        # either, which is why this is not simply `cond["obj"]`.
        _held = [str(h) for h in (cond.get("held_objs") or [])]
        if _held:
            _restricted = [k for k in cand_names if str(k) in _held]
            if _restricted:
                cand_names = _restricted
        cands = []
        for cname in cand_names:
            try:
                cpos = np.asarray(get_object_pos(self._get_obj(cname)))
            except Exception:
                continue
            cz = float(cpos[2])
            in_band = bool(z_min - _band_tol <= cz <= z_max + _band_tol)
            horiz = None if ref_pos is None else float(np.linalg.norm(cpos[:2] - ref_pos[:2]))
            if ref_aabb is not None:
                _alo, _ahi = ref_aabb
                near = bool(_alo[0] <= cpos[0] <= _ahi[0] and _alo[1] <= cpos[1] <= _ahi[1])
            else:
                near = (horiz is None) or (horiz < _prox)
            init = self.initial_positions.get(cname)
            disp = None if init is None else float(np.linalg.norm(cpos - np.asarray(init)))
            moved = (disp is None) or (disp > _disp_min)
            cands.append({
                "name": cname, "z": round(cz, 4), "in_band": in_band,
                "near_fridge": bool(near),
                "displacement": (None if disp is None else round(disp, 3)),
                "moved": bool(moved), "counts": bool(in_band and near and moved),
                "horiz_to_fridge": (None if horiz is None else round(horiz, 4)),
            })
        passed = any(c["counts"] for c in cands)
        moved_cands = [c for c in cands if c["moved"]]
        if passed:
            _outcome = "correct"
        elif any(c["near_fridge"] for c in moved_cands):
            _outcome = "in_fridge_wrong"
        else:
            _outcome = "not_in_fridge"
        self.layer_diag = {
            "target_layer": cond.get("layer"),
            "z_min": z_min, "z_max": z_max, "band_tol": _band_tol, "disp_min": _disp_min,
            "outcome": _outcome,
            "n_counts": int(sum(c["counts"] for c in cands)),
            "candidates": cands,
        }
        logger.info(
            f"  in_layer(layer={cond.get('layer')}): "
            f"{sum(c['counts'] for c in cands)}/{len(cands)} count, outcome={_outcome} "
            f"-> {'PASS' if passed else 'FAIL'} | "
            + "; ".join(f"{c['name'].split('.n')[0]}(z={c['z']},mv={c['moved']},band={c['in_band']})" for c in cands)
        )
        return passed

    def check_grasped(self, cond) -> bool:
        obj_name = cond["obj"]
        obj = self._get_obj(obj_name)
        obj_pos = get_object_pos(obj)

        # DIAGNOSTIC: also log positions + displacements for all paperback-like
        # instances so we can tell which object was actually moved even if the
        # goal's target object didn't. Use print() to ensure output goes to
        # the subtask client's captured stdout regardless of logger config.
        scope = self.env.task.object_scope
        import sys
        diag_prefix_items = [
            k for k in scope.keys()
            if obj_name.split(".n.01")[0] in k  # e.g., 'paperback_book'
        ]
        print(f"[goal_checker.diag] check_grasped target={obj_name}", flush=True)
        for k in diag_prefix_items:
            try:
                o = scope[k]
                o_actual = o.wrapped_obj if hasattr(o, "wrapped_obj") else o
                p = get_object_pos(o_actual)
                init = self.initial_positions.get(k)
                if init is not None:
                    disp = float(np.linalg.norm(p - np.array(init)))
                    print(
                        f"[goal_checker.diag]   {k}: init={np.array(init).round(3).tolist()} "
                        f"final={p.round(3).tolist()} displacement={disp:.3f}m",
                        flush=True,
                    )
                else:
                    print(
                        f"[goal_checker.diag]   {k}: final={p.round(3).tolist()} z={p[2]:.3f} (no init pos)",
                        flush=True,
                    )
            except Exception as e:
                print(f"[goal_checker.diag]   {k}: error {e}", flush=True)

        if obj_name in self.initial_positions:
            init_pos = np.array(self.initial_positions[obj_name])
            displacement = float(np.linalg.norm(obj_pos - init_pos))
            # Require actual vertical lift to count as a grasp. The prior
            # `displacement > 3cm OR lift > 1cm` clause produced false positives
            # whenever the gripper just swiped/pushed the book horizontally
            # without lifting (e.g. v1_ep00290300 saw 15cm displacement with
            # lift_z=0). The displacement-OR branch is dropped so a horizontal
            # nudge no longer passes; brief lift-and-drop still passes via the
            # 1cm lift threshold.
            lift = float(obj_pos[2] - init_pos[2])
            passed = lift > 0.01
            print(
                f"[goal_checker.diag] RESULT grasped({obj_name}): displacement={displacement:.3f}m "
                f"lift_z={lift:+.3f}m -> {'PASS' if passed else 'FAIL'}",
                flush=True,
            )
            logger.info(
                f"  grasped({obj_name}): displacement={displacement:.3f}m "
                f"-> {'PASS' if passed else 'FAIL'}"
            )
        else:
            passed = obj_pos[2] > 1.0
            print(
                f"[goal_checker.diag] RESULT grasped({obj_name}): z={obj_pos[2]:.3f} (no init) "
                f"-> {'PASS' if passed else 'FAIL'}",
                flush=True,
            )
            logger.info(
                f"  grasped({obj_name}): z={obj_pos[2]:.3f} (no init) "
                f"-> {'PASS' if passed else 'FAIL'}"
            )
        return passed

    def check_touched_target(self, cond) -> bool:
        """Reach-task custom check: which candidate(s) is the agent touching at end-of-rollout?

        Records on `self.touched_diag`:
            touched_candidates:  list of bddl refs the agent is touching
            gt_target:           cond["obj"]
            outcome:             'correct' (gt touched), 'wrong' (other touched), or 'none'

        Returns:
            passed = (gt_target in touched_candidates)

        Designed to pair with a BDDL goal `(or (touching agent X1) (touching agent X2) ...)`
        so the sim terminates on the first touch of ANY candidate; this check then tells
        us which one was actually touched.
        """
        # Defer omnigibson import to method scope so this module can be imported in
        # contexts where omnigibson isn't loaded (e.g. tests, doc tools).
        from omnigibson.object_states import Touching  # type: ignore

        gt_ref = cond["obj"]
        candidate_refs = list(cond.get("candidates") or [gt_ref])
        try:
            agent = self._get_obj("agent.n.01_1")
        except KeyError:
            logger.warning("  touched_target: agent.n.01_1 not in object_scope; skipping")
            self.touched_diag = {
                "touched_candidates": [], "gt_target": gt_ref,
                "candidates": candidate_refs, "outcome": "unknown",
                "error": "agent.n.01_1 not in scope",
            }
            return False

        touched = []
        for cand_ref in candidate_refs:
            try:
                cand_obj = self._get_obj(cand_ref)
                # OmniGibson's Touching state is symmetric; compare agent ↔ candidate.
                if agent.states[Touching].get_value(cand_obj):
                    touched.append(cand_ref)
            except Exception as e:
                logger.warning(f"  touched_target: contact check failed for {cand_ref}: {e}")

        # GT may be a TIE (e.g. two equally-smallest food items): accept any of
        # `gt_targets` if provided, else just `obj`. Touching any GT = correct.
        gt_set = set(cond.get("gt_targets") or [gt_ref])
        touched_gt = [t for t in touched if t in gt_set]
        if not touched:
            outcome = "none"
        elif touched_gt:
            outcome = "correct"
        else:
            outcome = "wrong"

        self.touched_diag = {
            "touched_candidates": touched,
            "gt_target": gt_ref,
            "gt_targets": sorted(gt_set),
            "candidates": candidate_refs,
            "outcome": outcome,
        }
        # Per-candidate min-EEF distance → enables the 'closer-to-correct/wrong'
        # split on none/wrong outcomes for ANY number of candidates (generalises
        # the paperback 2-candidate left/right). closer_to compares the nearest GT
        # vs the nearest wrong candidate to whichever gripper got closest.
        try:
            cand_d = self._eef_dist_to_candidates(candidate_refs)
            gtd = [cand_d[c] for c in gt_set if c in cand_d]
            wrd = [cand_d[c] for c in candidate_refs if c not in gt_set and c in cand_d]
            self.touched_diag["eef_min_dist_to_candidate"] = {c: round(v, 4) for c, v in cand_d.items()}
            self.touched_diag["min_dist_to_gt"] = round(min(gtd), 4) if gtd else None
            self.touched_diag["min_dist_to_wrong"] = round(min(wrd), 4) if wrd else None
            if gtd and wrd:
                self.touched_diag["closer_to"] = "correct" if min(gtd) <= min(wrd) else "wrong"
        except Exception as _e:  # noqa: BLE001
            logger.warning(f"  touched_target: closer-dist calc failed: {_e}")
        passed = bool(touched_gt)
        logger.info(
            f"  touched_target(gt={sorted(gt_set)}): touched={touched} outcome={outcome} "
            f"-> {'PASS' if passed else 'FAIL'}"
        )
        return passed

    def check_base_near_target(self, cond) -> bool:
        """Floor-reach success by BASE POSITION (e.g. reach_soda_can).

        For floor objects scattered across a room, the discriminating behavior
        is NAVIGATING to the right object; requiring an eef touch inflates
        'none' outcomes (policies rarely bend to floor height). Success =
        the base ends within `tol` (m, default 1.5 — nav near_object
        convention) of a GT target AND the NEAREST candidate to the base is a
        GT (disambiguation when candidates are close together).

        Pairs with the usual `(or (touching agent X1) ...)` BDDL termination:
        touching is still allowed to END the episode early, but scoring is
        purely base-distance at end-of-rollout.

        Records `self.base_near_diag` (mirrors touched_diag: outcome
        correct/wrong/none + per-candidate base distances) for the summarizer.
        """
        gt_ref = cond["obj"]
        candidate_refs = list(cond.get("candidates") or [gt_ref])
        gt_set = set(cond.get("gt_targets") or [gt_ref])
        # tol calibrated on ALL 140 soda_can train instances (2026-07-21):
        # demo base->GT distance AT CONTACT = 0.40-0.70m (p95 0.65), while
        # base->WRONG can at that moment is min 0.68 / p5 0.92 -> 0.70m covers
        # every demonstrated success position and excludes ~all wrong-can
        # positions (the lone 0.68 overlap is resolved by the nearest rule).
        tol = float(cond.get("tol", 0.70))

        # UNION with contact: touching a GT candidate is success regardless of
        # base distance — the tight tol must never punish a genuine reach
        # (and keeps near-init configs honest via contact).
        try:
            from omnigibson.object_states import Touching  # type: ignore
            agent = self._get_obj("agent.n.01_1")
            for _g in gt_set:
                try:
                    if agent.states[Touching].get_value(self._get_obj(_g)):
                        self.base_near_diag = {
                            "outcome": "correct", "via": "touch",
                            "gt_target": gt_ref, "gt_targets": sorted(gt_set),
                            "candidates": candidate_refs, "tol": tol,
                        }
                        logger.info(f"  base_near_target: touching GT {_g} -> PASS (via touch)")
                        return True
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

        cand_d = self._base_dist_to_candidates(candidate_refs)
        if not cand_d:
            self.base_near_diag = {"outcome": "unknown", "error": "no candidate positions",
                                   "gt_target": gt_ref, "candidates": candidate_refs}
            return False

        nearest = min(cand_d, key=cand_d.get)
        gtd = [cand_d[c] for c in gt_set if c in cand_d]
        wrd = [cand_d[c] for c in candidate_refs if c not in gt_set and c in cand_d]
        passed = bool(gtd) and min(gtd) <= tol and nearest in gt_set
        if passed:
            outcome = "correct"
        elif wrd and min(wrd) <= tol and nearest not in gt_set:
            outcome = "wrong"
        else:
            outcome = "none"

        self.base_near_diag = {
            "outcome": outcome,
            "gt_target": gt_ref,
            "gt_targets": sorted(gt_set),
            "candidates": candidate_refs,
            "nearest": nearest,
            "tol": tol,
            "base_dist_to_candidate": {c: round(v, 4) for c, v in cand_d.items()},
            "min_dist_to_gt": round(min(gtd), 4) if gtd else None,
            "min_dist_to_wrong": round(min(wrd), 4) if wrd else None,
        }
        if gtd and wrd:
            self.base_near_diag["closer_to"] = "correct" if min(gtd) <= min(wrd) else "wrong"
        logger.info(
            f"  base_near_target(gt={sorted(gt_set)}, tol={tol}): nearest={nearest} "
            f"d={cand_d[nearest]:.2f} outcome={outcome} -> {'PASS' if passed else 'FAIL'}"
        )
        return passed

    def _get_seg_map(self):
        """Locate the scene's room-segmentation map (for nav `in_room` checks).

        Mirrors OmniGibson's own usage (`robot.scene._seg_map` in
        starter_semantic_action_primitives); tries env.scene then robot.scene.
        """
        for owner in (getattr(self.env, "scene", None), getattr(self.robot, "scene", None)):
            seg = getattr(owner, "_seg_map", None) if owner is not None else None
            if seg is not None:
                return seg
        return None

    def check_in_room(self, cond) -> bool:
        """Nav-task success: did the robot base REACH the target room instance?

        Primary path: success = the base entered `cond["room_instance"]` (e.g.
        "garage_0") at ANY point during the rollout (`self.entered_rooms`, tracked
        per-step by the runner). This is the correct "move-to" criterion — it's
        robust to the policy overshooting/wandering after arrival, which a
        final-frame-only check misses.

        Fallback (no tracking provided): instantaneous final-frame room membership
        via `scene._seg_map.get_room_instance_by_point(xy)`.
        """
        import sys
        target_room = cond["room_instance"]
        if self.entered_rooms is not None:
            passed = target_room in self.entered_rooms
            print(
                f"[goal_checker] in_room(robot, {target_room}): entered_rooms="
                f"{sorted(self.entered_rooms)} -> {'PASS' if passed else 'FAIL'}",
                flush=True,
            )
            return passed
        seg_map = self._get_seg_map()
        if seg_map is None:
            logger.error("  in_room: scene._seg_map not found; cannot check room membership")
            return False
        pos, _ = self.robot.get_position_orientation()
        xy = pos[:2]  # keep native dtype — OmniGibson passes pose tensors directly
        try:
            room = seg_map.get_room_instance_by_point(xy)
        except Exception as e:
            logger.error(f"  in_room: get_room_instance_by_point failed: {e}")
            return False
        passed = room == target_room
        xy_log = (xy.cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy)).round(3).tolist()
        print(
            f"[goal_checker] in_room(robot, {target_room}) [final-frame]: robot_xy={xy_log} "
            f"room={room} -> {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        return passed

    def check_near_object(self, cond) -> bool:
        """Nav-to-object success: did the base come within `tol_m` of the target
        object? Primary path: `self.reached_object` (tracked per-step by the runner,
        any-step — robust to overshoot). Fallback (no tracking): final-frame
        base-to-object distance, locating the target by SIM ID in the scene registry
        (the object may NOT be a BDDL task object, e.g. breakfast_table)."""
        import sys
        tol = float(cond.get("tol_m", 1.5))
        sim_id = cond.get("target_object_id")
        if self.reached_object is not None:
            passed = bool(self.reached_object)
            print(f"[goal_checker] near_object({sim_id}, tol={tol}): reached={passed} "
                  f"-> {'PASS' if passed else 'FAIL'}", flush=True)
            return passed
        obj = None
        try:
            obj = self.env.scene.object_registry("name", sim_id)
        except Exception:
            obj = None
        if obj is None:
            logger.error(f"  near_object: target {sim_id!r} not found in scene registry")
            return False
        _o = obj.wrapped_obj if hasattr(obj, "wrapped_obj") else obj
        op = get_object_pos(_o)
        pos, _ = self.robot.get_position_orientation()
        bxy = pos[:2].cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos[:2])
        # Mirror the per-step tracker: footprint distance for extended targets,
        # else centroid. Without this the fallback path would silently disagree
        # with the tracker on exactly the objects footprint mode exists for.
        _ab = obj_xy_aabb(_o) if cond.get("distance_mode") == "footprint" else None
        if _ab is not None:
            d = xy_dist_to_aabb(bxy, *_ab)
        else:
            d = float(np.linalg.norm(np.asarray(op[:2], dtype=float) - np.asarray(bxy, dtype=float)))
        passed = d <= tol
        req_room = cond.get("require_room")
        if passed and req_room:
            room = None
            try:
                _sm = getattr(getattr(self.env, "scene", None), "_seg_map", None)
                if _sm is not None:
                    room = _sm.get_room_instance_by_point(np.asarray(bxy, dtype=float))
            except Exception:
                room = None
            if room != req_room:
                passed = False
                print(f"[goal_checker] near_object require_room={req_room}: final room={room} "
                      f"-> FAIL despite dist", flush=True)
        print(f"[goal_checker] near_object({sim_id}, tol={tol}) [final-frame]: dist={d:.3f} "
              f"-> {'PASS' if passed else 'FAIL'}", flush=True)
        return passed

    def check_all(self):
        results: dict[str, bool] = {}
        all_passed = True
        logger.info("=== Subtask goal check ===")
        for cond in self.goal_conditions:
            ctype = cond["type"]
            key = f"{ctype}({cond.get('obj', '?')}, {cond.get('target', cond.get('ref', '?'))})"
            try:
                if ctype == "ontop":
                    passed = self.check_ontop(cond)
                elif ctype in ("left_of", "right_of", "in_front_of", "behind"):
                    passed = self.check_direction_condition(cond)
                elif ctype == "in_layer":
                    passed = self.check_in_layer(cond)
                elif ctype == "grasped":
                    passed = self.check_grasped(cond)
                elif ctype == "touched_target":
                    passed = self.check_touched_target(cond)
                elif ctype == "base_near_target":
                    passed = self.check_base_near_target(cond)
                elif ctype == "in_room":
                    passed = self.check_in_room(cond)
                elif ctype == "near_object":
                    passed = self.check_near_object(cond)
                else:
                    logger.warning(f"  Unknown goal type: {ctype}, skipping")
                    passed = False
            except Exception as e:
                logger.error(f"  Error checking {key}: {e}")
                passed = False
            results[key] = passed
            if not passed:
                all_passed = False
        logger.info(
            f"  Overall: {'SUCCESS' if all_passed else 'FAIL'} "
            f"({sum(results.values())}/{len(results)})"
        )
        return all_passed, results


# ── reach dwell tracker ──────────────────────────────────────────────────
class ReachDwellTracker:
    """Dwell-based reach outcome: the candidate where the hand STAYS.

    The reach BDDL goal `(or (touching agent X1) (touching agent X2) ...)`
    terminates the sim on the FIRST touch of ANY candidate, and
    check_touched_target judges the instantaneous contact at that moment. A
    transient graze of a distractor en route therefore ends the rollout as
    'wrong' even when the arm was heading to the correct object (observed on
    train_sanity: 6/10 reach failures were early wrong-touch stops).

    This tracker redefines the outcome as SUSTAINED contact: the first
    candidate the agent keeps touching for `dwell_steps` CONSECUTIVE sim steps
    wins. The eval loop treats the BDDL first-touch termination as ADVISORY
    for reach tasks and keeps stepping until a dwell decision or max-steps
    truncation. Same per-step tracker shape as NavSuccessTracker /
    LayerReachTracker (`active` / `update(step)` / `to_dict()`).

    Tie rule: if a GT candidate and a distractor complete the dwell on the
    same step (adjacent objects, hand touching both), GT wins — sustained GT
    contact is a reach of the target regardless of what else the hand brushes.

    STATIONARITY guard (2026-07-10): in clutter, contact alone is not
    commitment — a hand pushing PAST a neighbor keeps contact while moving.
    And with REPLAN_STEPS=10 the policy executes 10 env steps open-loop, so a
    10-step contact streak fits inside ONE blind chunk-execution window. Two
    changes: (a) default dwell_steps 10 -> 30 (~1 s = 3 replan cycles, so the
    policy gets >=2 chances to move OFF a grazed object before we commit);
    (b) the dwell only decides when the contacting gripper is also near-
    STATIONARY over the window — bbox diagonal of the closest-EEF trajectory
    across the last `dwell_steps` contact steps < `stationary_eps`. "Stopped
    at the object" signals intent; "brushed while passing" defers until the
    hand settles or contact breaks. stationary_eps <= 0 disables the guard.
    """

    def __init__(self, env, robot, goal_conditions, dwell_steps: int = 30,
                 stationary_eps: float = 0.03, target_only: bool = False):
        cond = next((c for c in (goal_conditions or [])
                     if c.get("type") == "touched_target"), None)
        self.active = bool(cond) and int(dwell_steps) > 0
        # TARGET-ONLY mode (2026-08-08, user decision for candidate-cluttered
        # reach tasks like reach_storage_box): a dwell on a NON-target candidate
        # does NOT end the episode — the robot may be pushing past clutter while
        # still heading for the target. Only a GT dwell decides ('correct');
        # otherwise the rollout runs to truncation and the caller scores by the
        # closest candidate at timeout (see decide_at_timeout()). Wrong-dwell
        # events are still RECORDED (n_nontarget_dwells / nontarget_dwelled) so
        # the old strict semantics remain reconstructible from the metrics.
        self.target_only = bool(target_only)
        self.n_nontarget_dwells = 0
        self.nontarget_dwelled = []
        if not self.active:
            return
        self.env = env
        self.robot = robot
        self.dwell_steps = int(dwell_steps)
        self.stationary_eps = float(stationary_eps)
        self.gt_ref = cond["obj"]
        self.candidates = list(cond.get("candidates") or [self.gt_ref])
        self.gt_set = set(cond.get("gt_targets") or [self.gt_ref])
        self._objs = None            # lazy: resolved on first update()
        self._agent = None
        self.streak = {c: 0 for c in self.candidates}
        self.max_streak = {c: 0 for c in self.candidates}
        self.n_touch_steps = {c: 0 for c in self.candidates}
        self.first_touch_step = {}
        # rolling window: closest-EEF position at each step of the current
        # contact streak, per candidate (cleared when the streak breaks)
        self._eef_window = {c: [] for c in self.candidates}
        self.n_deferred_moving = 0    # windows where contact met K but hand moved
        self.decided = False
        self.decided_step = None
        self.decided_candidates = []
        self.decided_eef_disp = None
        self.outcome = "none"        # none | correct | wrong
        self.n_update_errors = 0

    def _resolve(self):
        scope = self.env.task.object_scope
        self._agent = scope["agent.n.01_1"]
        self._objs = {}
        for c in self.candidates:
            try:
                self._objs[c] = scope[c]
            except KeyError:
                logger.warning(f"  [dwell] candidate {c} not in object_scope; ignored")

    def update(self, step: int) -> None:
        if not self.active or self.decided:
            return
        try:
            from omnigibson.object_states import Touching  # type: ignore
            if self._objs is None:
                self._resolve()
            # EEF world positions once per step (shared across candidates).
            eef_ps = [np.asarray(self.robot.eef_links[a].get_position_orientation()[0].cpu()
                                 if hasattr(self.robot.eef_links[a].get_position_orientation()[0], "cpu")
                                 else self.robot.eef_links[a].get_position_orientation()[0])
                      for a in self.robot.arm_names]
            finished = []
            disp_of = {}
            for c, obj in self._objs.items():
                touching = bool(self._agent.states[Touching].get_value(obj))
                if touching:
                    self.n_touch_steps[c] += 1
                    self.first_touch_step.setdefault(c, int(step))
                    self.streak[c] += 1
                    self.max_streak[c] = max(self.max_streak[c], self.streak[c])
                    # closest EEF to this candidate this step -> rolling window
                    cpos = np.asarray(get_object_pos(obj))
                    eef = min(eef_ps, key=lambda p: float(np.linalg.norm(p - cpos)))
                    w = self._eef_window[c]
                    w.append(eef)
                    if len(w) > self.dwell_steps:
                        w.pop(0)
                    if self.streak[c] >= self.dwell_steps:
                        if self.stationary_eps <= 0:
                            finished.append(c)
                        else:
                            arr = np.stack(w[-self.dwell_steps:])
                            disp = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
                            disp_of[c] = disp
                            if disp < self.stationary_eps:
                                finished.append(c)
                            else:
                                self.n_deferred_moving += 1
                                if self.n_deferred_moving in (1, 30, 150) or self.n_deferred_moving % 600 == 0:
                                    logger.info(
                                        f"  [dwell] step {step}: contact streak >= {self.dwell_steps} on {c} "
                                        f"but hand still moving (window disp {disp:.3f} >= {self.stationary_eps}); deferring"
                                    )
                else:
                    self.streak[c] = 0
                    self._eef_window[c].clear()
            if finished and self.target_only and not any(c in self.gt_set for c in finished):
                # non-target dwell in target-only mode: log + record, keep going
                for c in finished:
                    if c not in self.nontarget_dwelled:
                        self.nontarget_dwelled.append(c)
                self.n_nontarget_dwells += 1
                if self.n_nontarget_dwells in (1, 10) or self.n_nontarget_dwells % 200 == 0:
                    logger.info(
                        f"  [dwell] step {step}: NON-target dwell on {finished} "
                        f"(target_only mode; continuing toward gt={sorted(self.gt_set)})"
                    )
                for c in finished:
                    self.streak[c] = 0
                    self._eef_window[c].clear()
                finished = []
            if finished:
                self.decided = True
                self.decided_step = int(step)
                self.decided_candidates = finished
                self.decided_eef_disp = round(min(disp_of.get(c, 0.0) for c in finished), 4)
                gt_hit = [c for c in finished if c in self.gt_set]
                self.outcome = "correct" if gt_hit else "wrong"
                logger.info(
                    f"  [dwell] decided at step {step}: {finished} "
                    f"(gt={sorted(self.gt_set)}) -> {self.outcome} "
                    f"(K={self.dwell_steps}, eef_disp={self.decided_eef_disp}, "
                    f"deferred_moving={self.n_deferred_moving})"
                )
        except Exception as e:
            self.n_update_errors += 1
            if self.n_update_errors <= 3:
                logger.warning(f"  [dwell] update failed at step {step}: {e}")

    def decide_at_timeout(self) -> str:
        """target_only mode, truncation reached with no GT dwell: score by the
        candidate CLOSEST to either EEF at the end. 'correct' if that is a GT
        candidate, 'wrong' otherwise; 'none' when nothing is within 1.0 m (no
        commitment expressed). Records closest_at_timeout for the metrics."""
        if not self.active or self.decided:
            return self.outcome
        try:
            if self._objs is None:
                self._resolve()
            eef_ps = [np.asarray(self.robot.eef_links[a].get_position_orientation()[0].cpu()
                                 if hasattr(self.robot.eef_links[a].get_position_orientation()[0], "cpu")
                                 else self.robot.eef_links[a].get_position_orientation()[0])
                      for a in self.robot.arm_names]
            dists = {}
            for c, obj in self._objs.items():
                cpos = np.asarray(get_object_pos(obj))
                dists[c] = round(min(float(np.linalg.norm(p - cpos)) for p in eef_ps), 4)
            self.timeout_dists = dists
            closest = min(dists, key=dists.get)
            self.closest_at_timeout = closest
            if dists[closest] > 1.0:
                self.outcome = "none"
            else:
                self.outcome = "correct" if closest in self.gt_set else "wrong"
            logger.info(
                f"  [dwell] timeout decision (target_only): closest={closest} "
                f"@ {dists[closest]}m -> {self.outcome} (dists={dists})"
            )
        except Exception as e:
            logger.warning(f"  [dwell] decide_at_timeout failed: {e}")
        return self.outcome

    def to_dict(self) -> dict:
        if not self.active:
            return {"active": False}
        return {
            "active": True,
            "target_only": getattr(self, "target_only", False),
            "n_nontarget_dwells": getattr(self, "n_nontarget_dwells", 0),
            "nontarget_dwelled": getattr(self, "nontarget_dwelled", []),
            "closest_at_timeout": getattr(self, "closest_at_timeout", None),
            "timeout_dists": getattr(self, "timeout_dists", None),
            "dwell_steps": self.dwell_steps,
            "stationary_eps": self.stationary_eps,
            "outcome": self.outcome,
            "decided_step": self.decided_step,
            "decided_candidates": self.decided_candidates,
            "decided_eef_disp": self.decided_eef_disp,
            "n_deferred_moving": self.n_deferred_moving,
            "gt_target": self.gt_ref,
            "gt_targets": sorted(self.gt_set),
            "candidates": self.candidates,
            "first_touch_step": dict(self.first_touch_step),
            "max_streak": dict(self.max_streak),
            "n_touch_steps": dict(self.n_touch_steps),
            "n_update_errors": self.n_update_errors,
        }
