"""
Aggregate per-instance sim-eval metrics into per-slice and overall summary
JSONs.

Each per-instance metrics file (written by `eval_subtask.py`) contains:
  - touched_target.outcome     ('correct' | 'wrong' | 'none')
  - final_state.distance_to_target.{left, right, min}
  - slice, variant_id, demo_id, episode_unique_id, etc.

This script walks a directory of those metrics files (any nesting), groups
by (slice, variant_id), tallies outcomes + computes avg min-EEF-to-target
distance, and writes:

  <out_root>/<task_name>/<slice>/_summary.json   per slice
  <out_root>/<task_name>/_summary.json           overall (across slices)

Run:
    python summarize_eval_results.py \\
        --metrics_root /path/to/eval/run/metrics \\
        --output_root  /path/to/eval/run/summaries
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# Target z (world height) below this ⇒ the object ended on the FLOOR — a hard
# place failure (dropped), kept OUT of the soft "held, leaning correct/wrong" bucket.
_FLOOR_Z_M = 0.30

# A soft episode leans toward the instruction only if it also made progress from
# its start state (metrics["start_state"], recorded before the first policy
# action): the closer end effector (reach) or the base (base_near_target) ended at
# least this much nearer the instructed target than it started, or the knife
# (carving) ended at least this much farther to the side of the reference. A
# stationary policy's hands drift up to ~5 cm (p90), so leans from where an
# episode happened to start never count.
_MIN_PROGRESS_M = 0.05


# ── Per-instance extraction ──────────────────────────────────────────────────


def _load(fp: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.load(open(fp))
    except Exception:
        return None


def _is_eval_metrics_file(m: Dict[str, Any]) -> bool:
    """Heuristic: per-instance eval metrics files have at least one of
    `subtask_success`, `touched_target`, or `final_state`."""
    return (
        "subtask_success" in m
        or "touched_target" in m
        or "final_state" in m
    )


def _min_start(start: Dict[str, Any], key: str, refs) -> Optional[float]:
    d = start.get(key) or {}
    vals = [d[r] for r in refs if isinstance(d.get(r), (int, float))]
    return min(vals) if vals else None


def _progress(start_v: Optional[float], end_v: Optional[float]) -> Optional[float]:
    """Distance closed from start to end; None when either is unknown."""
    if isinstance(start_v, (int, float)) and isinstance(end_v, (int, float)):
        return round(start_v - end_v, 4)
    return None


def _extract_instance(m: Dict[str, Any]) -> Dict[str, Any]:
    """Pull analysis-relevant fields out of a raw metrics dict.

    For "none" outcomes (no candidate touched) we also need to know whether
    the EEF ended closer to the correct target or to a wrong candidate. We
    derive ``min_dist_to_wrong`` as the min over all non-target candidates of
    that candidate's min-arm distance.
    """
    final_state = m.get("final_state") or {}
    target_dist = final_state.get("distance_to_target") or {}
    distances_to_candidates = final_state.get("distances_to_candidates") or {}
    target_ref = final_state.get("target_bddl_ref")
    touched = m.get("touched_target") or {}

    wrong_min_dists = [
        d.get("min") for ref, d in distances_to_candidates.items()
        if ref != target_ref and isinstance(d.get("min"), (int, float))
    ]
    min_dist_to_wrong = min(wrong_min_dists) if wrong_min_dists else None
    min_dist_to_target = target_dist.get("min")
    closer_to = None
    if isinstance(min_dist_to_target, (int, float)) and isinstance(
        min_dist_to_wrong, (int, float)
    ):
        closer_to = "correct" if min_dist_to_target <= min_dist_to_wrong else "wrong"

    # Unified outcome across task families: reach uses `touched_target`; place-
    # direction (carving) uses `direction`; place-locate (tupperware) uses
    # `in_layer` (mapped: correct→correct, in_fridge_wrong→wrong, not_in_fridge→
    # none). For the soft `none` case we record which way it LEANED and whether the
    # target ended on the FLOOR (a hard drop, excluded from the soft buckets).
    outcome = touched.get("outcome")
    direction = m.get("direction") or {}
    in_layer = m.get("in_layer") or {}
    base_near = m.get("base_near_target") or {}
    obj_positions = final_state.get("object_positions") or {}
    _tpos = obj_positions.get(target_ref)
    target_z = _tpos[2] if isinstance(_tpos, (list, tuple)) and len(_tpos) == 3 else None
    floor_drop = bool(isinstance(target_z, (int, float)) and target_z < _FLOOR_Z_M)
    start = m.get("start_state") or {}
    progress_m = None    # set per family below; None = no start state recorded
    approached = None    # True / False once known

    if direction:  # carving: correct=placed correct side, wrong=placed wrong side, none=not placed
        outcome = direction.get("outcome")
        closer_to = None
        # on_ref: knife RELEASED onto/over the reference itself = "put down in
        # place", no clean direction. Without this bucket a drop-in-place is
        # scored correct/wrong by the sign of a ±3cm projection (a 235-step
        # "success" landed ON the board 5cm right of center). Criteria: episode
        # placed (outcome correct/wrong) but resting surface is NOT the
        # countertop, or the knife sits within the ref footprint (+margin).
        _ON_REF_HORIZ_M = 0.17  # ~board/pan half-extent + margin
        if outcome in ("correct", "wrong"):
            _surf = str(direction.get("on_surface") or "").lower()
            _h2r = direction.get("horiz_to_ref")
            _on_ctr = "counter" in _surf
            if (not _on_ctr) or (isinstance(_h2r, (int, float)) and _h2r < _ON_REF_HORIZ_M):
                outcome = "on_ref"
        # SOFT LEAN over BOTH the soft carving outcomes (on_ref + none): the
        # knife's lateral sign vs the instructed side. on_ref = placed on the ref
        # (degenerate direction), none = not placed on a surface; for both the
        # lean is the informative signal (matches how reach soft± covers its
        # nones). right_of ⇒ proj_right>0 is "leaning correct".
        if outcome in ("none", "on_ref"):
            _pr, _exp = direction.get("proj_right"), direction.get("expected")
            if isinstance(_pr, (int, float)) and _exp in ("right_of", "left_of"):
                closer_to = "correct" if ((_pr > 0) == (_exp == "right_of")) else "wrong"
            _ps = start.get("proj_right")
            if isinstance(_pr, (int, float)) and isinstance(_ps, (int, float)):
                progress_m = round(abs(_pr) - abs(_ps), 4)   # moved out to a side
    elif base_near:  # soda (2026-07-21 metric): base_near_target — goal_checker
        # already computes the unified taxonomy (nearest-can rule: correct=base
        # within tol of GT & GT nearest, wrong=within tol of a wrong can & it's
        # nearest, none=neither) plus the closer_to lean from base distances.
        # Consume its diag verbatim; no floor_drop concept for a reach-nav goal.
        outcome = base_near.get("outcome")
        closer_to = base_near.get("closer_to") if outcome == "none" else None
        min_dist_to_target = base_near.get("min_dist_to_gt")
        min_dist_to_wrong = base_near.get("min_dist_to_wrong")
        floor_drop = False
        progress_m = _progress(
            _min_start(start, "base_dist_to_candidate", base_near.get("gt_targets") or [base_near.get("gt_target")]),
            min_dist_to_target)
    elif in_layer:  # tupperware: correct=right layer, wrong=in-fridge wrong layer, none=not in fridge
        outcome = {"correct": "correct", "in_fridge_wrong": "wrong",
                   "not_in_fridge": "none"}.get(in_layer.get("outcome"), in_layer.get("outcome"))
        closer_to = None
        # DROP (Rule B): a floor-drop is a "never reached the fridge, put it on
        # the floor" failure — meaningful ONLY when nothing was placed
        # (outcome=='none'/not_in_fridge). The target ref is one of interchangeable
        # candidates, so use ANY robot-MOVED candidate on the floor (a distractor
        # falling during a correct or in_fridge_wrong placement is NOT a drop).
        _moved_z = [c.get("z") for c in (in_layer.get("candidates") or []) if c.get("moved")]
        _any_floor = any(isinstance(z, (int, float)) and z < _FLOOR_Z_M for z in _moved_z)
        floor_drop = bool(outcome == "none" and _any_floor)
        if outcome == "none":
            # soft±: use the PEAK height of the robot-MOVED tup (its attempted layer),
            # NOT any_reached/min_dist — an overshoot passes THROUGH the target band en
            # route to a higher layer, which those would misread as "reached". peak_z
            # reveals the true aimed layer: within ±half-a-layer of the target band ⇒
            # leaned correct; a clearly different layer (or never entered) ⇒ wrong.
            _HALF_LAYER_M = 0.17
            zmin, zmax = in_layer.get("z_min"), in_layer.get("z_max")
            _lr = (m.get("layer_reached") or {}).get("per_candidate") or {}
            _moved = [c.get("name") for c in (in_layer.get("candidates") or []) if c.get("moved")]
            _peaks = [_lr.get(n, {}).get("peak_z") for n in _moved]
            _peaks = [p for p in _peaks if isinstance(p, (int, float))]
            if not _moved:
                # nothing was moved: no attempt to lean either way (layer_reached
                # also tracks objects that start inside the container)
                closer_to = None
            elif _peaks and isinstance(zmin, (int, float)) and isinstance(zmax, (int, float)):
                _peak = max(_peaks)  # highest layer the moved tup attempted inside the fridge
                closer_to = "correct" if (zmin - _HALF_LAYER_M) <= _peak <= (zmax + _HALF_LAYER_M) else "wrong"
            else:
                # fallback for OLD metrics lacking peak_z: coarse any_reached
                closer_to = "correct" if (m.get("layer_reached") or {}).get("any_reached") else "wrong"
        # Moving the object (> disp_min) is the progress itself; no start state needed.
        approached = bool([c for c in (in_layer.get("candidates") or []) if c.get("moved")])
    else:  # plain REACH (touched_target): the referred object is never grasped,
        # so it can't be "dropped" — the top-level target_z<_FLOOR_Z_M check
        # misfires for reach targets that legitimately rest on the floor
        # (reach_storage_box). Clear it so `none` cases flow into the soft± lean
        # (closer_to is already computed above from min_dist_to_target/_wrong).
        floor_drop = False
        progress_m = _progress(_min_start(start, "eef_dist_to_candidate", [target_ref]), min_dist_to_target)

    if approached is None and progress_m is not None:
        approached = progress_m >= _MIN_PROGRESS_M

    return {
        "episode_unique_id": m.get("episode_unique_id"),
        "demo_id": m.get("demo_id"),
        "variant_id": m.get("variant_id"),
        "slice": m.get("slice"),
        "test_task_name": m.get("test_task_name"),
        "task_type": m.get("task_type"),
        "split_role": m.get("split_role"),
        "instruction": m.get("instruction"),
        "outcome": outcome,
        "touched_candidates": touched.get("touched_candidates"),
        "subtask_success": m.get("subtask_success"),
        "q_score_final": m.get("q_score.final"),
        "min_dist_to_target": min_dist_to_target,
        "dist_left_to_target": target_dist.get("left"),
        "dist_right_to_target": target_dist.get("right"),
        "min_dist_to_wrong": min_dist_to_wrong,
        "closer_to": closer_to,
        "floor_drop": floor_drop,
        "progress_m": progress_m,
        "approached": approached,
    }


# ── Aggregation ──────────────────────────────────────────────────────────────


def _aggregate(insts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary stats for a list of per-instance dicts.

    For "none" (no candidate touched) outcomes we want to know whether the
    EEF actually ended closer to the correct target or to a wrong candidate.
    For "correct" the EEF was in contact with the target (≤ contact threshold),
    and for "wrong" it was in contact with the wrong one — neither resolves
    near-miss vs. wrong-intent ambiguity. The interesting question is "when
    the policy reached nothing, where did it stop?".

    For the 'none' subset we report:
      - n_none_closer_to_correct / n_none_closer_to_wrong: counts per side
      - pct_none_closer_to_correct: fraction reaching toward the correct target
      - avg_min_dist_to_target_for_none / avg_min_dist_to_wrong_for_none:
        average min-arm distances to the correct target and to the nearest
        wrong candidate.
    """
    n = len(insts)
    outcomes = Counter(i.get("outcome") for i in insts if i.get("outcome"))
    success_count = sum(1 for i in insts if i.get("subtask_success"))

    # SOFT set = the non-clean outcomes whose LEAN we report: reach/tupperware ->
    # {none}; carving -> {none, on_ref} (on_ref = placed on the reference, a soft
    # success/fail whose direction lean is still informative). soft± = closer_to
    # split over this set; floor-drops (Rule B: only the not-reached 'none' side,
    # never on_ref) are pulled OUT of the leaning buckets.
    soft_insts = [i for i in insts if i.get("outcome") in ("none", "on_ref")]
    soft_floor = [i for i in soft_insts if i.get("floor_drop")]
    soft_held = [i for i in soft_insts if not i.get("floor_drop")]
    dists_tgt = [
        i["min_dist_to_target"] for i in soft_held
        if isinstance(i.get("min_dist_to_target"), (int, float))
    ]
    dists_wrong = [
        i["min_dist_to_wrong"] for i in soft_held
        if isinstance(i.get("min_dist_to_wrong"), (int, float))
    ]
    closer_counts = Counter(
        i.get("closer_to") for i in soft_held if i.get("closer_to")
    )
    # soft+ = leaned toward the instruction AND made progress from the start state.
    lean_correct = [i for i in soft_held if i.get("closer_to") == "correct"]
    n_closer_correct = sum(1 for i in lean_correct if i.get("approached") is True)
    n_lean_no_progress = sum(1 for i in lean_correct if i.get("approached") is False)
    n_lean_no_start = sum(1 for i in lean_correct if i.get("approached") is None)
    n_closer_wrong = closer_counts.get("wrong", 0)
    n_with_both = len(lean_correct) + n_closer_wrong
    n_floor_drop = len(soft_floor)

    # SUCCESS = strict `correct` only (NOT the raw subtask_success flag, which for
    # carving also counts on_ref = placed-on-the-reference). `→correct%` adds the
    # soft+ lean (correct + n_soft_closer_to_correct) — "placed correctly OR at
    # least aimed/leaned the right way" (on_ref cases are all soft+, plus any
    # none/soft outcome that leaned correct). subtask_success_* kept for reference
    # (raw BDDL goal: correct + on_ref for carving; == correct elsewhere).
    n_correct = outcomes.get("correct", 0)

    return {
        "n": n,
        "outcomes": dict(outcomes),
        "outcome_pct": {
            k: round(100 * v / n, 1) for k, v in outcomes.items()
        } if n else {},
        "correct_count": n_correct,
        "success_rate": round(n_correct / n, 3) if n else None,   # strict = correct only
        "to_correct_count": n_correct + n_closer_correct,          # correct + soft+
        "to_correct_rate": round((n_correct + n_closer_correct) / n, 3) if n else None,
        "subtask_success_count_bddl": success_count,
        "subtask_success_rate_bddl": round(success_count / n, 3) if n else None,
        # Soft stats over the {none, on_ref} subset (see comment above).
        "n_soft": len(soft_insts),
        "avg_min_dist_to_target_for_soft": (
            round(sum(dists_tgt) / len(dists_tgt), 4) if dists_tgt else None
        ),
        "avg_min_dist_to_wrong_for_soft": (
            round(sum(dists_wrong) / len(dists_wrong), 4) if dists_wrong else None
        ),
        "n_soft_with_dist": len(dists_tgt),
        "n_soft_closer_to_correct": n_closer_correct,   # soft+ (leaned correct, made progress)
        "n_soft_lean_correct_no_progress": n_lean_no_progress,   # leaned correct from where it started
        "n_soft_lean_correct_no_start_state": n_lean_no_start,   # metrics without start_state: not credited
        "n_soft_closer_to_wrong": n_closer_wrong,        # soft-
        "n_soft_floor_drop": n_floor_drop,               # drop (Rule B)
        "pct_soft_closer_to_correct": (
            round(100 * len(lean_correct) / n_with_both, 1) if n_with_both else None
        ),
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_root", required=True, type=Path,
        help="Root directory containing per-instance metrics JSON files. "
             "The script walks this tree recursively.",
    )
    parser.add_argument(
        "--output_root", default=None, type=Path,
        help="Where to write the summary files. Defaults to --metrics_root.",
    )
    parser.add_argument(
        "--task_name", default=None,
        help="Optional override; otherwise inferred from per-instance "
             "test_task_name fields.",
    )
    args = parser.parse_args()

    out_root = args.output_root or args.metrics_root
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect per-instance dicts. SKIP archive/backup dirs so stray copies of old
    # metrics can't inflate counts: any path component that starts with '_'
    # (e.g. _none_prev_metrics, _archive, _bddl_backup), contains '__' (the
    # `<task>__PRE_STOVE` / `__IR_ON_OBJECT_INVALID` archive convention), or
    # contains 'backup'. Real metrics live under '<task>/<slice>/metrics/'.
    def _is_archive_path(fp: Path) -> bool:
        for part in fp.relative_to(args.metrics_root).parts:
            low = part.lower()
            if part.startswith("_") or "__" in part or "backup" in low:
                return True
        return False

    files = sorted(args.metrics_root.rglob("*.json"))
    print(f"Scanning {len(files)} json files under {args.metrics_root}")
    instances: List[Dict[str, Any]] = []
    n_archive = 0
    for fp in files:
        if _is_archive_path(fp):
            n_archive += 1
            continue
        m = _load(fp)
        if m is None or not _is_eval_metrics_file(m):
            continue
        instances.append(_extract_instance(m))
    if n_archive:
        print(f"  (skipped {n_archive} json files under archive/backup dirs)")
    print(f"Found {len(instances)} per-instance eval metrics files")
    if not instances:
        return

    # Infer task name if not given.
    task_name = args.task_name or next(
        (i.get("test_task_name") for i in instances if i.get("test_task_name")),
        "unknown_task",
    )

    # Group by slice → list of instances; then by (slice, variant) within each.
    by_slice: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for inst in instances:
        sl = inst.get("slice") or "_unknown"
        by_slice[sl].append(inst)

    # Per-slice summary.
    by_slice_overall: Dict[str, Dict[str, Any]] = {}
    for sl, insts in by_slice.items():
        by_vid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for inst in insts:
            by_vid[inst.get("variant_id") or "_unknown"].append(inst)

        slice_summary = {
            "task_name": task_name,
            "slice": sl,
            "n_instances": len(insts),
            "overall": _aggregate(insts),
            "by_variant": {
                vid: _aggregate(vinsts) for vid, vinsts in sorted(by_vid.items())
            },
            "instances": insts,
        }
        slice_path = out_root / task_name / sl / "_summary.json"
        slice_path.parent.mkdir(parents=True, exist_ok=True)
        with open(slice_path, "w") as f:
            json.dump(slice_summary, f, indent=2)
        print(f"  wrote {sl} ({len(insts)} insts) → {slice_path}")
        by_slice_overall[sl] = slice_summary["overall"]

    # Top-level overall summary across slices.
    overall = {
        "task_name": task_name,
        "n_total_instances": len(instances),
        "overall": _aggregate(instances),
        "by_slice": by_slice_overall,
    }
    overall_path = out_root / task_name / "_summary.json"
    overall_path.parent.mkdir(parents=True, exist_ok=True)
    with open(overall_path, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"  wrote overall → {overall_path}")

    # Pretty-print to stdout: one row per slice + overall.
    def _row(label: str, s: Dict[str, Any]) -> str:
        oc = s.get("outcomes", {})
        return (
            f"{label:22s} {s['n']:>3d} "
            f"{s.get('correct_count',0):>4d} {(100*(s.get('success_rate') or 0)):4.0f}% "
            f"{(100*(s.get('to_correct_rate') or 0)):5.0f}% | "
            f"{oc.get('correct',0):>4d} {oc.get('wrong',0):>5d} {oc.get('on_ref',0):>6d} "
            f"{oc.get('none',0):>4d} | {s.get('n_soft_closer_to_correct',0):>5d} "
            f"{s.get('n_soft_closer_to_wrong',0):>5d} {s.get('n_soft_floor_drop',0):>4d}"
        )

    hdr = (f"{'slice':22s} {'n':>3s} {'corr':>4s} {'succ':>5s} {'→cor':>5s} | "
           f"{'corr':>4s} {'wrong':>5s} {'on_ref':>6s} {'none':>4s} | "
           f"{'soft+':>5s} {'soft-':>5s} {'drop':>4s}")
    print(f"\n=== {task_name}  (succ=correct only; →cor=correct+soft+) ===")
    print(hdr)
    print("-" * len(hdr))
    for sl, s in sorted(by_slice_overall.items()):
        print(_row(sl, s))
    print("-" * len(hdr))
    print(_row("OVERALL", overall["overall"]))


if __name__ == "__main__":
    main()
