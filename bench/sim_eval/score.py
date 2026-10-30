"""Report the paper's sim metrics for a directory of eval results.

Walks every per-episode metrics JSON under --results, groups episodes by slice (the
directory run_eval.py wrote them to, <out>/<task>[_ir]/<slice>/metrics/) and test task,
and reports:

  manipulation  strict success = correct
                soft success   = correct + non-committed episodes that leaned toward
                                 the instructed target and got at least 5 cm closer
                                 to it than at the start (no floor drops; METRICS.md)
                wrong-target   = commitments to a non-instructed candidate/location
  navigation    success        = the recorded binary goal result (subtask_success)

Per-task rows plus a pooled row per slice; pooled rates are over all episodes, not
averaged over tasks. Runs made with run_eval.py --with-ir (<task>_ir/ directories) use
privileged simulator state and are reported as a separate "<slice> [privileged IR]"
block, never pooled with the others. Outcome buckets come from summarize_eval_results.py (see
METRICS.md); this script only counts them.

    python bench/sim_eval/score.py --results runs/my_method [--json scores.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from summarize_eval_results import _aggregate, _extract_instance, _is_eval_metrics_file, _load  # noqa: E402


def is_archive(rel: pathlib.Path) -> bool:
    # Same skip rule as summarize_eval_results.py.
    return any(p.startswith("_") or "__" in p or "backup" in p.lower() for p in rel.parts)


def collect(root: pathlib.Path):
    manip, nav = defaultdict(list), defaultdict(list)
    for fp in sorted(root.rglob("*.json")):
        if is_archive(fp.relative_to(root)):
            continue
        m = _load(fp)
        if m is None or not _is_eval_metrics_file(m):
            continue
        slice_name = fp.parent.parent.name
        if fp.parent.parent.parent.name.endswith("_ir"):
            slice_name += " [privileged IR]"
        key = (slice_name, m.get("test_task_name") or "unknown")
        if m.get("task_type") == "nav":
            nav[key].append(bool(m.get("subtask_success")))
        else:
            manip[key].append(_extract_instance(m))
    return manip, nav


def manip_row(insts) -> dict:
    agg = _aggregate(insts)
    n = agg["n"]
    wrong = agg["outcomes"].get("wrong", 0)
    soft = agg["correct_count"] + agg["n_soft_closer_to_correct"]
    return {"n": n, "strict": agg["correct_count"] / n, "soft": soft / n, "wrong": wrong / n,
            "no_start_state": agg["n_soft_lean_correct_no_start_state"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=pathlib.Path, help="results root (e.g. run_eval.py --out)")
    ap.add_argument("--json", type=pathlib.Path, help="also write the scores here")
    args = ap.parse_args()

    manip, nav = collect(args.results.resolve())
    if not manip and not nav:
        print(f"No eval metrics found under {args.results}")
        return 1
    report = {"manipulation": {}, "navigation": {}}

    for sl in sorted({s for s, _ in manip}):
        tasks = sorted(t for s, t in manip if s == sl)
        rows = {t: manip_row(manip[(sl, t)]) for t in tasks}
        rows["POOLED"] = manip_row([i for t in tasks for i in manip[(sl, t)]])
        report["manipulation"][sl] = rows
        print(f"\nManipulation — {sl}")
        print(f"  {'task':30s} {'n':>4s} {'strict':>7s} {'soft':>7s} {'wrong':>7s}")
        for t, r in rows.items():
            print(f"  {t:30s} {r['n']:>4d} {100 * r['strict']:6.1f}% {100 * r['soft']:6.1f}% {100 * r['wrong']:6.1f}%")

    for sl in sorted({s for s, _ in nav}):
        tasks = sorted(t for s, t in nav if s == sl)
        rows = {t: {"n": len(nav[(sl, t)]), "success": sum(nav[(sl, t)]) / len(nav[(sl, t)])} for t in tasks}
        pooled = [x for t in tasks for x in nav[(sl, t)]]
        rows["POOLED"] = {"n": len(pooled), "success": sum(pooled) / len(pooled)}
        report["navigation"][sl] = rows
        print(f"\nNavigation — {sl}")
        print(f"  {'task':30s} {'n':>4s} {'success':>8s}")
        for t, r in rows.items():
            print(f"  {t:30s} {r['n']:>4d} {100 * r['success']:7.1f}%")

    missing = sum(r["no_start_state"] for rows in report["manipulation"].values()
                  for t, r in rows.items() if t != "POOLED")
    if missing:
        print(f"\nnote: {missing} episode(s) leaned toward the target but their metrics have no "
              f"start_state (written by an older harness), so soft success does not credit them. "
              f"Re-run them with this harness.")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
