"""Score point predictions on the IR grounding set (the IR columns of the paper's Table 2).

    python bench/vqa_ir/score_ir.py --data $DATA/ir --predictions preds/ [--per-item items.json] [--viz DIR]

--predictions is one JSON file, or a directory of <task>.json files, each holding a
list of records, one per query in $DATA/ir/gt/<task>.json:

    {"episode_unique_id": ..., "x_px": 313.9, "y_px": 503.3}    pixels in the 720x720 frame
    {"episode_unique_id": ..., "x_1000": 436, "y_1000": 699}    0-1000 normalized
    {"episode_unique_id": ..., "raw_output": "<model text>"}    parsed like the paper (parsing.py)

A record without a usable point, and a query without a record, both count as `none`.
prompts.py has the prompt the paper used for every model.

Per task this reports the strict and tolerant correct rates over the scored queries
(queries whose ground truth is not answerable from the frame are excluded). Tolerant
accepts a point within 2% of the image diagonal (~20 px) of the target on reach tasks;
on placement tasks the target is a region and tolerant equals strict. The headline
numbers are the unweighted means over the 14 tasks.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ir_buckets import draw_viz, score_item, tolerant_bucket  # noqa: E402
from parsing import parse_center_1000  # noqa: E402


def load_predictions(path: pathlib.Path) -> dict[str, dict]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    preds = {}
    for f in files:
        for rec in json.loads(f.read_text()):
            preds[rec["episode_unique_id"]] = rec
    return preds


def to_result(rec: dict | None, image_size) -> dict:
    """A prediction record -> the {parse_ok, x_px, y_px} the bucketing expects."""
    w, h = image_size
    x = y = None
    if rec is not None:
        if rec.get("x_px") is not None and rec.get("y_px") is not None:
            x, y = float(rec["x_px"]), float(rec["y_px"])
        elif rec.get("x_1000") is not None and rec.get("y_1000") is not None:
            x, y = rec["x_1000"] * w / 1000.0, rec["y_1000"] * h / 1000.0
        elif rec.get("raw_output") is not None:
            x1, y1 = parse_center_1000(rec["raw_output"])
            if x1 >= 0 and y1 >= 0:
                x, y = x1 * w / 1000.0, y1 * h / 1000.0
    if x is None or not (np.isfinite(x) and np.isfinite(y)):
        return {"parse_ok": False, "x_px": None, "y_px": None}
    return {"parse_ok": True, "x_px": x, "y_px": y}


def score_task(task: str, gt_entries: list[dict], preds: dict, data: pathlib.Path,
               viz_dir: pathlib.Path | None = None):
    items, n_missing = [], 0
    for gt in gt_entries:
        gt = dict(gt)
        for key in ("image_path", "mask_path"):
            if gt.get(key):
                gt[key] = str(data / gt[key])
        rec = preds.get(gt["episode_unique_id"])
        n_missing += rec is None
        res = to_result(rec, gt.get("image_size", [720, 720]))
        item = score_item(task, gt, res)
        item["episode_unique_id"] = gt["episode_unique_id"]
        items.append(item)
        if viz_dir is not None:
            draw_viz(gt, res, item, viz_dir / task / f"{gt['episode_unique_id']}.jpg")

    scored = [it for it in items if it["bucket"] != "invalid_gt"]
    n = len(scored)
    strict = sum(it["bucket"] == "correct" for it in scored) / n if n else 0.0
    if task.startswith("reach"):
        tol = 0.02 * float(np.hypot(*gt_entries[0].get("image_size", [720, 720])))
        tolerant = sum(tolerant_bucket(it, tol) == "correct" for it in scored) / n if n else 0.0
    else:
        tolerant = strict
    counts = {}
    for it in scored:
        counts[it["bucket"]] = counts.get(it["bucket"], 0) + 1
    row = {"n": n, "n_invalid_gt": len(items) - n, "n_missing": n_missing,
           "strict": strict, "tolerant": tolerant, "counts": counts}
    return row, items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path, help="the dataset's ir/ directory")
    ap.add_argument("--predictions", required=True, type=pathlib.Path, help="JSON file or directory of <task>.json")
    ap.add_argument("--tasks", help="comma-separated subset (default: every task in <data>/gt)")
    ap.add_argument("--json", type=pathlib.Path, help="also write the scores here")
    ap.add_argument("--per-item", type=pathlib.Path, help="also write every query's bucket here")
    ap.add_argument("--viz", type=pathlib.Path,
                    help="save each frame with the prediction (and the correct region on placement tasks) here")
    args = ap.parse_args()

    data = args.data.resolve()
    gt_files = sorted((data / "gt").glob("*.json"))
    tasks = args.tasks.split(",") if args.tasks else [f.stem for f in gt_files]
    preds = load_predictions(args.predictions)

    rows, per_item = {}, {}
    print(f"{'task':30s} {'n':>4s} {'strict':>7s} {'tolerant':>9s} {'missing':>8s}")
    for task in tasks:
        gt_entries = json.loads((data / "gt" / f"{task}.json").read_text())
        rows[task], per_item[task] = score_task(task, gt_entries, preds, data, args.viz)
        r = rows[task]
        print(f"{task:30s} {r['n']:>4d} {100 * r['strict']:6.1f}% {100 * r['tolerant']:8.1f}% {r['n_missing']:>8d}")

    mean = {k: sum(r[k] for r in rows.values()) / len(rows) for k in ("strict", "tolerant")}
    print(f"{'MEAN over ' + str(len(rows)) + ' tasks':30s} {'':>4s} {100 * mean['strict']:6.1f}% "
          f"{100 * mean['tolerant']:8.1f}%")
    if len(rows) != 14:
        print(f"note: the paper's numbers are means over all 14 tasks; this is {len(rows)}")
    if sum(r["n_missing"] for r in rows.values()):
        print("note: queries without a prediction were counted as `none`")

    if args.json:
        args.json.write_text(json.dumps({"tasks": rows, "mean": mean}, indent=2))
    if args.per_item:
        args.per_item.write_text(json.dumps(per_item, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
