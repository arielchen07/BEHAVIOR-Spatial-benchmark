"""Score multiple-choice answers on the VQA sets (the paper's VQA tables).

    python bench/vqa_ir/score_vqa.py --data $DATA/vqa --predictions answers.json [--json scores.json]

--predictions maps every question_id (unique across all VQA sets) to the model's
answer: either {"<question_id>": "<answer>", ...} or a list of
{"question_id": ..., "answer": ...} records, in one file or a directory of files.
An answer is an option letter ("B"), the letter-and-text form the prompt asks for
("B.front-right"), or an option's text ("front-right"); anything else, and a question
without an answer, is wrong. Only the sets present in the
predictions are reported.

Reported, as accuracy over all questions in the group (4-way choice, chance 25%):
  general / single-frame   All, Direction (incl. locate), Distance, Size
  general / multi-frame    All, Direction, Distance, Size, and the rotation / navigation subsets
  task / manipulation      pooled over the 14 tasks, per task, per question category
  task / navigation        pooled over the 10 tasks, per task
Some task questions have more than one correct option (ties); any of them counts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict


def extract_letter_answer(answer: str, options: str):
    """The paper's answer matching (an option letter, or text matched to an option)."""
    if answer is None:
        return None

    # Check if final_answer is already a single letter option
    answer_clean = answer.strip()
    if len(answer_clean) == 1 and answer_clean.isalpha():
        return answer_clean.upper()

    # If it's text, try to find the matching option and extract the letter
    answer_text = answer_clean.lower().strip()

    # Parse options (format: "A: text, B: text, C: text, D: text")
    option_parts = [opt.strip() for opt in options.split(",")]

    for option in option_parts:
        # Extract letter and text from option (e.g., "A: front" -> letter="A", text="front")
        if ":" in option:
            letter, text = option.split(":", 1)
            letter = letter.strip().upper()
            text = text.strip().lower()

            # Check if the answer text matches this option's text (exact match preferred)
            if answer_text == text:
                return letter
            # Fallback: check if answer is contained in option text or vice versa
            elif answer_text in text or text in answer_text:
                return letter


def options_str(q: dict) -> str:
    opts = q.get("options_dict") or q["options"]
    if isinstance(opts, dict):
        return ", ".join(f"{k}: {v}" for k, v in opts.items())
    return opts


_LEADING_LETTER = re.compile(r"^\s*\(?([A-Da-d])\s*[.):]")


def is_correct(q: dict, answer) -> bool:
    if not isinstance(answer, str):
        return False
    m = _LEADING_LETTER.match(answer)  # "C.front", "C: front", "(C) front"
    letter = m.group(1).upper() if m else extract_letter_answer(answer, options_str(q))
    return letter is not None and letter in (q.get("gt_answers") or [q["gt_answer"]])


def single_frame_category(q: dict) -> str:
    # The category is the question-id prefix: locate_ / direction_ / distance_ / size_.
    prefix = q["question_id"].split("_", 1)[0]
    return "direction" if prefix == "locate" else prefix


def load_predictions(path: pathlib.Path) -> dict:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    preds = {}
    for f in files:
        data = json.loads(f.read_text())
        if isinstance(data, dict):
            preds.update(data)
        else:
            preds.update({r["question_id"]: r.get("answer") for r in data})
    return preds


def acc(rows) -> dict:
    return {"n": len(rows), "accuracy": sum(rows) / len(rows) if rows else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path, help="the dataset's vqa/ directory")
    ap.add_argument("--predictions", required=True, type=pathlib.Path, help="answers: JSON file or directory")
    ap.add_argument("--json", type=pathlib.Path, help="also write the scores here")
    args = ap.parse_args()

    data = args.data.resolve()
    preds = load_predictions(args.predictions)
    report = {}

    def questions(rel: str):
        return json.loads((data / rel / "questions.json").read_text())

    def answered(qs) -> bool:
        return any(q["question_id"] in preds for q in qs)

    def show(title: str, groups: dict) -> None:
        print(f"\n{title}")
        for name, g in groups.items():
            print(f"  {name:28s} {g['n']:>5d} {100 * g['accuracy']:6.1f}%")

    sf = questions("general/single_frame")
    if answered(sf):
        by = defaultdict(list)
        for q in sf:
            ok = is_correct(q, preds.get(q["question_id"]))
            by["All"].append(ok)
            by[single_frame_category(q).capitalize()].append(ok)
        groups = {k: acc(by[k]) for k in ("All", "Direction", "Distance", "Size")}
        report["general/single_frame"] = groups
        show("General VQA — single-frame", groups)

    mf = {s: questions(f"general/multi_frame/{s}") for s in ("rotation", "navigation")}
    if any(answered(qs) for qs in mf.values()):
        by = defaultdict(list)
        for subset, qs in mf.items():
            for q in qs:
                ok = is_correct(q, preds.get(q["question_id"]))
                by["All"].append(ok)
                by[q["question_category"].capitalize()].append(ok)
                by[subset].append(ok)
        groups = {k: acc(by[k]) for k in ("All", "Direction", "Distance", "Size", "rotation", "navigation")}
        report["general/multi_frame"] = groups
        show("General VQA — multi-frame", groups)

    for kind in ("manipulation", "navigation"):
        task_dirs = sorted(d for d in (data / "task" / kind).iterdir() if d.is_dir())
        per_task = {d.name: questions(f"task/{kind}/{d.name}") for d in task_dirs}
        if not any(answered(qs) for qs in per_task.values()):
            continue
        pooled, by_cat, groups = [], defaultdict(list), {}
        for task, qs in per_task.items():
            oks = [is_correct(q, preds.get(q["question_id"])) for q in qs]
            pooled += oks
            for q, ok in zip(qs, oks):
                by_cat[q.get("question_category", "unknown")].append(ok)
            groups[task] = acc(oks)
        groups = {"POOLED": acc(pooled), **groups}
        if kind == "manipulation":
            groups.update({f"[{c}]": acc(v) for c, v in sorted(by_cat.items())})
        report[f"task/{kind}"] = groups
        show(f"Task VQA — {kind}", groups)

    if not report:
        print(f"No predictions matched any question under {data}")
        return 1
    n_q = sum(len(json.loads(p.read_text())) for p in data.rglob("questions.json"))
    print(f"\n{len(preds)} answers for {n_q} questions; unanswered questions count as wrong.")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
