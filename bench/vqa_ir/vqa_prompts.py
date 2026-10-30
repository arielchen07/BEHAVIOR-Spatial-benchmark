"""The paper's VQA prompt and images for any released question.

    from vqa_prompts import iter_questions
    for subset, record, prompt, images in iter_questions("$DATA/vqa"):
        answer = my_model(images, prompt)          # images: absolute paths, in order

`prompt` is exactly the text every model in the paper received (vqa_prompt_rules.py
holds the rules it is built from), and `images` the image(s) shown with it:

  general/single_frame             one frame; the copy with the bounding box drawn
                                   when the question refers to a boxed object
  general/multi_frame/<subset>     the frame sequence (rotation or navigation)
  task/manipulation/<task>         one frame
  task/navigation/<task>           the context-tour frames, then the current view last

Score the answers with score_vqa.py. Print one prompt with
    python bench/vqa_ir/vqa_prompts.py --data $DATA/vqa --question-id <id>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vqa_prompt_rules import build_prompt  # noqa: E402


def prompt_args(record: dict, subset: str) -> dict:
    """The build_prompt() arguments the paper's runs used for this record."""
    if subset == "general/single_frame":
        # Category and type are encoded in the question id: locate_ / direction_ /
        # distance_ (…objcamera… = distance to the camera) / size_.
        prefix = record["question_id"].split("_", 1)[0]
        if prefix == "locate":
            category, qtype = "direction", "locate"
        elif prefix == "distance" and "objcamera" in record["question_id"]:
            category, qtype = "distance", "camera"
        else:
            category, qtype = prefix, "object_object"
        return dict(question_category=category, question_type=qtype,
                    is_multi_frame=False, dataset_type=None)
    if subset.startswith("general/multi_frame/"):
        return dict(question_category=record["question_category"],
                    question_type=record["relationship_type"],
                    is_multi_frame=True, dataset_type=subset.rsplit("/", 1)[1])
    if subset.startswith("task/"):
        return dict(question_category=record.get("question_category"),
                    question_type=record.get("question_type"),
                    is_multi_frame=bool(record.get("image_paths")),
                    dataset_type="move_to_context" if record.get("question_type") == "move_to_context" else None)
    raise ValueError(f"unknown VQA subset {subset!r}")


def vqa_prompt(record: dict, subset: str) -> str:
    return build_prompt(record["question"], record["options"], **prompt_args(record, subset))


def images(record: dict, subset: str, base: pathlib.Path) -> list[pathlib.Path]:
    """Absolute paths of the image(s) to show, in order (`base` = the questions.json directory)."""
    if record.get("image_paths"):
        return [base / p for p in record["image_paths"]]
    if subset == "general/single_frame" and record.get("use_annotated_image"):
        return [base / record["image_path_2d"]]
    return [base / record["image_path"]]


def iter_questions(vqa_dir) -> Iterator[tuple[str, dict, str, list[pathlib.Path]]]:
    """(subset, record, prompt, image paths) for every released VQA question."""
    vqa_dir = pathlib.Path(vqa_dir).resolve()
    for qfile in sorted(vqa_dir.rglob("questions.json")):
        rel = qfile.parent.relative_to(vqa_dir).as_posix()
        subset = rel.rsplit("/", 1)[0] if rel.startswith("task/") else rel
        for record in json.loads(qfile.read_text()):
            yield subset, record, vqa_prompt(record, subset), images(record, subset, qfile.parent)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=pathlib.Path, help="the dataset's vqa/ directory")
    ap.add_argument("--question-id", required=True)
    args = ap.parse_args()
    for subset, record, prompt, imgs in iter_questions(args.data):
        if record["question_id"] == args.question_id:
            print(f"[{subset}] images: " + ", ".join(str(p) for p in imgs) + "\n")
            print(prompt)
            return 0
    print(f"no question {args.question_id} under {args.data}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
