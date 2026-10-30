# VQA and IR scorers

Both scorers read the released data (`$DATA` = the root of the Hugging Face dataset)
and a predictions file from your model. They need only numpy (and Pillow + OpenCV
for the IR masks); no model code is involved, so any VLM can be scored.

## VQA: BEHAVIOR-Spatial-VQA (Understand)

Run your model on every question in `$DATA/vqa/**/questions.json` (4-way multiple
choice). To reproduce the paper's setup, use the same prompt and images every model
got; `vqa_prompts.py` builds both for any released record:

```python
from vqa_prompts import iter_questions
for subset, record, prompt, images in iter_questions(f"{DATA}/vqa"):
    answers[record["question_id"]] = my_model(images, prompt)
```

The prompt (`vqa_prompt_rules.py`) adds, depending on the question: how to read
directions from the camera's viewpoint, that "the camera" is the viewpoint and not
the visible robot hand, how grid cells, layers and levels are counted (from the
bottom; layer 1 is the lowest), the robot-centric frame and context-tour layout
for navigation questions, and the answer format ("C.front"). To see one prompt:
`python bench/vqa_ir/vqa_prompts.py --data $DATA/vqa --question-id <id>`.

Save one answer per `question_id`:

```json
{"locate_task23_demo231930_frame13442_s94_q0_t0": "B", "...": "front-right"}
```

An answer is an option letter, the "B.front-right" form the prompt asks for, or an option's text. Then:

```bash
python bench/vqa_ir/score_vqa.py --data $DATA/vqa --predictions answers.json
```

Categories come from the question itself: single-frame questions by their
`question_id` prefix (`locate_` and `direction_` both count as Direction, then
`distance_`, `size_`), the others by `question_category`. These are the paper's
groupings, and the dataset card's single-frame breakdown uses the same ones.

It reports general single-frame (All / Direction / Distance / Size), general
multi-frame (All / Direction, plus Distance, Size and the rotation / navigation
subsets), and task-grounded VQA for manipulation (pooled over 1,082 questions,
per task, per category) and navigation (pooled over 602, per task). Only sets you
answered are reported; unanswered questions count as wrong.

## IR: BEHAVIOR-Spatial-IR (Ground)

For each query in `$DATA/ir/gt/<task>.json`, show the model the frame
(`image_path`, 720×720) and ask for one point on the target: the object to reach,
or for placement tasks the place to put the held object.

On reach tasks (`mode: "object"`) `gt_target` is the object to point at. On
placement tasks (`mode: "destination"`) `gt_target` is the held object, and the
answer is a region computed from the record's geometry: the countertop on the
instructed side of `reference_instance` (carving knife), or the instructed layer
`compartment_band` of `container_instance` (tupperware, bowl). A box center is
therefore not a valid point on placement tasks. `--viz DIR` saves each frame with
the prediction and, on placement tasks, the region that scores as correct shaded
green. `prompts.py` has the
prompt used for every model in the paper (`build_prompt(instruction, mode)`). Save
one record per query, in pixels, 0–1000 normalized coordinates, or the raw model
text:

```json
[{"episode_unique_id": "reach_plate_test_111010_0_f3071", "x_px": 313.9, "y_px": 503.3}]
```

```bash
python bench/vqa_ir/score_ir.py --data $DATA/ir --predictions preds/   # dir of <task>.json, or one file
python bench/vqa_ir/score_ir.py --data $DATA/ir --predictions preds/ --tasks place_carving_knife --viz viz/
```

It reports, per task and as the unweighted mean over the 14 tasks, the strict
correct rate (the point lands on the instructed target) and the tolerant rate
(within ~20 px on reach tasks; placement targets are regions, so tolerant = strict).
`ir_buckets.py` defines the buckets; `--per-item` writes every query's bucket.

## Checked against the paper

Run on the released data with the paper's stored model outputs, these scorers give:

| | paper value | reproduced |
|---|---|---|
| general VQA, Qwen3.5-4B (single-frame All / Dir / Dist / Size; multi-frame All / Dir) | 46.8 / 44.8 / 48.1 / 51.8; 46.6 / 44.1 | exact |
| task VQA, manipulation pooled (Qwen3.5-9B, Molmo2-4B, Molmo2-8B, Qwen3.5-4B, Gemma 3-4B) | 57.2, 53.0, 50.0, 49.4, 33.5 | exact |
| task VQA, navigation pooled (Qwen3.5-4B) | 30.2 | exact |
| IR strict / tolerant (same five models) | 44.8/49.4, 26.8/39.5, 44.6/52.2, 38.6/46.6, 34.1/44.1 | exact |
