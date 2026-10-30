# Sim-eval metrics

`score.py` reports the paper's numbers. The outcome buckets behind them are defined
in `summarize_eval_results.py` (`_extract_instance`, `_aggregate`) and the goal
predicates in `goal_checker.py`; this page describes what those files do. Score
results with the scripts rather than re-implementing the buckets, so every method is
scored by the same rules:

```
python bench/sim_eval/score.py --results <out>                          # paper metrics
python bench/sim_eval/summarize_eval_results.py --metrics_root <out>    # full bucket breakdown
```

Every episode writes `<out>/<task>[_ir]/<slice>/metrics/<instance>.json`. The family
of an episode is set by the goal condition its instance declares
(`goal_conditions[].type` in the instance JSON), not by the task name.

## Manipulation: one hard outcome per episode

| family | goal condition | tasks |
|---|---|---|
| reach by contact | `touched_target` | reach_dice, reach_food_item, reach_hardback_from_bookcase, reach_item, reach_paperback_book_aug, reach_pizza_plate, reach_plate, reach_storage_box, reach_tool, reach_toys |
| reach by base position | `base_near_target` | reach_soda_can |
| place by direction | `left_of` / `right_of` | place_carving_knife |
| place in a layer | `in_layer` | place_tupperware_in_fridge, place_bowl_in_cabinet |

**Reach by contact.** The episode ends when the hand has stayed in contact with a
candidate for 30 consecutive steps while nearly stationary (moving less than 3 cm),
or at the step budget. For reach_soda_can and reach_storage_box only a dwell on the
instructed target ends the episode early (`protocol.yaml`). The outcome is read from
what the robot touches at the end:

| bucket | rule |
|---|---|
| `correct` | touching an instructed target (any of them when several tie, e.g. two equally small items) |
| `wrong` | touching only non-target candidates |
| `none` | touching no candidate |

**Reach by base position** (floor cans spread across a room, where the deciding
behavior is driving to the right one). At the end of the rollout:

| bucket | rule |
|---|---|
| `correct` | the base is within 0.70 m of an instructed can and that can is the candidate nearest the base, or the robot touches an instructed can |
| `wrong` | the base is within 0.70 m of a non-target can and it is the nearest candidate |
| `none` | neither |

**Place by direction** (carving knife, starts in hand):

| bucket | rule |
|---|---|
| `correct` | knife released, resting on the countertop, at least 0.17 m from the reference's center, on the instructed side |
| `wrong` | the same, on the opposite side |
| `on_ref` | released, but resting on something other than the countertop or within 0.17 m of the reference (put down in place, no direction expressed) |
| `none` | never released |

The side is the sign of the knife's offset along the reference's right axis. A knife
released off the reference but laterally centered still gets the sign of a small
offset; this edge case is not guarded.

**Place in a layer** (tupperware in the fridge, bowl in the cabinet; both start in
hand). The placed object is any same-category candidate the robot moved by more than
0.10 m; objects already in the container are excluded.

| bucket | rule |
|---|---|
| `correct` | a moved candidate rests inside the container within the instructed layer's height band (±0.05 m) |
| `wrong` | it rests inside the container in another layer |
| `none` | no moved candidate is inside the container |

## Manipulation: the reported rates

| metric | definition |
|---|---|
| strict success | `correct` / all episodes |
| soft success (primary) | (`correct` + soft episodes that leaned toward the instruction and made progress) / all episodes |
| wrong-target rate | `wrong` / all episodes |

The soft episodes are `none`, plus `on_ref` for the carving knife. Each one counts
toward soft success when it both **leans toward the instruction** and **made
progress** from its start state. The start state is recorded in each metrics file
(`start_state`) right before the first policy action.

| family | leans toward the instruction when, at the end of the rollout | made progress when |
|---|---|---|
| reach by contact | the hand nearer to an instructed target is closer to it than any hand is to another candidate | that distance shrank by at least 5 cm from the start |
| reach by base position | the base is closer to an instructed can than to any other can | the base distance to it shrank by at least 5 cm |
| carving knife | the knife (held or released) is on the instructed side of the reference | its sideways offset from the reference grew by at least 5 cm |
| layers | the highest point the moved object reached inside the container is within half a layer (0.17 m) of the instructed band | a candidate was moved by more than 0.10 m |

The progress requirement keeps a policy from earning soft success for where an
episode happened to start: a policy that holds still scores the same soft and strict
success. The 5 cm margin is just above how far a stationary robot's hands drift
during an episode (90th percentile about 5 cm). For layers, the peak height is used
rather than the closest approach, because an object lifted to a higher layer passes
through the instructed band on the way. Metrics files without `start_state` (written
by an earlier harness) never count as making progress, and `score.py` prints how
many leans that left uncredited.

A soft episode in which the held object (carving knife, tupperware, bowl) ends on
the floor (below z = 0.30 m) is a drop and never counts toward soft success. Reach
targets are never held, so reach episodes have no drops.

`subtask_success` in the metrics file is the raw BDDL goal and is more permissive: for
the carving knife it is also true for `on_ref`. The rates above do not use it.

## Navigation

Success is `subtask_success`, recorded at any step of the rollout (arriving and then
leaving still counts):

| goal condition | tasks | success when |
|---|---|---|
| `in_room` | move_to_bathroom, move_to_garage | the base enters the target room, or comes within 1.3 m of an opened door of that room |
| `near_object` | the other 8 | the base comes within `tol_m` of the target object (1.5 m unless the instance sets it); extended objects such as the bar counter are measured to their footprint, not their center |

Each navigation instance carries its own step budget (`max_steps`), which
`run_eval.py` applies per instance.
