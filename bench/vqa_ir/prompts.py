"""The single-point prompt used for every IR result in the paper.

The same wording drives the VLM sidecar in the paper's VLM-IR policy rollouts,
so the offline IR numbers measure what the policy consumes.
"""

_TARGET_DESC = {
    "object": (
        "the object that the robot must reach or manipulate to accomplish "
        "this task"
    ),
    "destination": (
        "the target location where the robot must place the object it is "
        "holding to accomplish this task"
    ),
}


# Target phrases for Molmo's native "Point to ..." prompt (noun phrases read
# better for Molmo than the JSON sidecar's clause form).
MOLMO_TARGET_DESC = {
    "object": "the object the robot must reach or manipulate for this task",
    "destination": "the location where the robot must place the object it is holding",
}


def build_prompt(instruction: str, mode: str) -> str:
    desc = _TARGET_DESC.get(mode, _TARGET_DESC["object"])
    layer_hint = ""
    if mode == "destination" and "layer" in instruction.lower():
        layer_hint = (
            'Layers are counted from bottom to top: for example, "layer 5" '
            "is the fifth layer from the bottom of the fridge.\n"
        )
    return (
        f"The robot's task is: {instruction}\n"
        f"{layer_hint}"
        f"Predict the center of the target in the frame: {desc}.\n"
        "Output the spatial coordinate of the center as x, y "
        "(integer coordinates normalized to a 0-1000 scale).\n"
        "You must provide a valid prediction for the target — do not return "
        "null or empty values.\n"
        "Predict exactly ONE point: x and y must each be a single integer, "
        "never a list. If multiple objects are plausible, commit to the "
        "single most likely one.\n"
        "Respond ONLY with valid JSON in the exact format shown below. "
        "Do not include any other text.\n\n"
        'Example: {"results": {"target": {"x": <int>, "y": <int>}}}'
    )
