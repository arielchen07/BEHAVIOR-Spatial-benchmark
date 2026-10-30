"""Parse a single center point (0-1000 normalized) from VLM output.

Same semantics as the rollout sidecar's parser (strip <think>, first JSON
blob, results.target.{x,y} with fallbacks, regex rescue); returns (-1, -1)
on failure so parse failures can be bucketed as "none".
"""

import json
import re


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return re.sub(r".*</think>", "", text, flags=re.DOTALL).strip() or text


def _scalar(v):
    """Coerce int/float/str/[a, ...] to a float, else None."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    if isinstance(v, (list, tuple)) and v:
        return _scalar(v[0])
    return None


def parse_center_1000(raw: str) -> tuple:
    """Return (x, y) on the 0-1000 scale, or (-1, -1) if unparseable."""
    text = strip_thinking(raw or "").strip()
    text = re.sub(r"```json\s*|```\s*", "", text)
    # Molmo point-pair quirk: {"x": 826 642} — two space-separated numbers and
    # no "y" key. Rewrite to {"x": 826, "y": 642} so the JSON path works.
    text = re.sub(r'"x"\s*:\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*}',
                  r'"x": \1, "y": \2}', text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        blob = match.group()
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            # Models often hit EOS mid-object, leaving unbalanced braces
            # (e.g. '{"results": {"target": {"x": [439, 690], "y": [690]}}').
            try:
                parsed = json.loads(_balance_braces(blob))
            except json.JSONDecodeError:
                parsed = None
        if isinstance(parsed, dict):
            results = parsed.get("results", parsed)
            target = None
            if isinstance(results, dict):
                if "target" in results:
                    target = results["target"]
                elif "x" in results and "y" in results:
                    target = results
                elif results:
                    first = next(iter(results.values()))
                    target = first
            if isinstance(target, dict):
                x, y = _scalar(target.get("x")), _scalar(target.get("y"))
                # Qwen quirk: {"x": [a, b]} with no y
                if y is None and isinstance(target.get("x"), (list, tuple)) \
                        and len(target["x"]) >= 2:
                    x, y = _scalar(target["x"][0]), _scalar(target["x"][1])
                if x is not None and y is not None:
                    return (x, y)
            elif isinstance(target, (list, tuple)) and len(target) >= 2:
                x, y = _scalar(target[0]), _scalar(target[1])
                if x is not None and y is not None:
                    return (x, y)

    # Regex rescue for list-valued coords: "x": [a, b], "y": [c] -> (a, b),
    # and "x": [a], "y": [b] -> (a, b). First element of x is the x coordinate;
    # y comes from x[1] if present else the y list.
    mxl = re.search(r'"x"\s*:\s*\[\s*(\d+(?:\.\d+)?)(?:\s*,\s*(\d+(?:\.\d+)?))?', text)
    myl = re.search(r'"y"\s*:\s*\[\s*(\d+(?:\.\d+)?)', text)
    if mxl:
        x = float(mxl.group(1))
        if myl:
            return (x, float(myl.group(1)))
        if mxl.group(2):
            return (x, float(mxl.group(2)))

    # Regex rescue: "x": N ... "y": M
    mx = re.search(r'"x"\s*:\s*(\d+(?:\.\d+)?)', text)
    my = re.search(r'"y"\s*:\s*(\d+(?:\.\d+)?)', text)
    if mx and my:
        return (float(mx.group(1)), float(my.group(1)))
    # Qwen3.5-9B quirk: {"x": 883, 533} — second bare number is y
    if mx:
        mq = re.search(r'"x"\s*:\s*\d+(?:\.\d+)?\s*,\s*(\d+(?:\.\d+)?)', text)
        if mq:
            return (float(mx.group(1)), float(mq.group(1)))

    return (-1.0, -1.0)


def _balance_braces(s: str) -> str:
    """Append missing closing brackets/braces to a truncated JSON fragment,
    ignoring brackets inside strings, and drop any dangling trailing comma."""
    stack, in_str, esc = [], False, False
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    out = s.rstrip().rstrip(",")
    for opener in reversed(stack):
        out += "}" if opener == "{" else "]"
    return out
