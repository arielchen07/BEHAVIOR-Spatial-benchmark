"""Namespace shim for the OmniGibson 3.7.x -> 3.9.x eval-module rename.

OmniGibson >=3.9 renamed ``omnigibson.learning`` to ``omnigibson.eval`` and folded
``RGBLowResWrapper`` into ``DefaultWrapper``. The two wrappers are equivalent for
our purposes: both force every robot camera to 224x224 RGB. 3.7.x additionally set
the head camera's horizontal aperture to 40.0 inside the wrapper; 3.9.x applies the
same value from ``Evaluator._apply_robot_eval_settings`` via
``EVAL_HEAD_HORIZONTAL_APERTURE = 40.0``, so observations match either way.

Hydra ``_target_`` entries in configs/ point here so one config tree drives both
installs.
"""

import importlib.util

# Dispatch on which namespace EXISTS rather than catching ImportError: a missing
# transitive dep (e.g. msgpack) would otherwise be mistaken for "old OmniGibson"
# and produce a confusing second failure from the fallback branch.
if importlib.util.find_spec("omnigibson.eval") is not None:  # OmniGibson >= 3.9
    from omnigibson.eval.policies import LocalPolicy, WebsocketPolicy
    from omnigibson.eval.wrappers import DefaultWrapper as EvalRGBWrapper
else:  # OmniGibson 3.7.x
    from omnigibson.learning.policies import LocalPolicy, WebsocketPolicy
    from omnigibson.learning.wrappers import RGBLowResWrapper as EvalRGBWrapper

__all__ = ["EvalRGBWrapper", "LocalPolicy", "WebsocketPolicy"]
