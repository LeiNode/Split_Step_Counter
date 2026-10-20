"""Practice plan: every tip the analyses can emit maps to a drill (or is filming advice / praise)."""

import ast
import pathlib

from split_step.coach import _POSITIVE
from split_step.practice import drill_for, practice_plan

PKG = pathlib.Path(__file__).resolve().parent.parent / "split_step"
TIP_MODULES = ["tennis_metrics.py", "strokes.py", "timing.py", "rallies.py", "ready_position.py",
               "footwork.py", "fatigue.py", "camera_view.py"]
NO_DRILL = ("filmed", "film from behind", "out of view", "no swings detected")


def _text(node) -> str:
    """Literal text of a tip expression; formatted values become {}."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(_text(v) if isinstance(v, ast.Constant) else "{}" for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _text(node.left) + _text(node.right)
    return "{}"


def _all_tips() -> list[str]:
    tips = []
    for name in TIP_MODULES:
        for node in ast.walk(ast.parse((PKG / name).read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append" and node.args):
                target = ast.unparse(node.func.value)
                if target.endswith("tips") or target.endswith("['tips']"):
                    tips.append(_text(node.args[0]))
    return tips


def test_every_work_tip_has_a_drill():
    tips = _all_tips()
    assert len(tips) > 20                                     # the scan found them
    work = [t for t in tips if not any(p in t.lower() for p in _POSITIVE)]
    missing = [t for t in work if drill_for(t) is None and not any(p in t.lower() for p in NO_DRILL)]
    assert not missing, missing


def test_recovery_speed_tip_gets_its_own_drill():
    tip = "Recover faster: you take about 2.1 s to get back to a ready position after each shot."
    assert drill_for(tip)["name"] == "Two-ball recovery"


def test_plan_order_dedup_and_skips():
    focus = ["Most of this video shows the player from the side. ... film from behind the baseline.",
             "Only 2 of 9 shots had a split step before them. Make it a habit.",
             "There was a 12 s stretch without a split step. Aim to split step before every opponent shot.",
             "Forehand: keep your head still through contact (it moves ~20 cm)."]
    plan = practice_plan(focus)
    assert [d["name"] for d in plan] == ["Split on the feed", "Watch the strings"]
    assert plan[0]["for"].startswith("Only 2 of 9") and plan[0]["dose"]
