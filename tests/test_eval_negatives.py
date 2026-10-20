"""Hard-negative evaluation on a synthetic labelled CSV (uses the checked-in LSTM)."""

import numpy as np
import pandas as pd

from split_step.eval_negatives import LANDMARK_NAMES, evaluate_csv
from tests.test_strokes import _body


def test_counts_grouping_and_rates(tmp_path):
    w = _body(300)                                            # 10 s standing still, fully visible
    rows = {"timestamp_ms": np.arange(300) * 33, "label": [0] * 270 + [1] * 30,
            "source_video": ["a"] * 150 + ["b"] * 150}
    for j, name in enumerate(LANDMARK_NAMES):
        for c, k in zip("xyz", range(3)):
            rows[f"{name}_world_{c}"] = w[:, j, k]
        rows[f"{name}_world_vis"] = 1.0
    path = tmp_path / "s.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    r = evaluate_csv(str(path))
    assert set(r) == {"a", "b", "ALL"}
    assert r["ALL"]["neg_frames"] == 270 and r["ALL"]["pos_frames"] == 30
    a, b, all_ = r["a"], r["b"], r["ALL"]
    assert a["pos_recall"] is None and a["neg_frames"] == 150               # video a has no positives
    assert all_["false_starts"] == a["false_starts"] + b["false_starts"]     # totals are summed counts...
    assert all_["neg_fp_rate"] == round((a["neg_on"] + b["neg_on"]) / 270, 4)  # ...and rates come from them
    assert abs(a["neg_minutes"] - 150 * 33 / 60000) < 1e-3
