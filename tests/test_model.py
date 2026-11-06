"""The shipped artifact loads and scores every known series."""
import sys
from pathlib import Path

import joblib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from app import SERIES, build_prediction_row, MODEL

ROOT = Path(__file__).resolve().parent.parent


def test_artifact_has_expected_keys():
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    assert set(bundle) >= {"model", "feature_columns", "spike_threshold"}
    assert bundle["spike_threshold"] == 0.02


@pytest.mark.parametrize("s", SERIES[:8] + SERIES[-8:])
def test_model_scores_series(s):
    x, row, target_month = build_prediction_row(s["market"], s["commodity"])
    proba = float(MODEL.predict_proba(x)[0, 1])
    assert 0.0 <= proba <= 1.0
    assert target_month > row["month"] - pd.offsets.MonthBegin(1)


import pandas as pd  # noqa: E402

def test_template_uses_existing_metric_keys():
    """Regression: the UI must only reference keys present in metrics.json
    (v4 renamed accuracy/f1 keys; a bare cm.accuracy rendered 'undefined')."""
    import json, re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    metrics = json.loads((root / "models" / "metrics.json").read_text())["retrospective_eval"]
    html = (root / "templates" / "index.html").read_text()
    for key in re.findall(r"\$\{cm\.([a-zA-Z0-9_]+)\}", html):
        assert key in metrics, f"template references missing metrics key: {key}"
