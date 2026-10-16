"""One runnable check: train -> predict roundtrip works end to end on the committed sample data."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alz.data import clean, load_combined
from alz.explain import top_drivers
from alz.model import predict, train_model


def test_train_predict_roundtrip():
    df = load_combined()
    assert df["Subject ID"].str.startswith("OAS1").any() and df["Subject ID"].str.startswith("OAS2").any()
    X, y = clean(df)
    pipeline, metrics = train_model(X, y, groups=df["Subject ID"])

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["auroc"] <= 1.0

    result = predict(pipeline, X.iloc[[0]])
    assert result["label"] in ("Normal", "At-risk")
    assert 0.0 <= result["score"] <= 1.0

    drivers = top_drivers(pipeline, X.iloc[[0]])
    assert len(drivers) == 3
    assert all(d["direction"] in ("raises risk", "lowers risk") for d in drivers)
    assert all(0.0 <= d["percentile"] <= 100.0 for d in drivers)
    assert all(isinstance(d["value"], float) for d in drivers)


if __name__ == "__main__":
    test_train_predict_roundtrip()
    print("ok")
