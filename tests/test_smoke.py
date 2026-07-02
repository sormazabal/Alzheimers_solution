"""One runnable check: train -> predict roundtrip works end to end on the committed sample data."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alz.data import clean, load_raw
from alz.explain import top_drivers
from alz.model import predict, train_model

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "oasis_longitudinal.csv")


def test_train_predict_roundtrip():
    df = load_raw(DATA_PATH)
    X, y = clean(df)
    pipeline, accuracy = train_model(X, y)

    assert 0.0 <= accuracy <= 1.0

    result = predict(pipeline, X.iloc[[0]])
    assert result["label"] in ("Normal", "At-risk")
    assert 0.0 <= result["score"] <= 1.0

    drivers = top_drivers(pipeline, X.iloc[[0]])
    assert len(drivers) == 3
    assert all(d["direction"] in ("raises risk", "lowers risk") for d in drivers)


if __name__ == "__main__":
    test_train_predict_roundtrip()
    print("ok")
