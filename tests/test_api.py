"""One runnable check: the FastAPI /predict seam works end to end (requires a trained model)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_roundtrip():
    record = {
        "Age": 75, "EDUC": 12, "SES": 3, "MMSE": 27,
        "nWBV": 0.75, "ASF": 1.2, "Visit": 1,
        "MR Delay": 0, "M/F": "F",
    }
    result = client.post("/predict", json=record).json()

    assert result["label"] in ("Normal", "At-risk")
    assert 0.0 <= result["score"] <= 1.0
    assert len(result["drivers"]) == 3


if __name__ == "__main__":
    test_health()
    test_predict_roundtrip()
    print("ok")
