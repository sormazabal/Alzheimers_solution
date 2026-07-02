"""Public integration seam: `from alz import predict, load_model`."""
import os

from alz.data import record_to_frame
from alz.explain import top_drivers
from alz.model import load_model, predict as _predict_pipeline

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "model.joblib")

__all__ = ["predict", "load_model"]


def predict(record: dict, model_path: str = _DEFAULT_MODEL_PATH) -> dict:
    """record: raw-style patient dict (see data.FEATURE_COLUMNS + 'M/F').

    Returns {'score': float, 'label': str, 'drivers': [{'feature', 'direction'}, ...]}.
    """
    pipeline = load_model(model_path)
    record_df = record_to_frame(record)
    result = _predict_pipeline(pipeline, record_df)
    result["drivers"] = top_drivers(pipeline, record_df)
    return result
