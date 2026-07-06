"""Tabular risk model: sklearn Pipeline(impute -> scale -> LogisticRegression).

Logistic regression is used over the vendor notebook's RandomForest because its coefficients
give a direct, honest explanation for explain.py (see model choice note in the plan).
"""
from functools import lru_cache

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alz.data import FEATURE_COLUMNS
from alz.metrics import binary_metrics

LABELS = {0: "Normal", 1: "At-risk"}


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train_model(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[Pipeline, dict]:
    """Metrics come from pooled 5-fold out-of-fold predictions (n=82 is too small for a
    single honest held-out split); the returned pipeline is then refit on all the data."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_proba = cross_val_predict(build_pipeline(), X, y, cv=cv, method="predict_proba")
    metrics = binary_metrics(y, oof_proba[:, 1])

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline, metrics


def save_model(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)


@lru_cache(maxsize=None)  # ponytail: caches per model path; drop if paths change at runtime
def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def predict(pipeline: Pipeline, record_df: pd.DataFrame) -> dict:
    record_df = record_df[FEATURE_COLUMNS]
    proba = pipeline.predict_proba(record_df)[0]
    label_idx = int(proba.argmax())
    return {
        "score": float(proba[1]),
        "label": LABELS[label_idx],
    }
