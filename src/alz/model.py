"""Tabular risk model: sklearn Pipeline(impute -> scale -> LogisticRegression).

Logistic regression is used over the vendor notebook's RandomForest because its coefficients
give a direct, honest explanation for explain.py (see model choice note in the plan).
"""
from functools import lru_cache

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alz.data import FEATURE_COLUMNS

LABELS = {0: "Normal", 1: "At-risk"}


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train_model(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[Pipeline, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=random_state, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    return pipeline, accuracy


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
