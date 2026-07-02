"""OASIS-2 loading + cleaning, adapted from vendor/alzheimers-disease-prediction/dimentia_pred.ipynb."""
import os
from functools import lru_cache

import pandas as pd

REQUIRED_COLUMNS = [
    "Subject ID", "MRI ID", "Group", "Visit", "MR Delay", "M/F", "Hand",
    "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF",
]

# CDR (Clinical Dementia Rating) is dropped as a predictor even though the vendor notebook
# keeps it: CDR is itself the clinical diagnosis, so using it to predict Group would leak
# the answer into a tool meant to screen patients *before* that diagnosis exists.
FEATURE_COLUMNS = ["Visit", "MR Delay", "Age", "EDUC", "SES", "MMSE", "nWBV", "ASF", "Sex_M"]

# Human-readable labels + clinical context, for hover tooltips in the app. Ranges reflect the
# widget bounds already used in app/streamlit_app.py plus standard OASIS/MMSE conventions.
FEATURE_META = {
    "Visit": {"label": "Visit number", "definition": "Which study visit this record is from.", "range": "1-10"},
    "MR Delay": {"label": "MRI delay", "definition": "Days between clinical visit and MRI scan.", "range": "0+ days"},
    "Age": {"label": "Age", "definition": "Patient age in years.", "range": "18-110"},
    "EDUC": {"label": "Years of education", "definition": "Total years of formal education completed.", "range": "0-30"},
    "SES": {"label": "Socioeconomic status", "definition": "Hollingshead-style index; 1=highest, 5=lowest.", "range": "1-5"},
    "MMSE": {"label": "MMSE score", "definition": "Mini-Mental State Exam; cognitive screening score, lower means more impairment.", "range": "0-30 (normal >= 27)"},
    "nWBV": {"label": "Whole-brain volume", "definition": "Normalized whole-brain volume; shrinks with atrophy.", "range": "0.5-0.95"},
    "ASF": {"label": "Atlas scaling factor", "definition": "Scaling factor to align brain volume to a standard atlas.", "range": "0.7-1.7"},
    "Sex_M": {"label": "Sex (male)", "definition": "1 if patient is male, 0 if female.", "range": "0 or 1"},
}

_DEFAULT_POPULATION_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "oasis_longitudinal.csv")


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected OASIS columns: {sorted(missing)}")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (X, y) ready for model.py, following the vendor notebook's preprocessing."""
    df = df.copy()
    df["SES"] = df["SES"].fillna(df["SES"].mean())
    df["MMSE"] = df["MMSE"].fillna(df["MMSE"].mean())
    df = pd.get_dummies(df, columns=["M/F"], prefix="Sex", drop_first=True)
    y = df["Group"].apply(lambda g: 0 if g == "Nondemented" else 1)
    X = df[FEATURE_COLUMNS]
    return X, y


@lru_cache(maxsize=None)  # ponytail: caches per path; drop if paths change at runtime
def load_population(path: str = _DEFAULT_POPULATION_PATH) -> pd.DataFrame:
    """Cleaned training population (feature columns only), for population comparisons in the app."""
    X, _ = clean(load_raw(path))
    return X


def record_to_frame(record: dict) -> pd.DataFrame:
    """Turns a single patient dict (raw-style keys, e.g. 'M/F': 'F') into a feature frame."""
    row = {col: record.get(col) for col in FEATURE_COLUMNS if col != "Sex_M"}
    row["Sex_M"] = 1 if record.get("M/F") == "M" else 0
    return pd.DataFrame([row])[FEATURE_COLUMNS]
