"""OASIS-2 loading + cleaning, adapted from vendor/alzheimers-disease-prediction/dimentia_pred.ipynb."""
import pandas as pd

REQUIRED_COLUMNS = [
    "Subject ID", "MRI ID", "Group", "Visit", "MR Delay", "M/F", "Hand",
    "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF",
]

# CDR (Clinical Dementia Rating) is dropped as a predictor even though the vendor notebook
# keeps it: CDR is itself the clinical diagnosis, so using it to predict Group would leak
# the answer into a tool meant to screen patients *before* that diagnosis exists.
FEATURE_COLUMNS = ["Visit", "MR Delay", "Age", "EDUC", "SES", "MMSE", "nWBV", "ASF", "Sex_M"]


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


def record_to_frame(record: dict) -> pd.DataFrame:
    """Turns a single patient dict (raw-style keys, e.g. 'M/F': 'F') into a feature frame."""
    row = {col: record.get(col) for col in FEATURE_COLUMNS if col != "Sex_M"}
    row["Sex_M"] = 1 if record.get("M/F") == "M" else 0
    return pd.DataFrame([row])[FEATURE_COLUMNS]
