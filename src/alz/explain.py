"""Top contributing factors from the logistic regression's own coefficients — no SHAP needed."""
import pandas as pd
from sklearn.pipeline import Pipeline

from alz.data import FEATURE_COLUMNS


def top_drivers(pipeline: Pipeline, record_df: pd.DataFrame, n: int = 3) -> list[dict]:
    scaler = pipeline.named_steps["scale"]
    clf = pipeline.named_steps["clf"]
    imputed = pipeline.named_steps["impute"].transform(record_df[FEATURE_COLUMNS])
    scaled = scaler.transform(imputed)[0]
    contributions = scaled * clf.coef_[0]

    ranked = sorted(zip(FEATURE_COLUMNS, contributions), key=lambda kv: abs(kv[1]), reverse=True)
    return [
        {"feature": feature, "direction": "raises risk" if value > 0 else "lowers risk"}
        for feature, value in ranked[:n]
    ]
