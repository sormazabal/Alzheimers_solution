"""Top contributing factors from the logistic regression's own coefficients — no SHAP needed."""
import pandas as pd
from sklearn.pipeline import Pipeline

from alz.data import FEATURE_COLUMNS, load_population

_MRI_CLASS_MEANINGS = (
    "Non Demented: no clinical signs of cognitive decline. "
    "Mild Dementia: early, noticeable memory/cognitive changes that still allow mostly "
    "independent daily living. "
    "Moderate Dementia: more pronounced cognitive and functional decline, typically "
    "requiring regular assistance with daily activities."
)


def top_drivers(pipeline: Pipeline, record_df: pd.DataFrame, n: int = 3) -> list[dict]:
    scaler = pipeline.named_steps["scale"]
    clf = pipeline.named_steps["clf"]
    imputed = pipeline.named_steps["impute"].transform(record_df[FEATURE_COLUMNS])
    scaled = scaler.transform(imputed)[0]
    contributions = scaled * clf.coef_[0]

    ranked = sorted(zip(FEATURE_COLUMNS, contributions), key=lambda kv: abs(kv[1]), reverse=True)
    population = load_population()
    return [
        {
            "feature": feature,
            "direction": "raises risk" if value > 0 else "lowers risk",
            "value": float(record_df[feature].iloc[0]),
            "percentile": float((population[feature] < record_df[feature].iloc[0]).mean() * 100),
        }
        for feature, value in ranked[:n]
    ]


def explain_mri(result: dict) -> str | None:
    """Plain-language narrative for an MRI severity prediction, via the configured LLM.

    'result' is predict_mri_probs()'s output: {'probs', 'label', 'score'}. Returns None if
    the LLM call fails for any reason (missing key, package not installed, network) so the
    UI can degrade gracefully.

    ponytail: one hardcoded prompt, no caching/retries -- add if this becomes a hot path.
    """
    import json
    import os
    import sys

    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from llm import default_client

        probs_str = ", ".join(f"{c}: {p:.1%}" for c, p in result["probs"].items())
        prompt = (
            f"An MRI-based dementia severity classifier predicted '{result['label']}' "
            f"with {result['score']:.1%} confidence. Full class probabilities: {probs_str}. "
            f"Class meanings: {_MRI_CLASS_MEANINGS} "
            "In 2-3 sentences, explain to a clinician in plain language why this severity "
            "level was likely assigned, note the model's uncertainty where relevant, and "
            "make clear this is a screening aid, not a diagnosis. "
            'Respond as JSON: {"explanation": "..."}'
        )
        response = default_client().complete([{"role": "user", "content": prompt}])
        return json.loads(response)["explanation"]
    except Exception:
        return None
