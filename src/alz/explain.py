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


def _default_client():
    """llm.py lives at the repo root, outside src/ -- put it on sys.path and grab the client."""
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from llm import default_client

    return default_client()


def explain_mri(result: dict) -> str | None:
    """Plain-language narrative for an MRI severity prediction, via the configured LLM.

    'result' is predict_mri_probs()'s output: {'probs', 'label', 'score'}. Returns None if
    the LLM call fails for any reason (missing key, package not installed, network) so the
    UI can degrade gracefully.

    ponytail: one hardcoded prompt, no caching/retries -- add if this becomes a hot path.
    """
    import json

    try:
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
        response = _default_client().complete([{"role": "user", "content": prompt}])
        return json.loads(response)["explanation"]
    except Exception:
        return None


def _case_context(patient: dict, clinical_result: dict | None, mri_result: dict | None, eeg_result: dict | None) -> str:
    """Plain-text digest of whichever assessments are available, for prompting the LLM."""
    lines = [f"Patient: {patient['sex']}, DOB {patient['dob']}. History: {patient['history']}"]

    if clinical_result:
        drivers = ", ".join(f"{d['feature']} ({d['direction']})" for d in clinical_result.get("drivers", []))
        lines.append(
            f"Clinical risk model: {clinical_result['label']} ({clinical_result['score']:.0%}). "
            f"Top drivers: {drivers or 'none'}."
        )
    else:
        lines.append("Clinical risk model: not yet assessed.")

    if mri_result:
        lines.append(f"MRI severity classifier: {mri_result['label']} ({mri_result['score']:.0%} confidence).")
    else:
        lines.append("MRI severity classifier: not yet assessed.")

    if eeg_result:
        lines.append(f"EEG classification: {eeg_result['label']} ({eeg_result['score']:.0%} confidence).")
    else:
        lines.append("EEG classification: not yet assessed.")

    return "\n".join(lines)


def synthesize_summary(patient: dict, clinical_result: dict | None, mri_result: dict | None, eeg_result: dict | None) -> str | None:
    """Draft clinical-conclusions note synthesizing whichever assessments are available.

    Returns None on any LLM failure so the UI can degrade gracefully, same as explain_mri.
    """
    import json

    try:
        prompt = (
            f"{_case_context(patient, clinical_result, mri_result, eeg_result)}\n\n"
            "Write a concise clinical note (3-5 sentences) synthesizing these AI screening "
            "results into an overall impression, for a clinician to review and edit. "
            "State explicitly that these are screening aids, not diagnoses. "
            'Respond as JSON: {"summary": "..."}'
        )
        response = _default_client().complete([{"role": "user", "content": prompt}])
        return json.loads(response)["summary"]
    except Exception:
        return None


def chat_about_case(
    messages: list[dict], patient: dict, clinical_result: dict | None, mri_result: dict | None, eeg_result: dict | None
) -> str | None:
    """Answer a clinician's free-form question about the case using the assembled context.

    'messages' is the [{"role": "user"|"assistant", "content": str}, ...] history llm.py
    expects. Returns None on any LLM failure so the UI can degrade gracefully.
    """
    try:
        system = (
            "You are assisting a clinician reviewing a dementia-risk screening case. Answer "
            "using only the context below. If asked to diagnose, remind the user these are "
            "screening aids, not diagnoses.\n\n"
            + _case_context(patient, clinical_result, mri_result, eeg_result)
        )
        return _default_client().complete(messages, system=system)
    except Exception:
        return None
