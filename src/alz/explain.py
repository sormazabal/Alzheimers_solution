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


_PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
_TRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"


def _parse_pubmed(esummary_json: dict) -> list[dict]:
    """Extract [{'pmid', 'title', 'source', 'pubdate'}] from an ESummary JSON payload."""
    result = esummary_json.get("result", {})
    return [
        {
            "pmid": pmid,
            "title": result[pmid].get("title", ""),
            "source": result[pmid].get("source", ""),
            "pubdate": result[pmid].get("pubdate", ""),
        }
        for pmid in result.get("uids", [])
    ]


def _parse_trials(v2_json: dict) -> list[dict]:
    """Extract [{'nct', 'title', 'status'}] from a ClinicalTrials.gov v2 studies payload."""
    trials = []
    for study in v2_json.get("studies", []):
        section = study.get("protocolSection", {})
        ident = section.get("identificationModule", {})
        status = section.get("statusModule", {})
        trials.append({
            "nct": ident.get("nctId", ""),
            "title": ident.get("briefTitle", ""),
            "status": status.get("overallStatus", ""),
        })
    return trials


def _pubmed_guidelines(query: str, retmax: int = 3) -> list[dict]:
    """Recent PubMed articles matching 'query', via ESearch -> ESummary (both JSON, no XML)."""
    import httpx

    search = httpx.get(
        f"{_PUBMED_BASE}esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmode": "json", "sort": "date", "retmax": retmax},
        timeout=10,
    )
    search.raise_for_status()
    pmids = search.json().get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    summary = httpx.get(
        f"{_PUBMED_BASE}esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
        timeout=10,
    )
    summary.raise_for_status()
    return _parse_pubmed(summary.json())


def _recruiting_trials(condition: str, retmax: int = 3) -> list[dict]:
    """Currently recruiting trials matching 'condition', via the ClinicalTrials.gov v2 API."""
    import httpx

    resp = httpx.get(
        _TRIALS_BASE,
        params={"query.cond": condition, "filter.overallStatus": "RECRUITING", "pageSize": retmax},
        timeout=10,
    )
    resp.raise_for_status()
    return _parse_trials(resp.json())


def _format_recommendation(value) -> str:
    """Coerce the LLM's 'recommendation' field to text, however it's shaped.

    The prompt asks for a plain string but LLMs don't always comply -- e.g. they
    sometimes return a list of {"action", "rationale"} objects instead. Passing
    a non-string straight to st.write() would render it as a raw JSON tree, so
    normalize here.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, dict) and "action" in item for item in value):
        return "\n".join(f"- **{item['action']}**: {item.get('rationale', '')}" for item in value)
    return str(value)


def evidence_for_case(
    patient: dict, clinical_result: dict | None, mri_result: dict | None, eeg_result: dict | None
) -> dict:
    """Live-retrieved guidelines + recruiting trials for this case, plus a grounded LLM recommendation.

    Always returns {'recommendation', 'pubmed', 'trials'}; each part degrades independently
    (empty list / None recommendation) if its network or LLM call fails, so the UI can
    render whatever succeeded rather than an all-or-nothing failure.

    ponytail: fixed "Alzheimer Disease" condition/query, no per-case query tuning -- revisit
    if case-specific search terms (e.g. staging, comorbidities) prove valuable.
    """
    import json

    try:
        pubmed = _pubmed_guidelines("Alzheimer Disease AND guideline[Publication Type]")
    except Exception:
        pubmed = []

    try:
        trials = _recruiting_trials("Alzheimer Disease")
    except Exception:
        trials = []

    recommendation = None
    if pubmed or trials:
        try:
            sources = "\n".join(f"[PMID:{a['pmid']}] {a['title']} ({a['source']}, {a['pubdate']})" for a in pubmed)
            sources += "\n" + "\n".join(f"[NCT:{t['nct']}] {t['title']} ({t['status']})" for t in trials)
            prompt = (
                f"{_case_context(patient, clinical_result, mri_result, eeg_result)}\n\n"
                f"Available sources:\n{sources}\n\n"
                "Recommend 2-4 concrete next actions for the clinician, citing ONLY the "
                "sources above as [PMID:x] or [NCT:y] inline. Do not invent sources or cite "
                "anything not listed. State these are screening aids, not diagnoses. "
                'Respond as JSON: {"recommendation": "..."}'
            )
            response = _default_client().complete([{"role": "user", "content": prompt}])
            recommendation = _format_recommendation(json.loads(response)["recommendation"])
        except Exception:
            recommendation = None

    return {"recommendation": recommendation, "pubmed": pubmed, "trials": trials}
