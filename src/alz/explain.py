"""Top contributing factors from the logistic regression's own coefficients — no SHAP needed."""
import logging

import pandas as pd
from sklearn.pipeline import Pipeline

from alz.data import FEATURE_COLUMNS, load_population
from alz.fusion import integrated_score

_log = logging.getLogger(__name__)


def _strip_code_fence(text: str) -> str:
    """Strip a markdown ```json ... ``` fence some models wrap JSON responses in."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


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


def _attention_summary(cam) -> str:
    """Turn a 224x224 Grad-CAM array into a short phrase a clinician can read.

    Derives coarse location (center/periventricular vs peripheral/cortical, top/mid/bottom),
    left-right symmetry, and focal-vs-diffuse spread. No anatomical atlas is involved -- this
    describes where the *model* attended, not confirmed anatomy.
    """
    h, w = cam.shape
    ys, xs = (cam > cam.mean() + cam.std()).nonzero() if (cam > cam.mean() + cam.std()).any() else cam.nonzero()
    cy, cx = (ys.mean() if len(ys) else h / 2), (xs.mean() if len(xs) else w / 2)

    frac_h = cy / h
    vertical = "superior" if frac_h < 0.4 else "inferior" if frac_h > 0.6 else "mid"
    frac_w = cx / w
    horiz_dist = abs(frac_w - 0.5)
    location = "central/periventricular" if horiz_dist < 0.2 else "peripheral/cortical"

    left_mean = cam[:, : w // 2].mean()
    right_mean = cam[:, w // 2 :].mean()
    diff = abs(left_mean - right_mean) / (left_mean + right_mean + 1e-8)
    symmetry = "roughly symmetric" if diff < 0.15 else ("left-dominant" if left_mean > right_mean else "right-dominant")

    focal = "focal" if (cam > 0.5).mean() < 0.25 else "diffuse"

    return f"strongest in the {vertical}, {location} region, {symmetry}, and {focal} in spread"


def _vision_findings(image_bytes: bytes, prompt: str) -> str:
    """Neuro-relevant findings from a vision-capable LLM looking at the actual scan.

    Uses HuggingFace's gemma-3-27b-it (Llama-3.2-11B-Vision-Instruct was dropped from
    HF's Inference Providers catalog -- inferenceProviderMapping is now empty for it).
    Requires HF_TOKEN in the environment.
    """
    import base64
    import io
    import json
    import os

    from huggingface_hub import InferenceClient
    from PIL import Image

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    png_buf = io.BytesIO()
    Image.open(io.BytesIO(image_bytes)).convert("RGB").save(png_buf, format="PNG")
    b64 = base64.b64encode(png_buf.getvalue()).decode()

    client = InferenceClient(token=os.getenv("HF_TOKEN"))
    response = client.chat_completion(
        model="google/gemma-3-27b-it",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=512,
    )
    return _coerce_findings_text(json.loads(_strip_code_fence(response.choices[0].message.content), strict=False)["findings"])


def _findings_prompt(result: dict, attention: str | None) -> str:
    probs_str = ", ".join(f"{c}: {p:.1%}" for c, p in result["probs"].items())
    attention_line = (
        f"The model's Grad-CAM attention was {attention}. " if attention else ""
    )
    return (
        f"An MRI-based dementia severity classifier predicted '{result['label']}' "
        f"with {result['score']:.1%} confidence. Full class probabilities: {probs_str}. "
        f"Class meanings: {_MRI_CLASS_MEANINGS} {attention_line}"
        "In 2-3 sentences, comment on neurologically relevant structural features visible "
        "in the scan -- lateral ventricle size, medial-temporal/hippocampal atrophy, cortical "
        "sulcal widening or global atrophy, and hemispheric symmetry -- and connect what you "
        "observe to the predicted severity level. Note the model's uncertainty where relevant, "
        "and make clear this is a screening aid, not a diagnosis. You are a clinical decision-support "
        "tool for a licensed clinician, not a chatbot talking to a patient -- always give your best "
        "concrete, specific findings; never refuse or reply with a generic disclaimer like "
        "\"I can't provide specific medical information\". "
        'Respond as JSON: {"findings": "..."}'
    )


def explain_mri(result: dict, image_bytes: bytes | None = None, cam=None) -> str | None:
    """Plain-language radiology-style findings for an MRI severity prediction.

    'result' is predict_mri_probs()'s output: {'probs', 'label', 'score'}. When 'image_bytes'
    is given, a vision LLM looks at the actual scan and comments on structural features
    (ventricles, atrophy, symmetry). Otherwise (or if the vision call fails), falls back to a
    text LLM describing where the Grad-CAM 'cam' array shows the model's attention. Returns
    None if everything fails (missing key, package not installed, network) so the UI can
    degrade gracefully.

    ponytail: one hardcoded prompt, no caching/retries -- add if this becomes a hot path.
    """
    import json

    attention = _attention_summary(cam) if cam is not None else None
    prompt = _findings_prompt(result, attention)

    if image_bytes is not None:
        try:
            return _vision_findings(image_bytes, prompt)
        except Exception:
            _log.exception("Vision LLM findings failed, falling back to text LLM")

    try:
        response = _default_client().complete([{"role": "user", "content": prompt}])
        return _coerce_findings_text(json.loads(_strip_code_fence(response), strict=False)["findings"])
    except Exception:
        _log.exception("explain_mri LLM call failed")
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

    fused = integrated_score(clinical=clinical_result, mri=mri_result, eeg=eeg_result)
    if fused:
        lines.append(f"Integrated prognosis ({fused['note']}): {fused['score']:.0%} posterior probability of dementia.")

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
            "State explicitly that these are screening aids, not diagnoses. You are a clinical "
            "decision-support tool for a licensed clinician -- always give your best concrete "
            "impression; never refuse or reply with a generic disclaimer like \"I can't provide "
            "specific medical information\". "
            'Respond as JSON: {"summary": "..."}'
        )
        response = _default_client().complete([{"role": "user", "content": prompt}])
        return json.loads(_strip_code_fence(response), strict=False)["summary"]
    except Exception:
        _log.exception("synthesize_summary LLM call failed")
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
            "screening aids, not diagnoses -- but still give your best concrete, specific answer "
            "using the case context; never refuse or reply with a generic disclaimer like "
            "\"I can't provide specific medical information\", since the user is a licensed "
            "clinician relying on this tool for decision support, not a patient.\n\n"
            + _case_context(patient, clinical_result, mri_result, eeg_result)
        )
        return _default_client().complete(messages, system=system)
    except Exception:
        _log.exception("chat_about_case LLM call failed")
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


def _coerce_findings_text(value) -> str:
    """Coerce the LLM's 'findings' field to text, however it's shaped -- recursively.

    Some providers (Groq's JSON mode in particular) nest structured fields inside
    'findings', sometimes several levels deep, instead of writing a plain sentence.
    Flatten to readable text so the UI never renders a raw Python dict/list repr.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n\n".join(f"{k.replace('_', ' ').capitalize()}: {_coerce_findings_text(v)}" for k, v in value.items())
    if isinstance(value, list):
        return "\n\n".join(_coerce_findings_text(item) for item in value)
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
                "anything not listed. State these are screening aids, not diagnoses. You are a "
                "clinical decision-support tool for a licensed clinician -- always give concrete, "
                "specific recommendations; never refuse or reply with a generic disclaimer like "
                "\"I can't provide specific medical recommendations\". "
                'Respond as JSON: {"recommendation": "..."}'
            )
            response = _default_client().complete([{"role": "user", "content": prompt}])
            recommendation = _format_recommendation(json.loads(_strip_code_fence(response), strict=False)["recommendation"])
        except Exception:
            _log.exception("evidence_for_case recommendation LLM call failed")
            recommendation = None

    return {"recommendation": recommendation, "pubmed": pubmed, "trials": trials}
