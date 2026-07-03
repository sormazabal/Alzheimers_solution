# Demo Walkthrough: Clinical Triage Workflow

## Use Case & End User

This application is designed for **primary care physicians and neurologists** (the end users) operating in a clinical triage setting. The core use case is assessing patients presenting with early signs of cognitive decline or memory loss. Rather than immediately ordering expensive and time-consuming imaging, the clinician can use this tool to synthesize routine clinical data (demographics, MMSE) alongside optional multimodal data (MRI, EEG) to make a fast, evidence-backed decision on whether a specialist referral or further imaging is warranted.

## Walkthrough

This walks through one patient's visit in the Streamlit app
([app/streamlit_app.py](app/streamlit_app.py)) end to end, then explains why this triage
pattern is valuable for a Siemens Healthineers integration.

Run it locally:
```
uv run streamlit run app/streamlit_app.py
```

## Scenario

**Patient:** Jane Doe, ID `OASIS-10234`, F, DOB 1950-03-14. History: hypertension, type 2
diabetes, former smoker. (This is the app's seeded demo patient — sidebar fields are
editable for a real intake.)

## Step 1 — Clinical risk (Clinical risk tab)

The clinician enters what's already collected at a routine visit: age, education, SES,
MMSE score, and brain-volumetric fields if available (nWBV, ASF, MRI delay). No imaging
has been ordered yet.

On submit, `alz.predict()` returns a risk score and label instantly:
- **Risk score** — e.g. 62% ("At-risk"), from the logistic regression's probability.
- **Top contributing factors** — e.g. "MMSE lowers risk (27, 68th percentile)", "Age
  raises risk (75, 82nd percentile)" — each with a population-comparison chart.
- **MMSE trend** — a sparkline against the patient's prior-visit scores, if this isn't
  their first visit.
- **Recommended next steps** — named per pending assessment, e.g. "Recommend completing
  the remaining assessment(s) (MRI/EEG) and a follow-up visit in 6 months."

This step alone is often enough to decide whether imaging is warranted at all.

## Step 2 — MRI confirmation (MRI records tab)

If the clinical score flags concern, the clinician uploads a brain MRI slice. The
ResNet18 severity classifier returns:
- **Predicted severity** — Non Demented / Mild / Moderate Dementia, with confidence.
- **Probability bar chart** across all three classes.
- **Grad-CAM overlay** — a heatmap on the scan itself showing which regions drove the
  prediction, so a radiologist can sanity-check the model against actual anatomy rather
  than trust a bare label.
- **AI-generated radiology-style FINDINGS** — a plain-language paragraph (from a
  vision-LLM read of the scan, or a text fallback from the Grad-CAM attention pattern)
  describing hippocampal volumes and lateral ventricle sizes, explicitly framed as a screening aid.

## Step 3 — EEG confirmation (EEG records tab)

The clinician selects or uploads an EEG recording. The app extracts band-power features
(the classic AD-associated "slowing" — more delta/theta, less alpha/beta) and shows:
- A **signal viewer** of the raw channels.
- A **gauge** for P(Alzheimer's pattern) with a Healthy/AD-pattern label. This gauge visualizes the model's confidence (from 0% to 100%) that the patient's EEG band-power features align with patterns typically associated with Alzheimer's Disease. A high percentage triggers the "Alzheimer's pattern" evaluation result, indicating significant slowing, while a lower score indicates a "Healthy pattern".

## Step 4 — Overview: the integrated picture

The Overview tab pulls every completed assessment together:
- **Integrated prognosis** — one fused probability across all modalities present,
  computed by a logarithmic opinion pool (see [docs/fusion-methodology.md](docs/fusion-methodology.md)).
  A confident modality naturally dominates an uncertain one; with only one modality run,
  the fused score equals that modality's own score.
- **Generate AI summary** — drafts a 3–5 sentence clinical note synthesizing all
  available results, for the clinician to review and edit — not a final report.
- **Find evidence & trials** — live PubMed and ClinicalTrials.gov lookups, returning
  recent guideline articles and recruiting trials with clickable PMID/NCT links, plus a
  short LLM recommendation that cites only those retrieved sources (no invented citations).
- **Ask about this case** — a chat grounded in this patient's actual results, for
  follow-up questions during the visit.

## Why this matters for Siemens

- **Triage before imaging.** The clinical-risk step runs on data already collected —
  before any MRI is ordered — so scanner time and imaging budget go to patients who show
  signal, not everyone screened.
- **One integration seam.** Every caller (this Streamlit demo, the FastAPI service, or a
  future SHS product UI) hits the same `alz.predict()` function and JSON shape — no
  bespoke glue per surface.
- **Multi-modal fusion, delivered.** Tabular + MRI + EEG fused into one prognosis is the
  differentiator both `PRESENTATION.md` and `README.md` flagged as future work — it's
  now implemented, not aspirational.
- **Explainable by construction.** Coefficient-based drivers, Grad-CAM overlays, and
  citation-grounded recommendations mean every AI output traces back to something a
  clinician can inspect — no opaque black-box score.
- **Runs on commodity infrastructure.** No paid LLM/cloud APIs required for the core
  scoring; models run locally on CPU (GPU used automatically if present).
