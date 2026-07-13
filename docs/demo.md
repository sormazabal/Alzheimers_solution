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

## Scene 2b — 3D MRI volume view (insert after Step 2, before Step 3)

**Narration:** "The 2D slice is a quick read, but a single slice can miss where atrophy actually
sits. Here's the same scan as a full 3D volume, so we can see the model's attention across
the whole brain, not just one cut."

- Click into the **"3D volume"** sub-tab, next to "2D slice."
- Upload a `.nii` / `.nii.gz` volume (or a matching `.hdr` + `.img` pair), or pick a bundled
  OASIS1 example subject from the dropdown, then click **Run evaluation**.
- A caption flags this as **v1**: the existing 2D model is applied to central axial slices and
  pooled, not a separately validated volumetric classifier — say this out loud on camera so it
  doesn't read as an overclaim.
- Same prediction card and probability bar as the 2D tab, plus the single central axial slice
  with the highest predicted risk shown side-by-side with its Grad-CAM overlay.
- Open the **"3D volume (rotable)"** expander — the headline shot for this scene: an
  interactive, mouse-rotatable render of the whole brain as a translucent gray shell, with a
  hot-colored Grad-CAM activation region glowing inside it. Rotate it on camera for a few
  seconds before moving on.
- Closes with the same radiology-style FINDINGS paragraph as the 2D tab.

## Scene 2c — MRI 2D + 3D combined confirmation (insert immediately after Scene 2b, before Step 3)

**Narration:** "Neither view has to stand alone. Once we've run both the slice and the volume,
the app cross-checks them against each other and gives one combined confirmation."

- Requires both the 2D and 3D analyses to have already been run in this session — the tab
  prompts "Run both the 2D slice and 3D volume analyses first" otherwise, so record Scene 2b
  before this one.
- Click into the third sub-tab, **"2D + 3D combined."**
- Shows a **"Combined dementia confirmation"** metric with confidence, and an explicit note on
  whether the 2D and 3D pathways agree or disagree on the predicted label.
- A 3-bar chart compares P(Demented) for 2D, 3D, and Combined side by side — a clean visual
  for showing the two pathways converging (or flagging a disagreement worth a second look).
- Closes with a combined radiology report summary, same FINDINGS format as the other MRI
  scenes.

## Step 3 — EEG confirmation (EEG records tab)

The clinician selects or uploads an EEG recording. The app extracts band-power features
(the classic AD-associated "slowing" — more delta/theta, less alpha/beta) and shows:
- A **signal viewer** of the raw channels.
- A **gauge** for P(Alzheimer's pattern) with a Healthy/AD-pattern label. This gauge visualizes the model's confidence (from 0% to 100%) that the patient's EEG band-power features align with patterns typically associated with Alzheimer's Disease. A high percentage triggers the "Alzheimer's pattern" evaluation result, indicating significant slowing, while a lower score indicates a "Healthy pattern".

## Scene 3b — EEG "Why this score" explainability panel (insert immediately after Step 3, same tab)

**Narration:** "The gauge gives a number, but a clinician needs to know what's driving it. This
panel breaks the score down by EEG band, compares the patient against a healthy and an AD
cohort, and maps where on the scalp the slowing is strongest."

- Appears automatically under **"Why this score"** as soon as the evaluation finishes — no
  extra click needed, so this scene continues straight out of Step 3's gauge shot.
- **Band contribution chart** — horizontal bars per EEG feature (delta, theta, alpha, beta,
  gamma power, plus the theta/alpha and slowing ratios), colored red where a band raises risk
  and green where it lowers it. Hovering a bar shows a plain-English tooltip and the exact
  contribution value — worth a slow hover-over on camera to show the tooltips.
- **Relative band power vs. cohort chart** next to it — this patient's band power plotted
  against a healthy-cohort mean and an AD-cohort mean, so the deviation is visible at a glance.
- **Scalp topography map** — a heatmap of slow-wave (delta + theta) power across the scalp,
  redder where slowing is strongest, with a colorbar. Another strong visual moment for the
  recording.
- Closes with a short LLM-drafted clinical readout tying the top contributing bands to the
  predicted label in plain language.
- Narration should land on: this turns the gauge number into something a clinician can audit,
  not a black-box score.

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
  now implemented, not aspirational. The MRI pathway itself is now cross-checked internally
  too: a 2D slice read and a full 3D volumetric read are fused into one combined confirmation,
  not just tabular + MRI + EEG.
- **Explainable by construction.** Coefficient-based drivers, Grad-CAM overlays, and
  citation-grounded recommendations mean every AI output traces back to something a
  clinician can inspect — no opaque black-box score. The EEG band-contribution chart and
  scalp topography map extend this to the EEG pathway, turning a single confidence number
  into an auditable, per-band breakdown.
- **Runs on commodity infrastructure.** No paid LLM/cloud APIs required for the core
  scoring; models run locally on CPU (GPU used automatically if present).

## Publishing to the company repo

`docs/` and `.claude/` stay in this personal repo only; they should not reach the company
repo. One-time setup, then a push:

```bash
git remote add company <company-repo-url>     # once
git checkout -b company-publish master
git rm -r --cached docs .claude
git commit -m "chore: strip personal-only dirs for company repo"
git push company company-publish:master --force
git checkout master && git branch -D company-publish
```

`master` (and `origin`) keep `docs/` and `.claude/` untouched; only the throwaway
`company-publish` branch drops them before pushing.
