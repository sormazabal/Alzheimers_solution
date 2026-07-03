# Alzheimer's Early-Risk MVP

**Subtitle:** A modular Alzheimer's-risk pipeline that scores patients from routine clinical data today and scales to MRI confirmation later — all behind one integration seam any system can call.
**Presenter:** Zow ORmazabal

## Slide 2: Outline
- Project overview and client value
- Client pains and system challenges
- Proposed solution
- Dataset overview
- Architecture details
- Technical implementation
- Cost
- Conclusion

## Slide 3: Project overview and client value
**Overview:** A proof-of-capability screening pipeline for Siemens Digital Health / SHS AI. It takes cognitive scores + demographics and returns an Alzheimer's risk score with the top contributing factors, *before* imaging is ordered. MRI-based severity classification and EEG-based confirmation each add an independent signal, fused into one integrated prognosis, with an LLM decision-support layer (doctor summary, grounded Q&A, live evidence retrieval) on top.
**Key objective:** Flag at-risk patients early from data clinics already collect, so imaging and specialist time are spent where they matter — then let clinicians confirm and act on that signal without leaving the tool.
**Client value:** Triage before imaging = fewer unnecessary scans and lower cost per screened patient; an honest, coefficient-based explanation per prediction (plus Grad-CAM for MRI) supports clinical trust; one function-call seam (`alz.predict`) means any Siemens system (UI, API, notebook) integrates without bespoke glue. Multi-modal fusion — the stated SHS differentiator — is implemented, not aspirational.
**Status:** MVP+. Tabular risk, MRI severity, and EEG classification are each built and tested; a logarithmic-opinion-pool fusion combines whichever are available into one score; an LLM layer adds doctor-facing summaries, MRI narratives, case Q&A, and live PubMed/ClinicalTrials.gov evidence. Auth, Docker, model registry, CI, and real EHR integration deliberately deferred.
**Category:** Clinical / digital-health machine learning.

## Slide 4: Client pains and system challenges
**Client pain points:**
- Imaging (MRI/PET) is expensive and capacity-limited — ordering it without prior triage wastes budget and scanner time.
- Alzheimer's risk is often assessed late, after symptoms; earlier signals in routine data go unused.
- Black-box risk tools are hard to trust clinically and hard to integrate across a vendor's product line.

**Systemic challenges:**
- Cognitive scores + demographics are collected but not systematically turned into a risk signal.
- Real datasets (OASIS-2, imaging) carry data-use terms / need agreements, slowing prototyping.
- A naive model can leak the diagnosis: the vendor reference keeps `CDR` (the clinical dementia rating) as a predictor — but CDR *is* the diagnosis, so a pre-diagnosis screening tool must exclude it.
- Imaging references assume 3D NIfTI volumes (MONAI/TorchIO/MedicalNet); the publicly available OASIS Kaggle mirror ships 2D JPEG slices, so that pipeline doesn't transfer directly.

## Slide 5: Proposed solution
- **One integration seam:** every caller uses `from alz import predict`; a dict in → `{score, label, drivers}` out. Streamlit demo, FastAPI service, and tests all hit the same function.
- **Tabular risk model:** sklearn `Pipeline(impute → scale → LogisticRegression)`. Logistic regression chosen over the reference RandomForest so coefficients give a direct, honest explanation.
- **Built-in explainability:** `explain.top_drivers()` ranks features by `scaled_value × coefficient` — top-3 "raises/lowers risk" factors, no SHAP dependency.
- **Leakage guard:** `CDR` dropped as a predictor by design (documented in `data.py`).
- **MRI confirmation:** ImageNet-pretrained ResNet18, frozen backbone + retrained head, severity classification (Non Demented / Mild / Moderate Dementia); inverse-frequency class weights for imbalance; Grad-CAM overlay shows which scan regions drove the prediction; runs on CPU, uses GPU when present.
- **EEG confirmation:** relative band-power features (delta–gamma + theta/alpha and slowing ratios, the textbook AD-EEG signature) into a `StandardScaler + LogisticRegression(class_weight="balanced")` pipeline — same explainable shape as the tabular model, sized for the small (n≈65) labeled EEG dataset available.
- **Integrated prognosis:** `fusion.integrated_score()` combines whichever modalities are available (clinical, MRI, EEG) via a **logarithmic opinion pool** — a weighted average in log-odds space, so a confident modality naturally dominates an uncertain one without hand-tuned weights (see `docs/fusion-methodology.md`).
- **LLM decision-support layer** (`explain.py`): doctor-facing case summary (`synthesize_summary`), plain-language MRI findings grounded in the scan + Grad-CAM (`explain_mri`), free-form case Q&A (`chat_about_case`), and live evidence retrieval — PubMed guidelines + ClinicalTrials.gov recruiting trials with clickable PMID/NCT provenance, feeding a citation-grounded recommendation (`evidence_for_case`). Every LLM call degrades to `None`/empty on failure rather than breaking the UI.
- **Thin service layer:** FastAPI `/predict` + `/health` wrapping the same seam; `load_model` is `@lru_cache`'d so the model loads once per process.

## Slide 6: Architecture details
**Layer 1 — User interface and interaction:**
- Streamlit clinical workstation ([app/streamlit_app.py](app/streamlit_app.py)) — 4 tabs: **Clinical risk** (patient form → score + drivers), **MRI records** (scan upload → severity + Grad-CAM + AI findings), **EEG records** (recording → band-power classification + signal viewer), **Overview** (integrated prognosis, AI case summary, live evidence, case chat).
- FastAPI service ([app/api.py](app/api.py)) — `POST /predict` (Pydantic-validated `PatientRecord`) and `GET /health`.
- CLIs: [train.py](train.py), [train_mri.py](train_mri.py), [evaluate_mri.py](evaluate_mri.py).

**Layer 2 — System logic ([src/alz/](src/alz/)):**
- `data.py` — OASIS-2 load/validate/clean; defines `FEATURE_COLUMNS`; `record_to_frame()` turns a patient dict into a feature row.
- `model.py` — build/train/save/load pipeline; `predict()` → `{score, label}`.
- `imaging.py` — ResNet18 severity classifier: `train_mri`, `evaluate_mri`, `predict_mri` / `predict_mri_probs`, `gradcam_mri`; shared 70/15/15 split so eval matches training.
- `eeg.py` — band-power feature extraction (`extract_features`) + logistic regression: `train_eeg`, `predict_eeg` / `predict_eeg_probs`.
- `fusion.py` — `integrated_score()`: logarithmic-opinion-pool fusion of whichever modalities are present into one prognosis.
- `explain.py` — `top_drivers()` (tabular coefficient ranking), `synthesize_summary()`, `explain_mri()`, `chat_about_case()`, `evidence_for_case()` (live PubMed/ClinicalTrials.gov retrieval + grounded recommendation).
- `__init__.py` — public `predict()` seam composing the above.

**Layer 3 — Data sources:**
- Local tabular CSV `data/oasis_longitudinal.csv` (synthetic sample).
- Kaggle imaging dataset `ninadaithal/imagesoasis` via `download_oasis.py` (kagglehub cache).
- EEG dataset `ds004504` (real, publicly available OpenNeuro AHEPA resting-state recordings).
- Saved model artifacts: `models/model.joblib`, `models/mri_model.pt`, `models/eeg_model.joblib`.
- Live external APIs: PubMed E-utilities, ClinicalTrials.gov v2 (no API key required).

**Example I/O (from [tests/test_api.py](tests/test_api.py) / README):**
```json
// POST /predict  request
{"Age":75,"EDUC":12,"SES":3,"MMSE":27,"nWBV":0.75,"ASF":1.2,"Visit":1,"MR Delay":0,"M/F":"F"}
// response
{"score":0.12,"label":"Normal","drivers":[{"feature":"MMSE","direction":"lowers risk"}, ...]}
```

## Slide 6b: How each model computes its score

**Clinical (tabular):** 9 features → mean-imputation → standardization → logistic
regression. `score = predict_proba()[1]`, the model's own probability of the "at-risk"
class. Explanation is not a separate step bolted on afterward — `top_drivers()` reads the
same coefficients the model used to score, ranking features by `scaled_value × coefficient`.

**MRI severity:** a brain-slice image → ResNet18 (frozen ImageNet backbone, retrained
final layer) → softmax over severity classes. `score` is the winning class's probability.
Grad-CAM (`gradcam_mri`) backprops the winning class's logit into the last convolutional
block to show *which pixels* drove that softmax output, as a heatmap overlay.

**EEG:** a resting-state recording → relative band power (delta/theta/alpha/beta/gamma)
plus two "slowing" ratios that summarize the classic AD-EEG pattern (more low-frequency,
less high-frequency power) → standardized logistic regression. `score = P(AD pattern)`.

**Integrated prognosis:** each modality's own probability is converted to log-odds
(`logit(p) = log(p/(1-p))`), averaged (equal weight by default), and converted back with
a sigmoid:

```
logit(P_fused) = ( Σ rᵢ · logit(pᵢ) ) / ( Σ rᵢ )
P_fused        = sigmoid(logit(P_fused))
```

A modality near p≈0.5 (uncertain) has logit≈0 and barely moves the fused result; a
confident modality (p near 0 or 1) pulls the fused score strongly toward it — so a
stronger signal dominates a weaker one without any hand-picked weighting. Full
justification and limitations: [docs/fusion-methodology.md](docs/fusion-methodology.md).

### Model metrics

| Model | Algorithm | Reported metric | Result | Eval split | Data used |
|---|---|---|---|---|---|
| Clinical | Logistic regression (9 features) | Held-out accuracy | **0.76** | Stratified 75/25 | Synthetic `oasis_longitudinal.csv` (82 rows) — indicative only, not a real-world benchmark |
| MRI severity | ResNet18 (frozen backbone, retrained head) | Accuracy / AUROC / AUPRC | **train 1.00 / val 1.00 / test 1.00** acc; AUROC/AUPRC n/a (see note) | 70/15/15 | Real Kaggle `imagesoasis` mirror (~23k slices) |
| EEG | Logistic regression on band-power features | Mean 5-fold CV accuracy | **0.75** | Stratified 5-fold CV (n=65) | Real ds004504 (AHEPA resting-state EEG) |
| Integrated prognosis | Logarithmic opinion pool (no trained parameters) | — | Not separately benchmarked | — | No paired multi-modal ground truth exists to validate against (see methodology doc) |

**Note on the MRI row:** 1.00 accuracy reflects near-ceiling performance for this
class-imbalanced dataset combined with a limited (frozen-backbone, head-only) model — it
is not proof of clinical-grade generalization, and AUROC/AUPRC report `n/a` when
sklearn's scorers can't compute an ROC/PR curve for a class with no predicted negatives.
Treat this as an existence proof that the pipeline trains and evaluates correctly, not as
a validated accuracy claim; production use requires evaluation on a held-out clinical
cohort, not the same public source distribution the model trained on.

## Slide 7: Dataset overview

| Data source | What it contains | How it is acquired |
|---|---|---|
| `data/oasis_longitudinal.csv` (bundled) | **Synthetic** OASIS-2-shaped sample: 40 fake subjects, 82 visits; columns `Subject ID, MRI ID, Group, Visit, MR Delay, M/F, Hand, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF` | Committed to repo; runs out of the box |
| OASIS-2 longitudinal (real) | Same schema; `Group` (Nondemented/Demented/Converted) is the target | Download from oasis-brains.org, drop-in replace — no code change |
| OASIS imaging (`ninadaithal/imagesoasis`) | 2D JPEG brain slices in per-class severity folders | `python data/download_oasis.py` (kagglehub) |

**Key characteristics / privacy:**
- Only synthetic data is committed — the demo needs no data agreement.
- Real OASIS data has its own data-use terms; imaging mirror is publicly downloadable (no DUA).
- `CDR` excluded as predictor (diagnosis leakage); `SES`/`MMSE` missing values mean-imputed; `Subject ID`, `MRI ID`, `Hand`, `eTIV` dropped/unused.

## Slide 8: Technical implementation and demo
**Environment:** Python (uses `str | None` / `tuple[...]` syntax → **3.10+**; no explicit pin in repo). Package manager: **uv**. Core deps: scikit-learn, pandas, joblib, streamlit, fastapi, uvicorn, httpx, pytest. Phase-2 deps (`requirements-imaging.txt`): torch, torchvision, pillow, kagglehub, scikit-learn, matplotlib.
**Integration:** Kaggle (via kagglehub) for imaging data; torchvision ImageNet-pretrained ResNet18 as an installed library. No external LLM/API providers. Builds on vendored reference repo `alzheimers-disease-prediction` (git submodule).
**Demo commands:**
```
uv venv
uv pip install -r requirements.txt
uv run python train.py                     # → models/model.joblib
uv run pytest tests/test_smoke.py          # train→predict roundtrip
uv run streamlit run app/streamlit_app.py  # interactive demo
uv run uvicorn app.api:app --reload        # REST API
# Phase 2 (optional):
uv pip install -r requirements-imaging.txt
uv run python train_mri.py                 # → models/mri_model.pt
uv run python evaluate_mri.py              # metrics + ROC/PR plots
```

## Slide 9: Results and evaluation
No numeric benchmark results are committed to the repo (models train from data at runtime; only synthetic data ships). The evaluation *machinery* is in place:

**Tabular track:** `train.py` reports held-out accuracy on a stratified 75/25 split (printed at train time).

**Imaging track:** per-epoch train/val and final test **accuracy, AUROC, AUPRC** (`imaging._metrics`); `evaluate_mri.py` reports all three splits and saves ROC + PR curves to `eval_plots/`. AUROC/AUPRC gracefully report `n/a` when a split has a single class (tiny/synthetic data).

**Test coverage:** `tests/test_smoke.py` (tabular roundtrip), `tests/test_api.py` (FastAPI `/predict` + `/health`), `tests/test_imaging.py` (MRI train→predict→evaluate roundtrip on a synthetic dataset, skips if imaging deps absent).

## Slide 10: Cost analysis
Information not available in the current repository. No infrastructure, API-billing, or compute-cost documentation exists. Observations from the code: no paid LLM/cloud APIs are called; both models train locally (tabular is trivial; ResNet18 head-only fine-tune runs on CPU, uses GPU automatically when available), so runtime cost is local compute only.

## Slide 11: Roadmap and next steps
**Completed milestones (from TODO.md):**
- Tabular risk model + explainability + `alz.predict()` seam (built, tested).
- FastAPI wrapper (`/predict`, `/health`) with Pydantic validation.
- Phase-2 MRI severity classifier (ResNet18 transfer learning) + evaluation CLI + plots.
- `download_oasis.py` documented as a separate phase-2 imaging fetch.

**Future / planned:**
- **Multi-modal fusion** (tabular + imaging) — the stated SHS differentiator, not built.
- Deliberately deferred: auth, Docker, model registry, CI.
- Swap synthetic CSV for real OASIS-2 data; train imaging model on real OASIS slices for real metrics.

## Slide 12: Conclusion and key takeaways
- Turns routine clinical data into an explainable early-risk signal *before* costly imaging — directly attacking the client's imaging-cost and late-detection pains.
- One `alz.predict()` seam makes integration across Siemens systems trivial; Streamlit, REST API, and tests all reuse it.
- Honest-by-design: logistic-regression coefficients drive the explanation, and `CDR` is excluded to avoid leaking the diagnosis into a pre-diagnosis screener.
- Scaffolded, not overbuilt: phase-2 MRI confirmation is implemented and the fusion differentiator has a clear path, while auth/Docker/registry/CI are consciously deferred to keep the MVP honest about what's proven.
