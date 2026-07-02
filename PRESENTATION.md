# Alzheimer's Early-Risk MVP

**Subtitle:** A modular Alzheimer's-risk pipeline that scores patients from routine clinical data today and scales to MRI confirmation later — all behind one integration seam any system can call.
**Presenter:** Sofia Ormazabal Arriagada *(from git author; no maintainer named in repo)*

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
**Overview:** A proof-of-capability screening pipeline for Siemens Digital Health / SHS AI. It takes cognitive scores + demographics and returns an Alzheimer's risk score with the top contributing factors, *before* imaging is ordered. A phase-2 module adds MRI-based 4-class severity classification.
**Key objective:** Flag at-risk patients early from data clinics already collect, so imaging and specialist time are spent where they matter.
**Client value:** Triage before imaging = fewer unnecessary scans and lower cost per screened patient; an honest, coefficient-based explanation per prediction supports clinical trust; one function-call seam (`alz.predict`) means any Siemens system (UI, API, notebook) integrates without bespoke glue. Positions SHS toward the multi-modal (tabular + imaging) differentiator.
**Status:** MVP. Phase 1 (tabular risk) built and tested; phase 2 (MRI severity) implemented and tested on synthetic data. Multi-modal fusion, auth, Docker, model registry, CI deliberately deferred.
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
- **Phase-2 MRI confirmation:** ImageNet-pretrained ResNet18, frozen backbone + retrained head, 4-class severity (Non Demented / Very mild / Mild / Moderate Dementia); inverse-frequency class weights for imbalance; runs on CPU, uses GPU when present.
- **Thin service layer:** FastAPI `/predict` + `/health` wrapping the same seam; `load_model` is `@lru_cache`'d so the model loads once per process.

## Slide 6: Architecture details
**Layer 1 — User interface and interaction:**
- Streamlit demo ([app/streamlit_app.py](app/streamlit_app.py)) — "Clinical risk" tab (patient form → score + drivers) and "MRI severity" tab (scan upload → severity + probability bar chart).
- FastAPI service ([app/api.py](app/api.py)) — `POST /predict` (Pydantic-validated `PatientRecord`) and `GET /health`.
- CLIs: [train.py](train.py), [train_mri.py](train_mri.py), [evaluate_mri.py](evaluate_mri.py).

**Layer 2 — System logic ([src/alz/](src/alz/)):**
- `data.py` — OASIS-2 load/validate/clean; defines `FEATURE_COLUMNS`; `record_to_frame()` turns a patient dict into a feature row.
- `model.py` — build/train/save/load pipeline; `predict()` → `{score, label}`.
- `explain.py` — `top_drivers()` coefficient-based factor ranking.
- `imaging.py` — phase-2 ResNet18: `train_mri`, `evaluate_mri`, `predict_mri` / `predict_mri_probs`; shared 70/15/15 split so eval matches training.
- `__init__.py` — public `predict()` seam composing the above.

**Layer 3 — Data sources:**
- Local tabular CSV `data/oasis_longitudinal.csv` (synthetic sample).
- Kaggle imaging dataset `ninadaithal/imagesoasis` via `download_oasis.py` (kagglehub cache).
- Saved model artifacts: `models/model.joblib`, `models/mri_model.pt`.

**Example I/O (from [tests/test_api.py](tests/test_api.py) / README):**
```json
// POST /predict  request
{"Age":75,"EDUC":12,"SES":3,"MMSE":27,"nWBV":0.75,"ASF":1.2,"Visit":1,"MR Delay":0,"M/F":"F"}
// response
{"score":0.12,"label":"Normal","drivers":[{"feature":"MMSE","direction":"lowers risk"}, ...]}
```

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
