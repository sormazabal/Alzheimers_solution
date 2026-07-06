# Alzheimer's Early-Risk Triage

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
**The pitch:** Every unnecessary MRI or PET scan is money and scanner time the client can't get back, and every Alzheimer's case caught late is a patient the health system could have helped sooner. This product closes that gap: it scores Alzheimer's risk from data the clinic already has on file, days before anyone books a scan, and hands the clinician a second and third opinion (MRI, EEG) plus an AI-drafted case summary the moment those results come in.

**Why now:** Alzheimer's diagnosis today is imaging-first and symptom-triggered — expensive, slow, and reactive. This flips the order: cheap data in, risk score out, imaging reserved for the patients who actually need it.

**What the client gets:**

- **Lower cost per diagnosis** — imaging ordered by evidence, not by default, cutting unnecessary scans and freeing scanner capacity for patients who need it.
- **Earlier intervention** — risk surfaces from data already sitting in the chart, before symptoms force the issue.
- **A tool clinicians will actually trust** — every score comes with a plain "why," not a black-box number; every LLM claim is grounded in the patient's own record and cited literature.
- **Zero integration friction** — one call, `alz.predict()`, drops into any Siemens product (UI, API, notebook) with no bespoke glue code.
- **The differentiator the client asked for, delivered** — multi-modal fusion of clinical, MRI, and EEG signal into a single prognosis is built and working today, not a roadmap promise.

**Status:** MVP+ — working end to end. Clinical risk scoring, MRI severity confirmation, and EEG confirmation are each built and tested; a fusion layer combines whichever are available into one prognosis; an AI decision-support layer generates doctor-facing summaries and pulls live supporting evidence. Production-hardening items (auth, deployment, CI, live EHR feed) are the known, scoped remainder — not open questions.
**Category:** Clinical / digital-health machine learning.

## Slide 4: Client pains and system challenges
**Client pain points:**

- Imaging is the most expensive, most capacity-constrained step in the diagnostic path, every scan ordered without prior evidence is a scan that could have gone to a higher-priority patient.
- Diagnosis usually happens after symptoms are already visible. The data that could have flagged risk earlier is sitting unused in the chart.
- Existing risk tools are black boxes, which makes clinicians reluctant to act on them and makes them hard to certify and integrate across a product line.

**Why this is harder than "just train a model" — the scientific challenges:**

- **Target leakage:** the clinical dementia rating (CDR) is not a predictor of diagnosis, it *is* the diagnosis. The reference approach trained on it anyway, which inflates accuracy on paper while producing a tool that cannot run before a diagnosis exists. Removing it is correct, but it also removes the single strongest feature, so the honest model has to work harder for a lower, more defensible number.
- **Near-ceiling accuracy is a warning sign, not a win:** the current MRI classifier reports 1.00 accuracy on held-out data. For a frozen-backbone, head-only model on an imbalanced dataset, that is far more likely to reflect subject-level leakage across the train/val/test split (the same patient's slices appearing in more than one split) or a distribution too easy to be representative, than genuine clinical-grade generalization. This has to be treated as unproven until verified on a subject-disjoint, external cohort.
- **Small, synthetic, or narrow samples:** the tabular model is validated on 82 synthetic visits and the EEG model on 65 real recordings. Neither sample size supports a confidence interval tight enough for a clinical claim; both are consistent with a wide range of true performance, including much worse than the point estimate shown.
- **No paired ground truth for fusion:** the log-opinion-pool fusion combines clinical, MRI, and EEG scores mathematically, but no dataset in hand has all three modalities on the same patients. The fusion logic is sound, but its output has never been checked against a real, jointly-labeled case — it is unvalidated as a combined predictor, independent of how well each modality performs alone.
- **Explanation is not causation:** `top_drivers()` reports which features moved *this model's* score, not which factors cause Alzheimer's risk in the patient. Presenting coefficient weight as a clinical "reason" without that caveat risks overstating what the tool actually knows.
- **Population mismatch:** OASIS (tabular and imaging) and the AHEPA EEG cohort are specific, public research populations, not necessarily representative of the client's own patient demographics. A model earns its numbers on one distribution and has not yet been asked to generalize to another.


## Slide 5: Proposed solution
**The one-line pitch:** a single, honest risk score today, confirmed by imaging and EEG when needed, explained in plain language, reachable from anywhere with one function call.

- **Effortless to integrate:** one seam, `alz.predict()`, used identically by the clinician-facing app, the REST API, and every automated test — any Siemens system plugs in without custom glue.
- **Trustworthy by construction:** every risk score ships with its top 3 reasons in plain language ("raises/lowers risk"), because a number clinicians can't interrogate is a number they won't act on.
- **No diagnosis leakage:** the model is barred from ever seeing the clinical dementia rating, so it earns its accuracy on real pre-diagnosis signal instead of cheating off the answer.
- **A second opinion from imaging:** when a scan exists, the system classifies dementia severity from the MRI itself and highlights the exact regions that drove the call — confirmation a clinician can visually check, not just take on faith.
- **A third, independent signal from EEG:** a low-cost, already-collected recording adds a further, independent read on Alzheimer's-pattern brain activity — another vote before anyone commits to expensive imaging.
- **One prognosis, not three conflicting numbers:** clinical, MRI, and EEG signals are fused into a single score that automatically leans on whichever modality is most confident — the differentiator the client asked for, delivered.
- **An AI layer that does the busywork a clinician shouldn't have to:** an automatic case summary, plain-language MRI findings, free-form Q&A on the case, and live pulls from PubMed and ClinicalTrials.gov with citations — all optional, all fail-safe (it degrades gracefully rather than breaking the tool if the AI call fails).

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
