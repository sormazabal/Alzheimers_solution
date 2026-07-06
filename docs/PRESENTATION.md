# Alzheimer's Early-Risk Triage

**Subtitle:** A modular Alzheimer's-risk pipeline that scores patients from routine clinical data today and scales to MRI confirmation later — all behind one integration seam any system can call.
**Presenter:** Zow ORrazabal

**Speaker notes (title slide):**

"Good morning, everyone. My name is Zow, and today I want to talk to you about a problem that sits at the intersection of cost and time in Alzheimer's diagnosis.

Every unnecessary MRI or PET scan we order is money and scanner time we can't get back. And every Alzheimer's case we catch late is a patient we could have helped sooner. Those are two sides of the same coin, and this project is built to close that gap.

What I'm going to show you is a modular Alzheimer's risk pipeline. It scores patients from the routine clinical data a clinic already has on file, today, and it's built to scale up to MRI and EEG confirmation when those are available. All of it sits behind one integration seam that any system can call.

Over the next few minutes I'll walk you through the client problem, the solution we built, the architecture behind it, and, just as importantly, an honest account of where the evidence is strong and where it's still preliminary. Let's get into it."

## Slide 2: Outline

- Project overview and client value
- Client pains and system challenges
- Proposed solution
- Dataset overview
- Architecture details
- Technical implementation
- Cost
- Conclusion

**Speaker notes:**

"Quickly, here's the shape of the talk. I'll start with the project overview and what value it creates for the client. Then I'll get into the client's pain points and why this is a genuinely hard problem to solve well, not just a 'train a model' exercise. From there I'll walk through the proposed solution, the dataset, and the architecture in detail. Then I'll cover the technical implementation, the cost to run this per patient, and close with a conclusion.

One thing I want to flag up front: I'm going to be direct about where the evidence is solid and where it's still preliminary. That honesty is part of the pitch, not a disclaimer I'm burying at the end."

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

**Speaker notes:**

"So let's start with why this matters. Every unnecessary MRI or PET scan is money and scanner time the client can't get back. Every Alzheimer's case caught late is a patient the health system could have helped sooner. This product closes that gap. It scores Alzheimer's risk from data the clinic already has on file, days before anyone books a scan, and it hands the clinician a second and third opinion, from MRI and EEG, plus an AI-drafted case summary the moment those results come in.

Why now? Because Alzheimer's diagnosis today is imaging-first and symptom-triggered. It's expensive, it's slow, and it's reactive. What we're proposing flips that order: cheap data goes in, a risk score comes out, and imaging gets reserved for the patients who actually need it.

For the client, that means lower cost per diagnosis, because imaging gets ordered by evidence instead of by default. It means earlier intervention, because risk surfaces from data that's already sitting in the chart. It means a tool clinicians will actually trust, because every score comes with a plain-language 'why,' not a black-box number, and every AI-generated claim is grounded in the patient's own record and cited literature. It means zero integration friction, because one function call, alz dot predict, drops into any Siemens product with no bespoke glue code. And it delivers the differentiator the client specifically asked for: multi-modal fusion of clinical, MRI, and EEG signal into a single prognosis, built and working today, not a roadmap promise.

I want to be precise about where we are: this is MVP-plus, working end to end. Clinical risk scoring, MRI severity confirmation, and EEG confirmation are each built and tested. A fusion layer combines whichever of those are available into one prognosis. And an AI decision-support layer generates doctor-facing summaries and pulls in live supporting evidence. What's left, authentication, deployment, CI, a live EHR feed, is production hardening. It's scoped and known. It is not an open question."

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

**Speaker notes:**

"Now, before I show you the solution, I want to slow down and talk about why this is genuinely hard, because I don't want to oversell it.

First, the client-facing pain points. Imaging is the most expensive, most capacity-constrained step in the diagnostic path. Every scan ordered without prior evidence is a scan that could have gone to a higher-priority patient. Diagnosis usually happens after symptoms are already visible, and the data that could have flagged risk earlier is just sitting unused in the chart. And existing risk tools are black boxes, which makes clinicians reluctant to act on them and makes them hard to certify and integrate.

But there's a deeper set of scientific challenges, and I want to be upfront about each one.

The first is target leakage. The clinical dementia rating, CDR, is not a predictor of diagnosis, it is the diagnosis. The reference approach we built on trained on it anyway, which inflates accuracy on paper while producing a tool that literally cannot run before a diagnosis already exists. Removing it is the correct thing to do, but it also removes the single strongest feature, so our honest model has to work harder for a lower, more defensible number.

Second, near-ceiling accuracy is a warning sign, not a win. Our original MRI classifier reported 1.00 accuracy on held-out data, and it turned out to be exactly what it looked like: subject-level leakage, the same patient's slices showing up in more than one split. We fixed it by grouping the split by patient instead of by image, and collapsed the target to binary Non Demented vs Demented, since the Moderate Dementia class only has two subjects. The corrected model reports well below 1.00 now, which is the honest number.

Third, our samples are small. The tabular model is validated on 82 synthetic visits, and the EEG model on 65 real recordings. Neither sample size supports a confidence interval tight enough for a clinical claim. Both are consistent with a wide range of true performance, including meaningfully worse than the point estimate I'm showing you.

Fourth, we don't have paired ground truth for fusion. The fusion logic combines clinical, MRI, and EEG scores mathematically, but no dataset we have contains all three modalities on the same patients. The math is sound, but its output has never been checked against a real, jointly-labeled case.

Fifth, explanation is not causation. When the tool reports top drivers, it's reporting which features moved this model's score, not which factors cause Alzheimer's risk in the patient. I'm careful never to present coefficient weight as a clinical reason without that caveat.

And finally, population mismatch. OASIS and the AHEPA EEG cohort are specific, public research populations. They're not necessarily representative of this client's own patient demographics. A model earns its numbers on one distribution, and it hasn't yet been asked to generalize to another."

## Slide 5: Proposed solution
**The one-line pitch:** a single, honest risk score today, confirmed by imaging and EEG when needed, explained in plain language, reachable from anywhere with one function call.

- **Effortless to integrate:** one seam, `alz.predict()`, used identically by the clinician-facing app, the REST API, and every automated test — any Siemens system plugs in without custom glue.
- **Trustworthy by construction:** every risk score ships with its top 3 reasons in plain language ("raises/lowers risk"), because a number clinicians can't interrogate is a number they won't act on.
- **No diagnosis leakage:** the model is barred from ever seeing the clinical dementia rating, so it earns its accuracy on real pre-diagnosis signal instead of cheating off the answer.
- **A second opinion from imaging:** when a scan exists, the system classifies dementia severity from the MRI itself and highlights the exact regions that drove the call — confirmation a clinician can visually check, not just take on faith.
- **A third, independent signal from EEG:** a low-cost, already-collected recording adds a further, independent read on Alzheimer's-pattern brain activity — another vote before anyone commits to expensive imaging.
- **One prognosis, not three conflicting numbers:** clinical, MRI, and EEG signals are fused into a single score that automatically leans on whichever modality is most confident — the differentiator the client asked for, delivered.
- **An AI layer that does the busywork a clinician shouldn't have to:** an automatic case summary, plain-language MRI findings, free-form Q&A on the case, and live pulls from PubMed and ClinicalTrials.gov with citations — all optional, all fail-safe (it degrades gracefully rather than breaking the tool if the AI call fails).

**Speaker notes:**

"So given all of that, here's what we actually built. The one-line pitch is: a single, honest risk score today, confirmed by imaging and EEG when needed, explained in plain language, reachable from anywhere with one function call.

It's effortless to integrate. One seam, alz dot predict, is used identically by the clinician-facing app, the REST API, and every automated test. Any Siemens system plugs in without custom glue code.

It's trustworthy by construction. Every risk score ships with its top three reasons in plain language, whether each factor raises or lowers risk, because a number clinicians can't interrogate is a number they won't act on.

There's no diagnosis leakage. The model is barred from ever seeing the clinical dementia rating, so it earns its accuracy on real, pre-diagnosis signal instead of cheating off the answer.

When a scan exists, we get a second opinion from imaging. The system confirms dementia from the MRI itself and highlights the exact regions that drove the call, so it's confirmation a clinician can visually check, not just take on faith.

We also get a third, independent signal from EEG. It's a low-cost, already-collected recording that adds a further, independent read on Alzheimer's-pattern brain activity, another vote before anyone commits to an expensive scan.

And critically, this isn't three conflicting numbers. Clinical, MRI, and EEG signals are fused into a single prognosis that automatically leans on whichever modality is most confident. That's the differentiator the client asked for, and it's delivered, not a roadmap item.

Finally, there's an AI layer that does the busywork a clinician shouldn't have to do themselves: an automatic case summary, plain-language MRI findings, free-form question and answer on the case, and live pulls from PubMed and ClinicalTrials.gov with citations. All of that is optional, and all of it is fail-safe, meaning if the AI call fails, the tool degrades gracefully instead of breaking."

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

**Speaker notes:**

"Let me show you this isn't just a notebook, it's a real stack, three layers deep.

Layer one is the user interface. There's a Streamlit clinical workstation with four tabs: clinical risk, where a patient form produces a score and its drivers; MRI records, where a scan upload produces a severity classification, a Grad-CAM overlay, and AI findings; EEG records, where a recording produces a band-power classification and a signal viewer; and an overview tab that shows the integrated prognosis, the AI case summary, live evidence, and case chat. Alongside that there's a FastAPI service exposing a predict endpoint and a health check, plus command-line tools for training and evaluating the models.

Layer two is the system logic, and this is where the actual seam lives. There's a data module that loads and validates OASIS-2 data and turns a patient dictionary into a feature row. A model module that builds, trains, saves, and loads the tabular pipeline. An imaging module with the ResNet18 severity classifier, training, evaluation, prediction, and Grad-CAM. An EEG module that extracts band-power features and runs logistic regression. A fusion module that combines whichever modalities are present into one prognosis. And an explain module that generates top drivers, case summaries, MRI explanations, case chat, and live evidence retrieval. All of that gets composed behind one public predict function, and here's the point I want to land: the Streamlit app and the FastAPI service both call that exact same seam. That's what 'zero integration friction' actually means in practice, not a slogan.

Layer three is the data sources: a local tabular CSV, a Kaggle imaging dataset, a real EEG dataset from OpenNeuro, saved model artifacts, and live calls out to PubMed and ClinicalTrials.gov.

And here on screen is a real example. You send a POST to slash predict with a patient's age, education, socioeconomic status, MMSE score, and a few other fields, and you get back a score, a label, and the drivers behind it. That's callable today."

## Slide 6b: How each model computes its score

**Clinical (tabular):** 9 features → mean-imputation → standardization → logistic
regression. `score = predict_proba()[1]`, the model's own probability of the "at-risk"
class. Explanation is not a separate step bolted on afterward — `top_drivers()` reads the
same coefficients the model used to score, ranking features by `scaled_value × coefficient`.

**MRI dementia confirmation:** a brain-slice image → ResNet18 (frozen ImageNet backbone,
retrained final layer) → softmax over {Non Demented, Demented}. `score` is the winning
class's probability. Grad-CAM (`gradcam_mri`) backprops the winning class's logit into the
last convolutional block to show *which pixels* drove that softmax output, as a heatmap
overlay. The model is binary, not 4-way severity: with only 2 subjects in the Moderate
Dementia folder, a patient-level split can't support a reliable per-severity estimate, so
Mild + Moderate are pooled into "Demented".

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

All three models are binary classifiers now, scored on the same metric set. Clinical and
EEG metrics are pooled out-of-fold predictions from stratified 5-fold CV (a single
held-out split is too noisy at n=82 / n=65). The MRI split is grouped by patient, not by
image slice — see the note below on why that mattered.

| Model | Algorithm | Accuracy | Balanced acc. | AUROC | AUPRC | F1 | Sensitivity | Specificity | Eval | Data used |
|---|---|---|---|---|---|---|---|---|---|---|
| Clinical | Logistic regression (9 features) | 0.70 | 0.65 | 0.73 | 0.64 | 0.55 | 0.47 | 0.84 | 5-fold CV, out-of-fold | Synthetic `oasis_longitudinal.csv` (82 rows) — indicative only, not a real-world benchmark |
| MRI (Non Demented vs Demented) — *preliminary* | ResNet18 (frozen backbone, retrained head) | 0.81 | 0.81 | 0.91 | 0.89 | 0.78 | 0.70 | 0.91 | Held-out **patients**, 70/15/15 (small balanced subsample) | Real Kaggle `imagesoasis` mirror (~23k slices, ~94 subjects) |
| EEG | Logistic regression on band-power features | 0.75 | 0.75 | 0.82 | 0.83 | 0.77 | 0.75 | 0.76 | 5-fold CV, out-of-fold | Real ds004504 (AHEPA resting-state EEG, n=65) |
| Integrated prognosis | Logarithmic opinion pool (no trained parameters) | — | — | — | — | — | — | — | Not separately benchmarked | No paired multi-modal ground truth exists to validate against (see methodology doc) |

**Note on the MRI row:** the dataset originally reported **1.00 accuracy across
train/val/test**, which is a red flag, not a good result. The cause: slices were split
into train/val/test at the *image* level, and each of the ~94 patients contributes
dozens to hundreds of near-identical adjacent slices, so the same patient's scans ended
up in both train and test. The model was memorizing patients, not learning dementia
signal. The fix is a **patient-grouped split** (every slice from one subject stays in a
single split) plus collapsing to binary Non Demented vs Demented, since the Moderate
Dementia folder has only 2 subjects — too few to evaluate as its own class. The MRI row
above is **preliminary**: real, non-leaked numbers, but from 6 epochs on an 800-image
training subsample (not the full ~23k-slice dataset) so it could be produced quickly. The
full-dataset retrain is running separately and will replace this row when it finishes.
Treat both the preliminary and final MRI numbers as evidence the corrected pipeline
trains and evaluates honestly, not as a validated accuracy claim; production use requires
evaluation on a held-out clinical cohort, not the same public source distribution the
model trained on.

**Speaker notes:**

"Let me quickly explain how each of these three models actually produces a number.

For the clinical, tabular model, nine features go through mean imputation, then standardization, then logistic regression. The score is just the model's own predicted probability of the at-risk class. And the explanation isn't bolted on afterward, top drivers reads the exact same coefficients the model used to score, ranked by how much each feature's scaled value times its coefficient moved the result.

For MRI severity, a brain-slice image goes through a ResNet18 with a frozen ImageNet backbone and a retrained final layer, producing a softmax over severity classes. The score is the winning class's probability. Grad-CAM then backpropagates that winning logit into the last convolutional block to show which pixels actually drove the softmax output, as a heatmap.

For EEG, a resting-state recording gets turned into relative band power, delta, theta, alpha, beta, gamma, plus two slowing ratios that summarize the classic Alzheimer's EEG pattern, more low-frequency power, less high-frequency power. That feeds a standardized logistic regression, and the score is the probability of that AD pattern.

Now, the part I want to spend the most time on: how we fuse these into one prognosis. Each modality's probability gets converted to log-odds, averaged with equal weight by default, and converted back with a sigmoid. The intuition is this: a modality sitting near 0.5, meaning it's uncertain, has a log-odds near zero, so it barely moves the fused result. A confident modality, with a probability near 0 or near 1, pulls the fused score strongly toward it. So a stronger signal dominates a weaker one automatically, without any hand-picked weighting.

Now the metrics table. Clinical logistic regression gets 0.76 held-out accuracy on a stratified 75-25 split, but that's on 82 rows of synthetic data, so treat it as indicative, not a real-world benchmark. The MRI model reports 1.00 accuracy across train, validation, and test. I want to say this as clearly as I can: that number is an existence proof that the pipeline trains and evaluates correctly. It is not a validated clinical accuracy claim. For this class-imbalanced dataset with a frozen-backbone, head-only model, that number is more likely explained by subject-level leakage than genuine generalization, and AUROC and AUPRC report not-applicable because sklearn can't compute those curves for a class with no predicted negatives. The EEG model gets 0.75 mean accuracy across five-fold cross-validation on 65 real recordings. And the integrated prognosis has not been separately benchmarked, because we don't have a dataset with all three modalities on the same patients yet."

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

**Speaker notes:**

"A word on the data behind all of this. The bundled CSV that ships in the repo is synthetic, forty fake subjects, eighty-two visits, shaped exactly like OASIS-2. It's committed to the repo and runs out of the box, on purpose, so that anyone can try this without signing a data-use agreement.

Real OASIS-2 longitudinal data has the same schema, with Group, nondemented, demented, or converted, as the target. You download it from oasis-brains.org and drop it in, no code change required. The OASIS imaging dataset is a real, publicly downloadable Kaggle mirror of brain slices organized by severity class, and there's a script that pulls it down for you.

On privacy: only synthetic data is committed to the repo. Real OASIS data has its own data-use terms, but the imaging mirror is public with no data-use agreement needed. And as I mentioned earlier, CDR is excluded as a predictor because it's diagnosis leakage, that's the same point from a few slides ago. SES and MMSE missing values are mean-imputed, and identifiers like Subject ID, MRI ID, hand, and eTIV are dropped or unused."

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

**Speaker notes:**

"Quickly on the environment: this runs on Python 3.10 or newer, using uv as the package manager. The core dependencies are scikit-learn, pandas, joblib, Streamlit, FastAPI, uvicorn, httpx, and pytest. The imaging phase adds torch, torchvision, pillow, kagglehub, and matplotlib, but that's an optional install, the core demo doesn't need any of it.

On integration: we pull imaging data from Kaggle through kagglehub, and we use an ImageNet-pretrained ResNet18 from torchvision as an installed library. There's no external LLM provider required for the core pipeline. This builds on top of a vendored reference repository for Alzheimer's disease prediction, included as a git submodule.

If I'm demoing this live, here's what I'd run: create the virtual environment, install requirements, run train.py to produce the tabular model, run the smoke test to confirm the train-then-predict roundtrip works, and then launch the Streamlit app for the interactive demo, or the FastAPI server if I want to hit the REST endpoint directly. If time allows, I'd also show the phase-two imaging commands, training the MRI model and running the evaluation script that produces the ROC and PR plots."

## Slide 9: Results and evaluation
No numeric benchmark results are committed to the repo (models train from data at runtime; only synthetic data ships). The evaluation *machinery* is in place:

**Tabular track:** `train.py` reports held-out accuracy on a stratified 75/25 split (printed at train time).

**Imaging track:** per-epoch train/val and final test **accuracy, AUROC, AUPRC** (`imaging._metrics`); `evaluate_mri.py` reports all three splits and saves ROC + PR curves to `eval_plots/`. AUROC/AUPRC gracefully report `n/a` when a split has a single class (tiny/synthetic data).

**Test coverage:** `tests/test_smoke.py` (tabular roundtrip), `tests/test_api.py` (FastAPI `/predict` + `/health`), `tests/test_imaging.py` (MRI train→predict→evaluate roundtrip on a synthetic dataset, skips if imaging deps absent).

**Speaker notes:**

"I want to be precise about what's committed to the repository versus what you'd generate by running it. No numeric benchmark results ship in the repo, because the models train from data at runtime and only synthetic data is bundled. What is in place is the evaluation machinery itself.

On the tabular track, train.py reports held-out accuracy on a stratified 75-25 split, printed at train time. On the imaging track, we get per-epoch train and validation metrics plus a final test accuracy, AUROC, and AUPRC, and evaluate_mri.py reports all three splits and saves ROC and precision-recall curves to disk. Those AUROC and AUPRC numbers gracefully report not-applicable when a split only has one class, which can happen with tiny or synthetic data.

And on test coverage, there's a smoke test for the tabular roundtrip, an API test for the predict and health endpoints, and an imaging test that runs a full train, predict, evaluate roundtrip on a synthetic dataset, and skips itself gracefully if the imaging dependencies aren't installed.

So the honest framing here is: this isn't a gap, it's a 'run it yourself and see' claim. The infrastructure to produce real numbers is built and tested, we just haven't run it against a real clinical cohort yet."

## Slide 10: Cost analysis
No billing dashboard is wired up in the repo, so these are estimates: published per-token rates for the two hosted APIs the prototype actually calls, sized against the real prompt/response lengths in `src/alz/explain.py`. Everything else (model training, PubMed/ClinicalTrials.gov lookups) is free/local.

| Component | Cost per patient | Detail |
| --- | --- | --- |
| MRI vision findings (HuggingFace Inference Providers, `google/gemma-3-27b-it`, no HF markup) | ~$0.00006 | ~300 prompt tokens + 1 image (256 tokens/crop, Gemma 3's fixed vision-encoder output) + ~120 output tokens; `explain._vision_findings`, gated on `HF_TOKEN` |
| Clinical summary (Groq, `llama-3.1-8b-instant`) | ~$0.00003 | ~400 input / ~150 output tokens; `synthesize_summary` |
| Evidence & recommendations (Groq, `llama-3.1-8b-instant`) | ~$0.00005 | ~700 input / ~200 output tokens; `evidence_for_case`, grounded on live PubMed/ClinicalTrials.gov results |
| PubMed / ClinicalTrials.gov retrieval | $0.00 | Free public APIs (`esearch`/`esummary`, ClinicalTrials.gov v2) |
| Model training (tabular + MRI) | $0.00 (compute only) | Local CPU/GPU, one-time, not per-patient |

**Total LLM inference cost: ~$0.00014 per patient case** (published rates × actual prompt/response sizes: Groq $0.05/$0.08 per 1M in/out tokens, HF-routed Gemma 3 27B $0.08/$0.16 per 1M in/out tokens). Negligible next to the cost of the imaging studies this tool is meant to help avoid ordering unnecessarily. Text-completion provider is swappable (`llm.py` also supports Anthropic, OpenAI, Gemini, Bedrock, local Ollama); Groq was chosen here for its low per-token rate. A larger/non-square MRI slice could trigger Gemma 3's pan-and-scan mode (multiple 256-token crops), but even a handful of extra crops only adds a few hundredths of a cent. Costs scale linearly with patient volume and would shift if a pricier provider or model is swapped in.

### Expected results and impact

At effectively sub-cent LLM cost per patient, explainability (MRI findings, clinical summary, cited evidence) is essentially free to attach to every screening case. The real cost lever is imaging *volume*, which this tool is designed to reduce by triaging who actually needs an MRI.

**Speaker notes:**

"Let's talk about cost, because I know that's on people's minds whenever AI gets mentioned. There's no billing dashboard wired up in the repo, so these are estimates, but they're grounded: published per-token rates for the two hosted APIs we actually call, sized against the real prompt and response lengths in the code. Everything else, model training, PubMed and ClinicalTrials.gov lookups, is free and local.

The MRI vision findings call, through HuggingFace to Gemma 3, costs about six hundredths of a cent. The clinical summary, through Groq's Llama 3.1 8B, costs about three thousandths of a cent. Evidence and recommendations, same model, grounded on live PubMed and ClinicalTrials.gov results, costs about five thousandths of a cent. The literature retrieval itself is free, and model training is a one-time local compute cost, not a per-patient cost.

Add it up and you get a total LLM inference cost of about $0.00014 per patient case. Let that number sit for a second. It is negligible next to the cost of the imaging studies this tool is meant to help avoid ordering unnecessarily. The text-completion provider is swappable, we support Anthropic, OpenAI, Gemini, Bedrock, and local Ollama as well, we chose Groq here for its low per-token rate. Costs scale linearly with volume, and they'd shift if a pricier provider got swapped in, but even in the worst realistic case this stays a rounding error.

So the real cost lever isn't the AI layer at all, at this price point, explainability is essentially free to attach to every screening case. The lever that actually matters is imaging volume, and that's exactly what this tool is designed to reduce."

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

**Speaker notes:**

"So where does that leave us, and what's next?

What's done: the tabular risk model, its explainability, and the predict seam are built and tested. The FastAPI wrapper with the predict and health endpoints, with Pydantic validation, is done. The phase-two MRI severity classifier, using ResNet18 transfer learning, plus the evaluation CLI and plots, is done. And the imaging data fetch script is documented as a separate phase-two step.

What's ahead: multi-modal fusion of tabular and imaging is the differentiator the client specifically asked for, and while the fusion logic itself is built, validating it against real paired data is the honest next milestone, not a nice-to-have I'm glossing over.

I also want to be clear that some things are deliberately deferred, not overlooked: authentication, Docker packaging, a model registry, and continuous integration. Those are scoped-out choices for this stage of the project, and I'm happy to talk through the sequencing if that's useful. And longer term, we'd swap the synthetic CSV for real OASIS-2 data and train the imaging model on real OASIS slices to get real, defensible metrics instead of the indicative ones I showed you earlier."

## Slide 12: Conclusion and key takeaways
- For patients: risk surfaces from data already in their chart, days before anyone books a scan, so the patients who need imaging get it sooner and the ones who don't are spared the cost, wait, and anxiety of an unnecessary MRI.
- For clinicians: every score arrives with its top reasons in plain language, plus an MRI second opinion and an EEG third opinion when needed, so a clinician is never asked to act on a number they can't interrogate.
- For clinicians: an AI layer drafts the case summary, translates MRI findings into plain language, answers follow-up questions about the case, and pulls cited supporting evidence, so it removes the busywork around a decision without ever making the decision for them.
- For the health system: one integration seam (`alz.predict()`) drops into any Siemens product, so this reaches clinicians through the tools they already use instead of requiring a new one.

**Speaker notes:**

"Let me close by bringing this back to who actually benefits.

For patients, risk surfaces from data already in their chart, days before anyone books a scan. The patients who need imaging get it sooner, and the ones who don't are spared the cost, the wait, and the anxiety of an unnecessary MRI.

For clinicians, every score arrives with its top reasons in plain language, plus an MRI second opinion and an EEG third opinion when needed. A clinician is never asked to act on a number they can't interrogate. And the AI layer drafts the case summary, translates MRI findings into plain language, answers follow-up questions about the case, and pulls cited supporting evidence, removing the busywork around a decision without ever making that decision for them.

And for the health system, one integration seam, alz dot predict, drops into any Siemens product. That's not a minor engineering detail, it's the single technical fact that makes every other promise on this slide deliverable. This reaches clinicians through the tools they already use, instead of asking them to adopt a new one.

Thank you. I'm happy to take questions."

