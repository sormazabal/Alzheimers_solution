# Alzheimer's Early-Risk MVP

Proof of capability for Siemens Digital Health / SHS AI: a modular Alzheimer's-risk pipeline that
runs from structured clinical data today, and is scaffolded to scale to MRI confirmation later —
all behind one integration seam any system can call.

## Use cases

1. **Early-risk triage from routine clinical data** *(built)* — cognitive scores + demographics →
   risk score + top contributing factors, before imaging is ordered.
2. **MRI-based confirmation** *(stubbed, phase 2)* — brain scan → CNN classification via
   MONAI/TorchIO/MedicalNet. See [src/alz/imaging.py](src/alz/imaging.py).
3. **Multi-modal fusion** *(future)* — combine tabular + imaging scores. The SHS differentiator;
   not built in this MVP.

## Run it

```
pip install -r requirements.txt
python train.py                          # trains models/model.joblib on data/oasis_longitudinal.csv
python -m pytest tests/test_smoke.py     # train -> predict roundtrip check
streamlit run app/streamlit_app.py       # interactive demo
```

## Integration seam

Any system (UI, API, notebook) calls the same function:

```python
from alz import predict
predict({"Age": 75, "EDUC": 12, "SES": 3, "MMSE": 27, "nWBV": 0.75, "ASF": 1.2,
          "Visit": 1, "MR Delay": 0, "M/F": "F"})
# -> {"score": 0.12, "label": "Normal", "drivers": [...]}
```

Wrapping this in a FastAPI service is a small, deliberately deferred next step.

## Repo structure

```
src/alz/          # data cleaning, model, explainability, imaging stub
train.py          # CLI: train + save the model
app/               # Streamlit demo
tests/             # smoke test
data/              # synthetic sample dataset (see data/README.md for real-data swap-in)
vendor/            # reference repos we build on (git submodule)
```

## What this builds on

- [HimanS-sys/alzheimers-disease-prediction](https://github.com/HimanS-sys/alzheimers-disease-prediction)
  (vendored under `vendor/`) — its OASIS-2 preprocessing is the basis for `src/alz/data.py`, and
  its 3D-CNN notebook is the reference template for the phase-2 imaging stub. One deliberate
  deviation: we drop `CDR` as a predictor, since CDR is itself the clinical diagnosis and using it
  to predict risk would leak the answer into a tool meant to screen patients *before* diagnosis.
- [GMattheisen/predicting_Alzheimers_from_MRI](https://github.com/GMattheisen/predicting_Alzheimers_from_MRI) —
  feature-selection / accuracy reference.
- [invcble/Alzheimer-s-Classification-using-OASIS-Dataset](https://github.com/invcble/Alzheimer-s-Classification-using-OASIS-Dataset) —
  reference for CDR-based severity bucketing (not used in this binary-risk MVP, but relevant if
  staging severity becomes a requirement).
- MONAI / TorchIO / MedicalNet — the real phase-2 basis (`requirements-imaging.txt`); used as
  installed libraries, not reimplemented.

These reference repos ship no license, so we adapt their approach in our own code rather than
copying files verbatim, and credit them here. Only a synthetic OASIS-shaped sample is committed;
real OASIS data has its own data-use terms (see [data/README.md](data/README.md)).

## Deliberately skipped for this MVP

- FastAPI wrapper — `alz.predict()` is already the seam; ~15 lines when needed.
- MRI/CNN training — stubbed; needs GPU + a data-use agreement.
- Multi-modal fusion, auth, Docker, model registry, CI.
