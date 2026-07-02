# Data

`oasis_longitudinal.csv` here is a **synthetic** sample (40 fake subjects, 82 visits) that mimics
the schema of the OASIS-2 longitudinal dataset, so the demo runs out of the box without any data
agreement.

To use real data, download the OASIS-2 longitudinal CSV from
[oasis-brains.org](https://www.oasis-brains.org/) and replace this file — same column names, no
code changes needed.

## Expected columns

`Subject ID, MRI ID, Group, Visit, MR Delay, M/F, Hand, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF`

- `Group`: `Nondemented` / `Demented` / `Converted` — the prediction target.
- `SES`, `MMSE` commonly have missing values; `Subject ID`, `MRI ID`, `Hand`, `MR Delay`, `eTIV`,
  `CDR` are dropped or unused as predictors (see [src/alz/data.py](../src/alz/data.py)), following
  the preprocessing in
  [vendor/alzheimers-disease-prediction/dimentia_pred.ipynb](../vendor/alzheimers-disease-prediction/dimentia_pred.ipynb).
