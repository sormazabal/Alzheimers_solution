# Data

The clinical model trains on real OASIS data in `OASIS1_raw/`:

- `oasis_cross-sectional-*.xlsx` (OASIS-1): one visit per subject. No `Group` column, so it is
  derived from CDR (0 = Nondemented, above 0 = Demented). `Educ` is a 1-5 category, mapped to
  approximate years. Rows without CDR/MMSE (young controls) are dropped.
- `oasis_longitudinal_demographics-*.xlsx` (OASIS-2): repeat visits, `Group` is the target.

`alz.data.load_combined()` stacks both (608 rows). `train.py` uses them by default and splits
cross-validation by `Subject ID`. These files have their own data-use terms, so do not commit them.

`oasis_longitudinal.csv` is a **synthetic** sample (40 fake subjects, 82 visits) kept only for the
demo patient list in `src/alz/db.py`. Pass it with `train.py --data` for a quick test.

## Phase-2 imaging data

`download_oasis.py` in this directory fetches the OASIS **imaging** dataset from Kaggle
(`ninadaithal/imagesoasis`) for the phase-2 MRI severity classifier
([src/alz/imaging.py](../src/alz/imaging.py)). It lands at
`data/imagesoasis/versions/1/Data/<class folders>` (kagglehub's cache layout) and is
unrelated to the tabular workflow above -- running it does not affect
`oasis_longitudinal.csv` or `train.py`. Train with `python train_mri.py` (see
[README.md](../README.md)).

## Expected columns

`Subject ID, MRI ID, Group, Visit, MR Delay, M/F, Hand, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF`

- `Group`: `Nondemented` / `Demented` / `Converted` — the prediction target.
- `SES`, `MMSE` commonly have missing values; `Subject ID`, `MRI ID`, `Hand`, `MR Delay`, `eTIV`,
  `CDR` are dropped or unused as predictors (see [src/alz/data.py](../src/alz/data.py)), following
  the preprocessing in
  [vendor/alzheimers-disease-prediction/dimentia_pred.ipynb](../vendor/alzheimers-disease-prediction/dimentia_pred.ipynb).
