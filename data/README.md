# Data

`oasis_longitudinal.csv` here is a **synthetic** sample (40 fake subjects, 82 visits) that mimics
the schema of the OASIS-2 longitudinal dataset, so the demo runs out of the box without any data
agreement.

To use real data, download the OASIS-2 longitudinal CSV from
[oasis-brains.org](https://www.oasis-brains.org/) and replace this file — same column names, no
code changes needed.

## Phase-2 imaging data

`download_oasis.py` in this directory fetches the OASIS **imaging** dataset from Kaggle
(`ninadaithal/imagesoasis`) for the phase-2 MRI severity classifier
([src/alz/imaging.py](../src/alz/imaging.py)). It lands at
`data/imagesoasis/versions/1/Data/<class folders>` (kagglehub's cache layout) and is
unrelated to the tabular workflow above -- running it does not affect
`oasis_longitudinal.csv` or `train.py`. Train with `python train_mri.py` (see
[README.md](../README.md)).

`imagesoasis/versions/1/Data/` holds one folder per class, each full of `.jpg` MRI
slices: `Mild Dementia`, `Moderate Dementia`, `Non Demented` (a `Very mild Dementia`
class also exists upstream on Kaggle if you re-download).

## Phase-3 EEG data

`ds004504/` is the OpenNeuro BIDS dataset "A dataset of EEG recordings from
Alzheimer's disease, Frontotemporal Dementia and Healthy subjects" (Miltiadous et
al.; originally `github.com/OpenNeuroDatasets/ds004504`). It feeds the phase-3 EEG
confirmation classifier ([src/alz/eeg.py](../src/alz/eeg.py)), which extracts relative
band-power features (delta/theta/alpha/beta/gamma) and fits a logistic regression.

88 subjects: 36 Alzheimer's (AD), 23 Frontotemporal Dementia (FTD), 29 healthy
controls (CN). Each `sub-0XX/eeg/` holds the raw resting-state eyes-closed recording
(`.set`, Nihon Kohden 19-channel montage, 500 Hz); `derivatives/sub-0XX/eeg/` holds
the same recording after band-pass filtering, ASR artifact correction, and ICA
cleaning. `participants.tsv` columns: `participant_id, Gender, Age, Group, MMSE`,
where `Group` is `A` (AD), `F` (FTD), or `C` (healthy control).

## Raw OASIS-1 MRI (supplementary)

`OASIS1_raw/` is the original OASIS-1 cross-sectional MRI release from
[oasis-brains.org](https://www.oasis-brains.org/), split across two discs:
`oasis_cross-sectional_disc1/disc1/` (39 subjects) and
`oasis_cross-sectional_disc2/disc2/` (38 subjects), plus a demographics spreadsheet
(`oasis_cross-sectional-*.xlsx`). Each `OAS1_XXXX_MR1/` subject folder has `RAW/`
(unprocessed scans), `PROCESSED/MPRAGE/` (registered/atlas-aligned volumes), and
`FSL_SEG/` (tissue segmentation), all in Analyze format (`.hdr`/`.img`). This is raw
source data -- no script in this repo currently reads it; it's kept for reference and
future MRI work beyond the Kaggle `imagesoasis` slices used today.

## Expected columns

`Subject ID, MRI ID, Group, Visit, MR Delay, M/F, Hand, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF`

- `Group`: `Nondemented` / `Demented` / `Converted` — the prediction target.
- `SES`, `MMSE` commonly have missing values; `Subject ID`, `MRI ID`, `Hand`, `MR Delay`, `eTIV`,
  `CDR` are dropped or unused as predictors (see [src/alz/data.py](../src/alz/data.py)), following
  the preprocessing in
  [vendor/alzheimers-disease-prediction/dimentia_pred.ipynb](../vendor/alzheimers-disease-prediction/dimentia_pred.ipynb).
