# Research Statement (OASIS-3 data access application)

Note: this is written for an OASIS-3 request. Reuse for other applications by swapping the dataset name and the "gap" it fills.

## Aims

This project (Alzheimer's Early-Risk Triage) aims to score Alzheimer's disease risk from routine, already-collected clinical data before an expensive MRI/PET scan is ordered, then confirm and refine that risk using imaging and EEG as second and third opinions, fusing all available modalities into a single explainable prognosis. The model deliberately excludes the Clinical Dementia Rating (CDR) as a predictor to avoid diagnosis leakage.

The current pipeline is built on OASIS-2 (tabular/longitudinal), a Kaggle mirror of OASIS MRI slices, and the OpenNeuro AHEPA EEG dataset (ds004504). None of these pair tabular, MRI, and EEG data on the same patients, and the tabular cohort is small (82 visits, partly synthetic) and narrow. Access to OASIS-3 would provide a larger, longitudinal, more clinically representative cohort to validate the trained models and calibrate the fusion weights against real multi-modal ground truth, rather than the current small/disjoint samples.

## Proposed Research Methods

Four pipelines are already implemented and would be applied to OASIS-3 data:

- **Tabular clinical risk model** (`src/alz/data.py`, `model.py`): mean imputation and standardization of clinical/demographic features, followed by logistic regression, with per-patient explanations via coefficient x scaled-value ranking.
- **MRI severity classification** (`src/alz/imaging.py`): transfer learning on a ResNet18 backbone (frozen ImageNet weights, retrained final layer) to classify severity into four classes, with Grad-CAM for pixel-level attribution.
- **EEG risk model** (`src/alz/eeg.py`): relative band-power features (delta, theta, alpha, beta, gamma) extracted via the Welch method, plus theta/alpha and slowing-ratio features, fed into a standardized logistic regression trained with 5-fold stratified cross-validation.
- **Multi-modal fusion** (`src/alz/fusion.py`): a logarithmic opinion pool that converts each modality's probability to a logit, takes a weighted average, and maps back to a probability, combining tabular, imaging, and EEG risk estimates into one prognosis.

## Variables of Interest

- **Demographics**: age, sex, years of education, socioeconomic status.
- **Cognitive/clinical scores**: MMSE (as a predictor); CDR (used only for outcome validation, not as a model input, to avoid diagnosis leakage).
- **MRI-derived measures**: normalized whole-brain volume (nWBV), atlas scaling factor (ASF), and raw structural MRI slices for severity classification.
- **Longitudinal/visit data**: visit number and inter-visit delay, to support tracking risk change over time.
- **PET/amyloid data** (if available in OASIS-3): would be a valuable addition as a further modality for the fusion model, extending beyond the current tabular/MRI/EEG scope.
