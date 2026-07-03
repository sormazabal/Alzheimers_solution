# Integrated Prognosis Score — Methodology & Justification

## Problem

Three independent models each produce a probability-like risk score for the
same patient (clinical/tabular, MRI imaging, EEG). There is no dataset that
pairs tabular + imaging + EEG records with a shared ground-truth label, so a
trained fusion model (e.g. a stacking classifier or joint neural net) cannot
be fit or validated. We still want one unified number for the Overview tab.

## Method: logarithmic opinion pool

Each modality is read as an independent estimate of P(dementia present):

- **Clinical**: `score` from `src/alz/model.py` — already P(at-risk) from the
  logistic regression's positive class.
- **MRI**: `1 − probs["Non Demented"]` from `src/alz/imaging.py` — complement
  of the softmax posterior for the healthy class.
- **EEG**: derived from the placeholder's `score`/`level` (flagged as a stub
  input; excluded once EEG has no result).

These probabilities are combined with a **logarithmic opinion pool** — a
weighted average in log-odds (logit) space, followed by a sigmoid:

```
logit(P_fused) = ( Σ_i r_i · logit(p_i) ) / ( Σ_i r_i )
P_fused        = sigmoid(logit(P_fused))
```

This is a standard, citable method for combining independent probabilistic
opinions without joint training data:

- Genest, C. & Zidek, J.V. (1986). "Combining Probability Distributions: A
  Critique and an Annotated Bibliography." *Statistical Science*, 1(1), 114-135.
- Kittler, J., Hatef, M., Duin, R.P.W., & Matas, J. (1998). "On Combining
  Classifiers." *IEEE Transactions on Pattern Analysis and Machine
  Intelligence*, 20(3), 226-239.

## Why this over a hand-set weighted average

An earlier draft used linear weights on raw scores (e.g. 0.5 MRI / 0.3
clinical / 0.2 EEG) with invented severity anchors (e.g. "Mild Dementia" →
0.5). Those constants had no basis — they were guesses dressed up as a
formula. The opinion pool avoids this in two ways:

1. **No invented anchors.** Each modality's *own* posterior is used directly;
   there is no hand-picked midpoint for "mild."
2. **Confidence-driven weighting, not asserted weighting.** In logit space, a
   modality near p ≈ 0.5 (uncertain) has logit ≈ 0 and contributes almost
   nothing to the pooled result; a confident modality (p near 0 or 1) pulls
   the fused logit strongly toward it. So a stronger, more confident signal
   naturally dominates a weaker one — the "imaging should count more"
   intuition emerges from the evidence itself rather than being hard-coded as
   a constant multiplier.

## The one remaining knob: reliability weights `r_i`

`r_i` defaults to **1.0 for every modality** (equal, uninformative prior). It
is the single calibration parameter in the model, and it is not tuned in this
implementation — there is no labeled multi-modal validation set to fit it
against. Raising `r_i` for a given modality is only justified once validation
data shows that modality is more reliable (e.g. lower Brier score, better
calibration curve); until then, equal weighting is the assumption-minimal
default.

## Known limitations (by design, not fixed here)

- **Calibration.** The pool assumes each input probability is reasonably
  calibrated. MRI softmax outputs from a small fine-tuned classifier can be
  overconfident; if so, that miscalibration propagates into the fused score.
  Temperature scaling per modality is a natural extension and would live in
  the same `reliability`/pre-processing step.
- **Independence assumption.** The pool treats modalities as conditionally
  independent given the disease state. In reality, clinical, imaging, and EEG
  signals are correlated (they all reflect the same underlying pathology), so
  the pool is a simplification, not a joint model. A trained fusion model
  that learns cross-modal correlation is the correct long-term replacement,
  but requires paired labeled data that does not currently exist.
- **Single-modality identity.** With exactly one modality present, `P_fused`
  equals that modality's own probability exactly — this is intentional so the
  Overview tile still means something when a patient has only been through
  one assessment.

## Upgrade path

Once a dataset with paired tabular + imaging (+ EEG) labels exists:
1. Fit `r_i` (or a full logistic/stacking meta-model) against that dataset,
   replacing the equal-weight default.
2. Add per-modality calibration (e.g. Platt scaling / temperature scaling)
   before pooling.
3. Consider replacing the opinion pool with a trained fusion classifier if it
   demonstrably outperforms the pool on held-out data — at that point the pool
   serves as the baseline to beat, not the final design.
