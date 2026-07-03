"""One runnable check: integrated_score's logarithmic opinion pool behaves as documented."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alz.fusion import integrated_score


def test_no_modalities_returns_none():
    assert integrated_score() is None


def test_single_modality_is_exact_identity():
    result = integrated_score(clinical={"score": 0.72})
    assert abs(result["score"] - 0.72) < 1e-9
    assert result["note"] == "Only Clinical assessed"


def test_confident_mri_dominates_uncertain_clinical():
    result = integrated_score(
        clinical={"score": 0.5},
        mri={"probs": {"Non Demented": 0.05}},
    )
    assert result["score"] > 0.8
    assert result["note"] == "Fused: Clinical + MRI"


def test_concordant_high_inputs_band_as_high():
    result = integrated_score(
        clinical={"score": 0.9},
        mri={"probs": {"Non Demented": 0.05}},
        eeg={"score": 0.85, "label": "Generalized slowing suspected", "level": "high"},
    )
    assert 0.0 <= result["score"] <= 1.0
    assert result["score"] > 0.8
    assert result["note"] == "Fused: Clinical + MRI + EEG"


def test_reliability_weight_shifts_toward_upweighted_modality():
    equal = integrated_score(clinical={"score": 0.5}, mri={"probs": {"Non Demented": 0.05}})
    mri_upweighted = integrated_score(
        clinical={"score": 0.5},
        mri={"probs": {"Non Demented": 0.05}},
        reliability={"MRI": 5.0},
    )
    assert mri_upweighted["score"] > equal["score"]
