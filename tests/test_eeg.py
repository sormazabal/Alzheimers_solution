"""One runnable check: EEG feature extraction and the train -> predict roundtrip
work on a tiny synthetic dataset. Skips cleanly if mne/neurokit2 aren't installed.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

mne = pytest.importorskip("mne")
pytest.importorskip("neurokit2")
pytest.importorskip("sklearn")

from alz import eeg as eeg_module
from alz.eeg import FEATURE_NAMES, LABELS, extract_features, predict_eeg, predict_eeg_probs, train_eeg

SFREQ = 200.0
DURATION = 4.0


def _make_raw(dominant_hz: float, n_channels: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, DURATION, 1 / SFREQ)
    tone = np.sin(2 * np.pi * dominant_hz * t)
    data = np.tile(tone, (n_channels, 1)) + 0.05 * rng.standard_normal((n_channels, len(t)))
    info = mne.create_info([f"Ch{i}" for i in range(n_channels)], SFREQ, "eeg")
    return mne.io.RawArray(data, info, verbose=False)


def _make_synthetic_dataset(root, n_per_class=8):
    """Writes empty placeholder .set files (just to exist on disk for build_dataset's
    os.path.exists check) + participants.tsv. The actual signal content is supplied by
    monkeypatching load_recording -- eeglabio isn't a real dependency, so tests don't
    need it just to fabricate EEGLAB files.
    """
    import pandas as pd

    rows = []
    os.makedirs(root, exist_ok=True)
    signals = {}
    for i in range(n_per_class):
        for group, hz, seed in [("A", 3.0, i), ("C", 10.0, 100 + i)]:
            sub_id = f"sub-{group}{i:03d}"
            eeg_dir = os.path.join(root, sub_id, "eeg")
            os.makedirs(eeg_dir, exist_ok=True)
            set_path = os.path.join(eeg_dir, f"{sub_id}_task-eyesclosed_eeg.set")
            open(set_path, "w").close()
            signals[set_path] = _make_raw(hz, seed=seed)
            rows.append({"participant_id": sub_id, "Group": group})
    participants_tsv = os.path.join(root, "participants.tsv")
    pd.DataFrame(rows).to_csv(participants_tsv, sep="\t", index=False)
    return participants_tsv, signals


def test_extract_features_alpha_dominant():
    raw = _make_raw(dominant_hz=10.0)
    features = extract_features(raw)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))
    assert features[FEATURE_NAMES.index("alpha")] == max(features[:5])


def test_extract_features_delta_dominant():
    raw = _make_raw(dominant_hz=2.0)
    features = extract_features(raw)
    assert features[FEATURE_NAMES.index("delta")] == max(features[:5])


def test_train_predict_roundtrip(tmp_path, monkeypatch):
    data_dir = str(tmp_path / "derivatives")
    model_path = str(tmp_path / "eeg_model.joblib")
    participants_tsv, signals = _make_synthetic_dataset(data_dir)
    monkeypatch.setattr(eeg_module, "load_recording", lambda path: signals[str(path)])

    accuracy = train_eeg(data_dir, out_path=model_path, participants_tsv=participants_tsv)
    assert 0.0 <= accuracy <= 1.0

    sample = os.path.join(data_dir, "sub-A000", "eeg", "sub-A000_task-eyesclosed_eeg.set")
    result = predict_eeg(sample, model_path=model_path)
    assert result["label"] in LABELS.values()
    assert 0.0 <= result["score"] <= 1.0

    probs_result = predict_eeg_probs(sample, model_path=model_path)
    assert set(probs_result["probs"]) == set(LABELS.values())
    assert probs_result["label"] == max(probs_result["probs"], key=probs_result["probs"].get)
    assert probs_result["score"] == pytest.approx(probs_result["probs"]["Alzheimer's pattern"])
    assert sum(probs_result["probs"].values()) == pytest.approx(1.0, abs=1e-5)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_extract_features_alpha_dominant()
    test_extract_features_delta_dominant()
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
        test_train_predict_roundtrip(Path(tmp), mp)
    print("ok")
