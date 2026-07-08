"""Phase-3: EEG-based confirmation via band-power features + logistic regression.

ds004504 (AHEPA resting-state eyes-closed EEG) has only 65 usable AD/CN subjects --
far too few to adapt a deep time-series foundation model honestly. Relative EEG
band power (the textbook "AD slowing": more delta/theta, less alpha/beta) is a
well-established, explainable signal that fits a small-n logistic regression, so
this follows src/alz/model.py's pipeline shape rather than imaging.py's torch one.

mne/neurokit2 imports are lazy (inside functions) so the tabular app path stays
importable without the EEG extras installed (see requirements-eeg.txt).
"""
import os
from functools import lru_cache

import numpy as np

DEFAULT_MODEL_PATH = "models/eeg_model.joblib"
LABELS = {0: "Healthy", 1: "Alzheimer's pattern"}

BANDS = [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 25), ("gamma", 25, 45)]
FEATURE_NAMES = [name for name, _, _ in BANDS] + ["theta_alpha_ratio", "slowing_ratio"]


def load_recording(path_or_file):
    """'path_or_file': a filesystem path to a .set file, or a file-like object
    (e.g. Streamlit UploadedFile) holding .set bytes. Returns an mne Raw object.

    EEGLAB .set files are read by path, not by buffer, so file-like input is
    spooled to a tempfile first.
    """
    import mne

    if isinstance(path_or_file, (str, os.PathLike)):
        return mne.io.read_raw_eeglab(path_or_file, preload=True, verbose=False)

    import tempfile

    data = path_or_file.read() if hasattr(path_or_file, "read") else path_or_file
    with tempfile.NamedTemporaryFile(suffix=".set", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return mne.io.read_raw_eeglab(tmp_path, preload=True, verbose=False)
    finally:
        os.unlink(tmp_path)


def extract_features(raw) -> np.ndarray:
    """Relative band power (delta..gamma, averaged across channels) + two slowing
    ratios that summarize the classic AD EEG pattern (more low-frequency, less
    high-frequency power). Returns a (7,) float array, ordered per FEATURE_NAMES.

    ponytail: channel-averaged bands; go per-region (frontal/temporal/etc.) only
    if accuracy stalls -- 65 subjects can't support many more features.
    """
    import neurokit2 as nk

    sfreq = raw.info["sfreq"]
    data = raw.get_data()  # (n_channels, n_samples)

    band_freqs = [(lo, hi) for _, lo, hi in BANDS]
    powers = []
    for ch in data:
        p = nk.signal_power(ch, frequency_band=band_freqs, sampling_rate=sfreq, method="welch")
        powers.append(p.iloc[0].to_numpy(dtype=float))
    band_power = np.mean(powers, axis=0)  # (5,) delta..gamma
    total = band_power.sum() + 1e-12
    relative = band_power / total

    delta, theta, alpha, beta, gamma = relative
    theta_alpha_ratio = theta / (alpha + 1e-12)
    slowing_ratio = (delta + theta) / (alpha + beta + 1e-12)

    return np.concatenate([relative, [theta_alpha_ratio, slowing_ratio]])


def build_dataset(data_dir: str, participants_tsv: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """data_dir: path to a BIDS 'derivatives' (or raw) directory of sub-XXX/eeg/*.set files.

    Keeps only Alzheimer's ('A') and healthy control ('C') subjects (drops FTD 'F')
    per participants.tsv's Group column. Returns (X, y, subject_ids); y=1 for AD.
    """
    import pandas as pd

    if participants_tsv is None:
        participants_tsv = os.path.join(os.path.dirname(data_dir.rstrip("/\\")), "participants.tsv")
    participants = pd.read_csv(participants_tsv, sep="\t").set_index("participant_id")

    X, y, subject_ids = [], [], []
    for sub_id in sorted(os.listdir(data_dir)):
        if sub_id not in participants.index:
            continue
        group = participants.loc[sub_id, "Group"]
        if group not in ("A", "C"):
            continue
        set_path = os.path.join(data_dir, sub_id, "eeg", f"{sub_id}_task-eyesclosed_eeg.set")
        if not os.path.exists(set_path):
            continue
        raw = load_recording(set_path)
        X.append(extract_features(raw))
        y.append(1 if group == "A" else 0)
        subject_ids.append(sub_id)

    return np.array(X), np.array(y), subject_ids


def build_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train_eeg(data_dir: str, out_path: str = DEFAULT_MODEL_PATH, participants_tsv: str | None = None) -> dict:
    """Trains on all AD/CN subjects in data_dir. Metrics come from pooled stratified
    5-fold out-of-fold predictions (a single 75/25 split is too noisy at n=65), then
    refits on all data and saves the pipeline with joblib.
    """
    import joblib
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from alz.metrics import binary_metrics

    X, y, _ = build_dataset(data_dir, participants_tsv)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(build_pipeline(), X, y, cv=cv, method="predict_proba")
    metrics = binary_metrics(y, oof_proba[:, 1])

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(pipeline, out_path)
    return metrics


@lru_cache(maxsize=None)  # ponytail: caches per model path; drop if paths change at runtime
def _load_eeg_model(model_path: str):
    import joblib

    pipeline = joblib.load(model_path)
    clf = pipeline.named_steps.get("clf")
    if clf is not None and not hasattr(clf, "multi_class"):
        # ponytail: pipeline pickled under an older sklearn missing this attr;
        # patch instead of retraining (no real EEG data available locally)
        clf.multi_class = "auto"
    return pipeline


def predict_eeg_probs(recording, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """'recording' is anything load_recording accepts, OR an already-loaded mne Raw.

    Returns {'probs': {'Healthy': float, "Alzheimer's pattern": float}, 'label': str,
    'score': float}. 'score' is P("Alzheimer's pattern") -- oriented high=worse, same
    polarity fusion.py expects.
    """
    pipeline = _load_eeg_model(model_path)
    raw = recording if hasattr(recording, "get_data") else load_recording(recording)
    features = extract_features(raw).reshape(1, -1)
    proba = pipeline.predict_proba(features)[0]
    label_idx = int(proba.argmax())
    return {
        "probs": {LABELS[i]: float(p) for i, p in enumerate(proba)},
        "label": LABELS[label_idx],
        "score": float(proba[1]),
    }


def predict_eeg(recording, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Same shape as model.predict(): {'score': float, 'label': str}."""
    result = predict_eeg_probs(recording, model_path)
    return {"score": result["score"], "label": result["label"]}


def demo():  # ponytail-required self-check for the only nontrivial pure logic here
    import mne

    sfreq = 500.0
    t = np.arange(0, 4, 1 / sfreq)
    alpha_tone = np.sin(2 * np.pi * 10 * t)  # pure 10 Hz -> should dominate the alpha band
    data = np.tile(alpha_tone, (2, 1))
    info = mne.create_info(["Ch1", "Ch2"], sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    features = extract_features(raw)
    assert features.shape == (7,)
    assert np.all(np.isfinite(features))
    alpha_idx = FEATURE_NAMES.index("alpha")
    assert features[alpha_idx] == max(features[:5]), "alpha band should dominate a pure 10 Hz tone"


demo()
