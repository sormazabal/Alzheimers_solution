"""Integrated multi-modal prognosis score.

Combines clinical, MRI, and EEG risk estimates into one P(dementia) via a
logarithmic opinion pool (weighted average in log-odds space). See
docs/fusion-methodology.md for the justification and citations.
"""
import math

_EPS = 1e-6


def _clamp(p: float) -> float:
    return min(max(p, _EPS), 1 - _EPS)


def _logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _p_dementia(clinical: dict | None, mri: dict | None, eeg: dict | None) -> list[tuple[str, float]]:
    components = []
    if clinical:
        components.append(("Clinical", clinical["score"]))
    if mri:
        components.append(("MRI", 1 - mri["probs"]["Non Demented"]))
    if eeg:
        # eeg["score"] is already P(abnormal EEG finding), same polarity as the others.
        components.append(("EEG", eeg["score"]))
    return components


def integrated_score(
    clinical: dict | None = None,
    mri: dict | None = None,
    eeg: dict | None = None,
    reliability: dict[str, float] | None = None,
) -> dict | None:
    """Logarithmic-opinion-pool fusion of present modalities into P(dementia).

    reliability: optional {modality_name: weight}, default 1.0 each. This is
    the sole calibration knob — see docs/fusion-methodology.md for why it is
    not tuned here (no paired multi-modal validation data exists yet).

    Returns None if no modality is supplied, else
    {"score": float, "components": [{"modality", "p", "weight"}, ...], "note": str}.
    """
    components = _p_dementia(clinical, mri, eeg)
    if not components:
        return None

    reliability = reliability or {}
    weights = [reliability.get(name, 1.0) for name, _ in components]
    total_weight = sum(weights)
    pooled_logit = sum(w * _logit(p) for (_, p), w in zip(components, weights)) / total_weight
    score = _sigmoid(pooled_logit)

    names = [name for name, _ in components]
    note = f"Fused: {' + '.join(names)}" if len(names) > 1 else f"Only {names[0]} assessed"

    return {
        "score": score,
        "components": [
            {"modality": name, "p": p, "weight": w}
            for (name, p), w in zip(components, weights)
        ],
        "note": note,
    }


def combine_mri(result_2d: dict, result_3d: dict) -> dict:
    """Equal-weight log-odds pool of the 2D and 3D MRI result dicts into one
    result-shaped dict {'probs': {'Non Demented', 'Demented'}, 'label', 'score'}.

    Pools P(Demented) (1 - probs['Non Demented']) in log-odds space -- NOT the
    'score' field, whose polarity flips with the predicted label.
    """
    p2d = 1 - result_2d["probs"]["Non Demented"]
    p3d = 1 - result_3d["probs"]["Non Demented"]
    p_dem = _sigmoid((_logit(p2d) + _logit(p3d)) / 2)
    probs = {"Non Demented": 1 - p_dem, "Demented": p_dem}
    label = max(probs, key=probs.get)
    return {"probs": probs, "label": label, "score": probs[label]}


def demo():  # ponytail-required self-check for the only nontrivial pure logic here
    assert integrated_score() is None
    only_clinical = integrated_score(clinical={"score": 0.7})
    assert abs(only_clinical["score"] - 0.7) < 1e-9
    confident_mri_uncertain_clinical = integrated_score(
        clinical={"score": 0.5}, mri={"probs": {"Non Demented": 0.1}}
    )
    assert confident_mri_uncertain_clinical["score"] > 0.7  # MRI dominates the uncertain clinical opinion

    agree = combine_mri(
        {"probs": {"Non Demented": 0.1, "Demented": 0.9}},
        {"probs": {"Non Demented": 0.2, "Demented": 0.8}},
    )
    assert agree["label"] == "Demented"
    assert 0.8 < agree["probs"]["Demented"] < 0.9

    disagree = combine_mri(
        {"probs": {"Non Demented": 0.9, "Demented": 0.1}},
        {"probs": {"Non Demented": 0.1, "Demented": 0.9}},
    )
    assert abs(disagree["probs"]["Demented"] - 0.5) < 1e-9  # symmetric disagreement cancels out


demo()
