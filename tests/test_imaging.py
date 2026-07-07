"""One runnable check: the MRI train -> predict roundtrip works on a tiny synthetic dataset.

Skips cleanly if the optional imaging deps (torch/torchvision/pillow) aren't installed.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from PIL import Image

from alz.imaging import _load_splits, evaluate_mri, gradcam_mri, predict_mri, predict_mri_probs, train_mri

FOLDERS = ["Non Demented", "Mild Dementia"]  # on-disk folder names; model classes are binary
CLASSES = ["Non Demented", "Demented"]


def _make_synthetic_dataset(root, subjects_per_class=8, images_per_subject=3):
    """Filenames encode a subject id (sub00_0.png, sub00_1.png, ...) like the real
    OAS1_XXXX_MR1_... slices, so the subject-grouped split has something to group."""
    paths = []
    for cls in FOLDERS:
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for s in range(subjects_per_class):
            for i in range(images_per_subject):
                img = Image.fromarray(
                    (torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)).numpy()
                )
                path = os.path.join(cls_dir, f"sub_{cls[:1]}{s}_{i}.png")
                img.save(path)
                paths.append(path)
    return paths


def test_split_has_no_subject_overlap(tmp_path):
    data_dir = str(tmp_path / "data")
    _make_synthetic_dataset(data_dir)

    train_set, val_set, test_set, classes = _load_splits(data_dir, limit=None)
    assert classes == CLASSES

    from alz.imaging import _subject_of

    def subjects_of(subset):
        return {_subject_of(subset.base.samples[i][0]) for i in subset.indices}

    train_subs, val_subs, test_subs = subjects_of(train_set), subjects_of(val_set), subjects_of(test_set)
    assert not (train_subs & val_subs)
    assert not (train_subs & test_subs)
    assert not (val_subs & test_subs)


def test_limit_keeps_both_classes(tmp_path):
    """ImageFolder orders samples alphabetically by class ("Mild Dementia" before "Non
    Demented"), so a naive positional truncation of `limit` would keep only the first
    class's subjects. Guards against that regression."""
    data_dir = str(tmp_path / "data")
    _make_synthetic_dataset(data_dir, subjects_per_class=12)

    train_set, val_set, test_set, _ = _load_splits(data_dir, limit=12)
    all_labels = set(train_set.labels) | set(val_set.labels) | set(test_set.labels)
    assert all_labels == {0, 1}


def test_train_predict_roundtrip(tmp_path):
    data_dir = str(tmp_path / "data")
    model_path = str(tmp_path / "mri_model.pt")
    paths = _make_synthetic_dataset(data_dir)

    metrics = train_mri(data_dir, out_path=model_path, epochs=1)
    assert 0.0 <= metrics["accuracy"] <= 1.0

    result = predict_mri(paths[0], model_path=model_path)
    assert result["label"] in CLASSES
    assert 0.0 <= result["score"] <= 1.0

    probs_result = predict_mri_probs(paths[0], model_path=model_path)
    assert set(probs_result["probs"]) == set(CLASSES)
    assert probs_result["label"] == max(probs_result["probs"], key=probs_result["probs"].get)
    assert probs_result["score"] == pytest.approx(probs_result["probs"][probs_result["label"]])
    assert sum(probs_result["probs"].values()) == pytest.approx(1.0, abs=1e-5)


def test_gradcam_mri(tmp_path):
    data_dir = str(tmp_path / "data")
    model_path = str(tmp_path / "mri_model.pt")
    paths = _make_synthetic_dataset(data_dir)
    train_mri(data_dir, out_path=model_path, epochs=1)

    expected = predict_mri_probs(paths[0], model_path=model_path)
    cam = gradcam_mri(paths[0], model_path=model_path)

    assert cam["overlay"].size == (224, 224)
    assert cam["overlay"].mode == "RGB"
    assert cam["label"] == expected["label"]
    assert cam["score"] == pytest.approx(expected["score"])


def test_evaluate_mri(tmp_path):
    data_dir = str(tmp_path / "data")
    model_path = str(tmp_path / "mri_model.pt")
    plot_dir = str(tmp_path / "plots")
    _make_synthetic_dataset(data_dir)
    train_mri(data_dir, out_path=model_path, epochs=1)

    results = evaluate_mri(data_dir, model_path=model_path, plot_dir=plot_dir)

    assert set(results) == {"train", "val", "test"}
    for metrics in results.values():
        assert 0.0 <= metrics["accuracy"] <= 1.0
    assert os.path.exists(os.path.join(plot_dir, "roc_curve.png"))
    assert os.path.exists(os.path.join(plot_dir, "pr_curve.png"))


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_split_has_no_subject_overlap(__import__("pathlib").Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_train_predict_roundtrip(__import__("pathlib").Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_gradcam_mri(__import__("pathlib").Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_evaluate_mri(__import__("pathlib").Path(tmp))
    print("ok")
