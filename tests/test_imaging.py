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

from alz.imaging import evaluate_mri, predict_mri, predict_mri_probs, train_mri

CLASSES = ["Non Demented", "Mild Dementia"]


def _make_synthetic_dataset(root):
    paths = []
    for cls in CLASSES:
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(10):
            img = Image.fromarray(
                (torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)).numpy()
            )
            path = os.path.join(cls_dir, f"{i}.png")
            img.save(path)
            paths.append(path)
    return paths


def test_train_predict_roundtrip(tmp_path):
    data_dir = str(tmp_path / "data")
    model_path = str(tmp_path / "mri_model.pt")
    paths = _make_synthetic_dataset(data_dir)

    accuracy = train_mri(data_dir, out_path=model_path, epochs=1)
    assert 0.0 <= accuracy <= 1.0

    result = predict_mri(paths[0], model_path=model_path)
    assert result["label"] in CLASSES
    assert 0.0 <= result["score"] <= 1.0

    probs_result = predict_mri_probs(paths[0], model_path=model_path)
    assert set(probs_result["probs"]) == set(CLASSES)
    assert probs_result["label"] == max(probs_result["probs"], key=probs_result["probs"].get)
    assert probs_result["score"] == pytest.approx(probs_result["probs"][probs_result["label"]])
    assert sum(probs_result["probs"].values()) == pytest.approx(1.0, abs=1e-5)


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
        test_train_predict_roundtrip(__import__("pathlib").Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_evaluate_mri(__import__("pathlib").Path(tmp))
    print("ok")
