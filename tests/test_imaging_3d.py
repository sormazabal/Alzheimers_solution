"""One runnable check: predict_mri_probs_3d / gradcam_mri_3d on a synthetic NIfTI volume.

Skips cleanly if the optional imaging deps (torch/torchvision/pillow/nibabel) aren't installed.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")
nib = pytest.importorskip("nibabel")

from alz.imaging import _central_axial_slices, gradcam_mri_3d, gradcam_volume_3d, predict_mri_probs_3d, train_mri
from test_imaging import FOLDERS, _make_synthetic_dataset

CLASSES = ["Non Demented", "Demented"]


def _make_synthetic_volume(path, shape=(64, 64, 40)):
    vol = np.random.default_rng(0).integers(0, 255, size=shape).astype(np.float64)
    nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), path)
    return vol


def test_central_axial_slices_clamps_to_depth():
    vol = np.zeros((10, 10, 5))
    slices = _central_axial_slices(vol, n_slices=16)
    assert len(slices) == 5
    assert all(s.shape == (10, 10) and s.dtype == np.uint8 for s in slices)

    slices = _central_axial_slices(np.zeros((10, 10, 40)), n_slices=16)
    assert len(slices) == 16


def test_predict_mri_probs_3d(tmp_path):
    data_dir = str(tmp_path / "data")
    model_path = str(tmp_path / "mri_model.pt")
    nifti_path = str(tmp_path / "scan.nii.gz")

    _make_synthetic_dataset(data_dir)
    train_mri(data_dir, out_path=model_path, epochs=1)
    _make_synthetic_volume(nifti_path)

    result = predict_mri_probs_3d(nifti_path, model_path=model_path)

    assert set(result["probs"]) == set(CLASSES)
    assert result["label"] == max(result["probs"], key=result["probs"].get)
    assert result["score"] == pytest.approx(result["probs"][result["label"]])
    assert sum(result["probs"].values()) == pytest.approx(1.0, abs=1e-5)
    assert result["slice_array"].shape == (64, 64)


def test_gradcam_mri_3d(tmp_path):
    data_dir = str(tmp_path / "data")
    model_path = str(tmp_path / "mri_model.pt")
    nifti_path = str(tmp_path / "scan.nii.gz")

    _make_synthetic_dataset(data_dir)
    train_mri(data_dir, out_path=model_path, epochs=1)
    _make_synthetic_volume(nifti_path)

    expected = predict_mri_probs_3d(nifti_path, model_path=model_path)
    cam = gradcam_mri_3d(nifti_path, model_path=model_path)

    # cam["label"]/["score"] reflect Grad-CAM's own argmax on the single representative
    # slice, so they need not match the mean-pooled label/score in `expected`.
    assert cam["overlay"].size == (224, 224)
    assert cam["label"] in CLASSES
    assert cam["slice_index"] == expected["slice_index"]


def test_gradcam_volume_3d(tmp_path):
    data_dir = str(tmp_path / "data")
    model_path = str(tmp_path / "mri_model.pt")
    nifti_path = str(tmp_path / "scan.nii.gz")

    _make_synthetic_dataset(data_dir)
    train_mri(data_dir, out_path=model_path, epochs=1)
    _make_synthetic_volume(nifti_path, shape=(64, 64, 40))

    result = gradcam_volume_3d(nifti_path, model_path=model_path, max_dim=16)

    assert result["label"] in CLASSES
    # cam_volume must line up voxel-for-voxel with mri_volume_figure's downsampled grid:
    # step = ceil(max(64,64,40)/16) = 4 -> (16, 16, 10)
    assert result["cam_volume"].shape == (16, 16, 10)
    assert result["cam_volume"].min() >= 0.0 and result["cam_volume"].max() <= 1.0 + 1e-6


if __name__ == "__main__":
    import pathlib
    import tempfile

    test_central_axial_slices_clamps_to_depth()
    with tempfile.TemporaryDirectory() as tmp:
        test_predict_mri_probs_3d(pathlib.Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_gradcam_mri_3d(pathlib.Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_gradcam_volume_3d(pathlib.Path(tmp))
    print("ok")
