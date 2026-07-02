"""Phase-2 stub: MRI-based confirmation. Not implemented in this MVP.

When built, this should follow the CNN template in
vendor/alzheimers-disease-prediction/Oasis2_Image_Data.ipynb, but swap the hand-rolled 3D CNN for
MONAI's monai.networks.nets.DenseNet121 with TorchIO preprocessing and pretrained MedicalNet
weights (see requirements-imaging.txt) rather than training a network from scratch.
"""


def predict_mri(path: str) -> dict:
    """Same return shape as model.predict(): {'score': float, 'label': str}."""
    raise NotImplementedError(
        "Phase 2: wire up MONAI DenseNet121 + TorchIO preprocessing here. "
        "See vendor/alzheimers-disease-prediction/Oasis2_Image_Data.ipynb for the reference pipeline."
    )
