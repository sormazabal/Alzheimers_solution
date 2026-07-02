"""Phase-2: MRI-based confirmation via 2D transfer learning.

The Kaggle imagesoasis mirror (see data/download_oasis.py) ships 2D JPEG brain slices
in per-class folders, not 3D NIfTI volumes -- so this follows a torchvision ResNet18
transfer-learning recipe (frozen backbone, retrained head) rather than the 3D
MONAI/TorchIO/MedicalNet pipeline in the vendor notebook, which targets volumetric data.
"""
from functools import lru_cache

DEFAULT_MODEL_PATH = "models/mri_model.pt"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _device(device: str | None = None) -> str:
    import torch

    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def build_mri_model(num_classes: int):
    """ResNet18 with an ImageNet-pretrained, frozen backbone and a fresh trainable head.

    ponytail: frozen-backbone head-only fine-tune; unfreeze layers if accuracy stalls.
    """
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_mri(
    data_dir: str,
    out_path: str = DEFAULT_MODEL_PATH,
    epochs: int = 3,
    limit: int | None = None,
    device: str | None = None,
) -> float:
    """Train on an ImageFolder-structured directory of class folders. Returns held-out accuracy."""
    import os

    import torch
    from torch.utils.data import DataLoader, Subset, random_split
    from torchvision.datasets import ImageFolder

    device = _device(device)
    dataset = ImageFolder(data_dir, transform=_transform())
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    classes = dataset.dataset.classes if isinstance(dataset, Subset) else dataset.classes

    train_size = int(0.75 * len(dataset))
    train_set, test_set = random_split(
        dataset, [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    # Inverse-frequency class weights for the imbalanced OASIS severity classes.
    # ponytail: inverse-freq class weights; add sampler/augmentation if minority recall stalls.
    if isinstance(dataset, Subset):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = dataset.targets
    counts = torch.bincount(torch.tensor(targets), minlength=len(classes)).float()
    weights = (1.0 / counts.clamp(min=1)).to(device)

    model = build_mri_model(len(classes)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
    accuracy = correct / len(test_set) if len(test_set) else 0.0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes}, out_path)
    return accuracy


@lru_cache(maxsize=None)  # ponytail: caches per model path; drop if paths change at runtime
def _load_mri_model(model_path: str, device: str):
    import torch

    checkpoint = torch.load(model_path, map_location=device)
    model = build_mri_model(len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["classes"]


def predict_mri(path: str, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Same return shape as model.predict(): {'score': float, 'label': str}.

    'label' is the predicted severity class (Non Demented / Very mild / Mild / Moderate
    Dementia); 'score' is the model's confidence in that class.
    """
    import torch
    from PIL import Image

    device = _device()
    model, classes = _load_mri_model(model_path, device)

    image = Image.open(path).convert("RGB")
    tensor = _transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    label_idx = int(probs.argmax())
    return {"score": float(probs[label_idx]), "label": classes[label_idx]}
