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


def _load_splits(data_dir: str, limit: int | None, seed: int = 42):
    """ImageFolder dataset -> 70/15/15 train/val/test split, reused by train_mri and evaluate_mri
    so a given (data_dir, limit) always yields the same test set a model was trained against."""
    import torch
    from torch.utils.data import Subset, random_split
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(data_dir, transform=_transform())
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    classes = dataset.dataset.classes if isinstance(dataset, Subset) else dataset.classes

    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_set, val_set, test_set, classes


def _predict_probs(model, loader, device):
    """Eval-mode inference over a loader -> (y_true, y_prob) numpy arrays of softmax probabilities."""
    import numpy as np
    import torch

    y_true, y_prob = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)
            y_true.append(labels.numpy())
            y_prob.append(probs.cpu().numpy())
    if not y_true:
        return np.empty(0), np.empty((0, 0))
    return np.concatenate(y_true), np.concatenate(y_prob)


def _metrics(y_true, y_prob, classes) -> dict:
    """{'accuracy', 'auroc', 'auprc'}; auroc/auprc are None when not computable (e.g. a split
    with only one class present, which happens with tiny/synthetic datasets)."""
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    if len(y_true) == 0:
        return {"accuracy": 0.0, "auroc": None, "auprc": None}

    accuracy = accuracy_score(y_true, y_prob.argmax(1))
    try:
        if len(classes) == 2:
            score = y_prob[:, 1]
            auroc = roc_auc_score(y_true, score)
            auprc = average_precision_score(y_true, score)
        else:
            from sklearn.preprocessing import label_binarize

            auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            y_bin = label_binarize(y_true, classes=range(len(classes)))
            auprc = average_precision_score(y_bin, y_prob, average="macro")
    except ValueError:
        auroc = auprc = None

    return {"accuracy": accuracy, "auroc": auroc, "auprc": auprc}


def _fmt_metric(value) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def train_mri(
    data_dir: str,
    out_path: str = DEFAULT_MODEL_PATH,
    epochs: int = 3,
    limit: int | None = None,
    device: str | None = None,
) -> float:
    """Train on an ImageFolder-structured directory of class folders. Returns held-out test accuracy."""
    import os

    import torch
    from torch.utils.data import DataLoader

    device = _device(device)
    train_set, val_set, test_set, classes = _load_splits(data_dir, limit)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    # Inverse-frequency class weights for the imbalanced OASIS severity classes.
    # ponytail: inverse-freq class weights; add sampler/augmentation if minority recall stalls.
    full_dataset = train_set.dataset
    if hasattr(full_dataset, "dataset"):
        targets = [full_dataset.dataset.targets[i] for i in full_dataset.indices]
    else:
        targets = full_dataset.targets
    counts = torch.bincount(torch.tensor(targets), minlength=len(classes)).float()
    weights = (1.0 / counts.clamp(min=1)).to(device)

    model = build_mri_model(len(classes)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches if n_batches else 0.0

        model.eval()
        train_metrics = _metrics(*_predict_probs(model, train_loader, device), classes)
        val_metrics = _metrics(*_predict_probs(model, val_loader, device), classes)
        print(
            f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - "
            f"train_acc: {train_metrics['accuracy']:.3f} - "
            f"train_auroc: {_fmt_metric(train_metrics['auroc'])} - "
            f"train_auprc: {_fmt_metric(train_metrics['auprc'])} - "
            f"val_acc: {val_metrics['accuracy']:.3f} - "
            f"val_auroc: {_fmt_metric(val_metrics['auroc'])} - "
            f"val_auprc: {_fmt_metric(val_metrics['auprc'])}"
        )

    model.eval()
    test_metrics = _metrics(*_predict_probs(model, test_loader, device), classes)
    print(
        f"Test - acc: {test_metrics['accuracy']:.3f} - "
        f"auroc: {_fmt_metric(test_metrics['auroc'])} - "
        f"auprc: {_fmt_metric(test_metrics['auprc'])}"
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes}, out_path)
    return test_metrics["accuracy"]


def evaluate_mri(
    data_dir: str,
    model_path: str = DEFAULT_MODEL_PATH,
    device: str | None = None,
    limit: int | None = None,
    plot_dir: str | None = None,
) -> dict:
    """Report accuracy/AUROC/AUPRC on the train/val/test splits of a trained model, and
    optionally save ROC/PR curve plots (from the test split) to plot_dir."""
    from torch.utils.data import DataLoader

    device = _device(device)
    train_set, val_set, test_set, classes = _load_splits(data_dir, limit)
    model, _ = _load_mri_model(model_path, device)

    splits = {"train": train_set, "val": val_set, "test": test_set}
    results = {}
    predictions = {}
    for name, split in splits.items():
        loader = DataLoader(split, batch_size=32)
        y_true, y_prob = _predict_probs(model, loader, device)
        predictions[name] = (y_true, y_prob)
        results[name] = _metrics(y_true, y_prob, classes)

    if plot_dir is not None:
        _plot_curves(*predictions["test"], classes, plot_dir)

    return results


def _plot_curves(y_true, y_prob, classes, plot_dir: str) -> None:
    import os

    import numpy as np
    from sklearn.metrics import precision_recall_curve, roc_curve

    os.makedirs(plot_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(y_true) == 0:
        return

    class_indices = range(1) if len(classes) == 2 else range(len(classes))

    fig, ax = plt.subplots()
    for i in class_indices:
        y_bin = (np.asarray(y_true) == 1).astype(int) if len(classes) == 2 else (np.asarray(y_true) == i).astype(int)
        y_score = y_prob[:, 1] if len(classes) == 2 else y_prob[:, i]
        fpr, tpr, _ = roc_curve(y_bin, y_score)
        label = classes[1] if len(classes) == 2 else classes[i]
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend()
    fig.savefig(os.path.join(plot_dir, "roc_curve.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    for i in class_indices:
        y_bin = (np.asarray(y_true) == 1).astype(int) if len(classes) == 2 else (np.asarray(y_true) == i).astype(int)
        y_score = y_prob[:, 1] if len(classes) == 2 else y_prob[:, i]
        precision, recall, _ = precision_recall_curve(y_bin, y_score)
        label = classes[1] if len(classes) == 2 else classes[i]
        ax.plot(recall, precision, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test set)")
    ax.legend()
    fig.savefig(os.path.join(plot_dir, "pr_curve.png"))
    plt.close(fig)


@lru_cache(maxsize=None)  # ponytail: caches per model path; drop if paths change at runtime
def _load_mri_model(model_path: str, device: str):
    import torch

    checkpoint = torch.load(model_path, map_location=device)
    model = build_mri_model(len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["classes"]


def predict_mri_probs(path, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Full severity distribution: {'probs': {class_label: float, ...}, 'label': str, 'score': float}.

    'path' may be a filesystem path or any file-like object PIL.Image.open accepts
    (e.g. a Streamlit UploadedFile).
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
    return {
        "probs": {c: float(p) for c, p in zip(classes, probs.tolist())},
        "label": classes[label_idx],
        "score": float(probs[label_idx]),
    }


def predict_mri(path, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Same return shape as model.predict(): {'score': float, 'label': str}.

    'label' is the predicted severity class (Non Demented / Very mild / Mild / Moderate
    Dementia); 'score' is the model's confidence in that class.
    """
    result = predict_mri_probs(path, model_path)
    return {"score": result["score"], "label": result["label"]}
