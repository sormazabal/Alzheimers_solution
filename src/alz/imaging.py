"""Phase-2: MRI-based confirmation via 2D transfer learning.

The Kaggle imagesoasis mirror (see data/download_oasis.py) ships 2D JPEG brain slices
in per-class folders, not 3D NIfTI volumes -- so this follows a torchvision ResNet18
transfer-learning recipe (frozen backbone, retrained head) rather than the 3D
MONAI/TorchIO/MedicalNet pipeline in the vendor notebook, which targets volumetric data.

3D NIfTI volumes (e.g. OASIS-3) are supported via predict_mri_probs_3d / gradcam_mri_3d,
which apply this same 2D model to central axial slices and pool the result -- a v1 that
skips training rather than a native volumetric model. See docs for the domain-gap caveat.
"""
from functools import lru_cache

DEFAULT_MODEL_PATH = "models/mri_model.pt"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _device(device: str | None = None):
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_directml

        if torch_directml.is_available():
            # ponytail: torch-directml pins torch==2.4.1/torchvision==0.19.1 (see
            # requirements-imaging.txt) -- it lags mainline releases; re-check the pin
            # if this stops importing after a torch upgrade.
            return torch_directml.device()
    except ImportError:
        pass
    return "cpu"


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


class _BinaryImageSubset:
    """A subset of an ImageFolder's samples, relabeled to binary (0/1) targets.

    Labels come from label_map (original class idx -> 0/1), not from re-reading the
    image, so subject-level label lookups (for class-balance weighting) stay cheap.
    """

    def __init__(self, base_dataset, indices, label_map):
        self.base = base_dataset
        self.indices = indices
        self.labels = [label_map[base_dataset.samples[i][1]] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, _ = self.base[self.indices[i]]
        return image, self.labels[i]


def _subject_of(path: str) -> str:
    """'.../OAS1_0028_MR1_mpr-1_100.jpg' -> 'OAS1_0028' (subject id encoded in the filename)."""
    import os

    name = os.path.basename(path)
    return "_".join(name.split("_")[:2])


def _load_splits(data_dir: str, limit: int | None, seed: int = 42):
    """ImageFolder dataset -> binary (Non Demented vs Demented) subject-grouped 70/15/15
    train/val/test split, reused by train_mri and evaluate_mri so a given (data_dir, limit)
    always yields the same test set a model was trained against.

    Grouped by subject (parsed from the filename) so slices from one patient can't land in
    both train and test -- a per-slice random split leaked patients across splits and
    produced a meaningless ~1.00 accuracy.

    `limit` caps the number of *subjects* (not slices) -- each subject contributes
    hundreds to thousands of slices, so capping by raw slice count could exhaust the pool
    on a single subject's own scans, leaving too few subjects to split.
    """
    from sklearn.model_selection import train_test_split
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(data_dir, transform=_transform())
    classes = ["Non Demented", "Demented"]
    non_demented_idx = dataset.class_to_idx["Non Demented"]
    label_map = {i: (0 if i == non_demented_idx else 1) for i in range(len(dataset.classes))}

    subjects: dict[str, list[int]] = {}
    for i, (path, _) in enumerate(dataset.samples):
        subjects.setdefault(_subject_of(path), []).append(i)

    subject_ids = list(subjects.keys())
    subject_labels = [label_map[dataset.samples[subjects[s][0]][1]] for s in subject_ids]

    if limit is not None and limit < len(subject_ids):
        # Plain positional truncation would keep only the alphabetically-first
        # class folder's subjects (ImageFolder orders samples by class name),
        # silently producing a single-class dataset -- sample stratified instead.
        # Proportional sampling alone isn't enough: this dataset's classes are
        # heavily imbalanced (~23 Demented vs ~1047 Non Demented subjects), so a
        # small limit can pick so few minority subjects that the later nested
        # 70/15/15 split fails ("least populated class has only 1 member").
        # Reserve a floor per class first so both nested splits stay >=2/class.
        import random

        rng = random.Random(seed)
        by_class: dict[int, list[str]] = {}
        for s, l in zip(subject_ids, subject_labels):
            by_class.setdefault(l, []).append(s)

        # ponytail: floor of 8/class survives two rounds of stratified splitting
        # (70/30 then 50/50) after rounding; bump if a finer split is added.
        floor = min(8, limit // max(len(by_class), 1))
        chosen = [s for ids in by_class.values() for s in rng.sample(ids, min(len(ids), floor))]

        remaining = [s for s in subject_ids if s not in chosen]
        budget = max(limit - len(chosen), 0)
        if budget and remaining:
            chosen += rng.sample(remaining, min(budget, len(remaining)))

        subject_ids = chosen
        subject_labels = [label_map[dataset.samples[subjects[s][0]][1]] for s in subject_ids]

    train_subs, temp_subs, train_y, temp_y = train_test_split(
        subject_ids, subject_labels, train_size=0.7, random_state=seed, stratify=subject_labels
    )
    val_subs, test_subs, _, _ = train_test_split(
        temp_subs, temp_y, train_size=0.5, random_state=seed, stratify=temp_y
    )

    def _indices_for(subs):
        return [i for s in subs for i in subjects[s]]

    train_set = _BinaryImageSubset(dataset, _indices_for(train_subs), label_map)
    val_set = _BinaryImageSubset(dataset, _indices_for(val_subs), label_map)
    test_set = _BinaryImageSubset(dataset, _indices_for(test_subs), label_map)
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
    """Binary classifier only (Non Demented vs Demented) -- full clinical metric set via
    alz.metrics.binary_metrics, keyed on P(Demented) = y_prob[:, 1]."""
    from alz.metrics import binary_metrics

    if len(y_true) == 0:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "auroc": None, "auprc": None,
                "f1": 0.0, "sensitivity": None, "specificity": None,
                "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}}
    return binary_metrics(y_true, y_prob[:, 1])


def _fmt_metric(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def train_mri(
    data_dir: str,
    out_path: str = DEFAULT_MODEL_PATH,
    epochs: int = 3,
    limit: int | None = None,
    device: str | None = None,
) -> dict:
    """Train on an ImageFolder-structured directory of class folders (binary: Non Demented
    vs Demented, subject-grouped split). Returns the held-out test metric set."""
    import os

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from alz.metrics import save_metrics

    device = _device(device)
    train_set, val_set, test_set, classes = _load_splits(data_dir, limit)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    # Inverse-frequency class weights for the imbalanced Non Demented / Demented split.
    # ponytail: inverse-freq class weights; add sampler/augmentation if minority recall stalls.
    counts = torch.bincount(torch.tensor(train_set.labels), minlength=len(classes)).float()
    weights = (1.0 / counts.clamp(min=1)).to(device)

    model = build_mri_model(len(classes)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")
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
    print("Test metrics (held-out subjects):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes}, out_path)
    save_metrics("mri", test_metrics)
    return test_metrics


def evaluate_mri(
    data_dir: str,
    model_path: str = DEFAULT_MODEL_PATH,
    device: str | None = None,
    limit: int | None = None,
    plot_dir: str | None = None,
) -> dict:
    """Report the full binary metric set on the train/val/test splits of a trained model,
    save results/mri_metrics.json (test split), and optionally save ROC/PR curve plots
    (from the test split) to plot_dir."""
    from torch.utils.data import DataLoader

    from alz.metrics import save_metrics

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

    save_metrics("mri", results["test"])
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

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = build_mri_model(len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["classes"]


def predict_mri_probs(path, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Binary dementia confirmation: {'probs': {'Non Demented': float, 'Demented': float},
    'label': str, 'score': float}.

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

    'label' is 'Non Demented' or 'Demented'; 'score' is the model's confidence in that label.
    """
    result = predict_mri_probs(path, model_path)
    return {"score": result["score"], "label": result["label"]}


def gradcam_mri(path, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Grad-CAM overlay showing which regions of the scan drove the prediction.

    Returns {'overlay': PIL.Image (224x224 RGB), 'label': str, 'score': float,
    'cam': np.ndarray (224x224, normalized 0-1 attention map)}.

    ponytail: plain hook-based Grad-CAM on model.layer4[-1] (the stock ResNet18's last conv
    block) -- swap for captum/grad-cam only if a fancier variant (Grad-CAM++, etc.) is needed.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    import matplotlib
    matplotlib.use("Agg")

    device = _device()
    model, classes = _load_mri_model(model_path, device)

    image = Image.open(path).convert("RGB")
    display_image = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])(image)
    # Backbone params are frozen (requires_grad=False), so autograd won't track activations
    # through them unless the input itself requires grad.
    tensor = _transform()(image).unsqueeze(0).to(device).requires_grad_(True)

    activations, gradients = {}, {}
    target_layer = model.layer4[-1]

    def _forward_hook(_module, _input, output):
        activations["value"] = output

    def _backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_f = target_layer.register_forward_hook(_forward_hook)
    handle_b = target_layer.register_full_backward_hook(_backward_hook)
    try:
        model.zero_grad()
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        label_idx = int(probs.argmax())
        logits[0, label_idx].backward()

        weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        score = float(probs[label_idx].detach())
    finally:
        handle_f.remove()
        handle_b.remove()

    heatmap = (matplotlib.colormaps["jet"](cam)[:, :, :3] * 255).astype(np.uint8)
    base = np.asarray(display_image).astype(np.float32)
    blended = (0.5 * heatmap.astype(np.float32) + 0.5 * base).clip(0, 255).astype(np.uint8)

    return {
        "overlay": Image.fromarray(blended),
        "label": classes[label_idx],
        "score": score,
        "cam": cam,
    }


def _central_axial_slices(vol, n_slices: int):
    """Middle n_slices along the axial axis of a canonical-orientation volume,
    percentile-clipped and rescaled to uint8. Returns a list of 2D arrays.

    ponytail: fixed axis=2 (axial) assumes nib.as_closest_canonical orientation.
    """
    import numpy as np

    depth = vol.shape[2]
    n = min(n_slices, depth)
    lo = max(0, depth // 2 - n // 2)
    selected = vol[:, :, lo:lo + n]

    lo_p, hi_p = np.percentile(selected, [1, 99])
    clipped = np.clip(selected, lo_p, hi_p)
    scaled = ((clipped - lo_p) / (hi_p - lo_p + 1e-6) * 255).astype(np.uint8)
    return [scaled[:, :, i] for i in range(scaled.shape[2])]


def predict_mri_probs_3d(
    path, model_path: str = DEFAULT_MODEL_PATH, n_slices: int = 16, device: str | None = None
) -> dict:
    """3D drop-in for predict_mri_probs: applies the existing 2D model to the central
    axial slices of a NIfTI volume and mean-pools the softmax probabilities.

    ponytail: v1 domain-gap caveat -- the model learned OASIS JPEG slices, not raw
    NIfTI intensities; percentile-clip + rescale narrows but doesn't close the gap.
    Upgrade to a fine-tuned native 3D model if accuracy on real OASIS-3 falls short.

    Returns {'probs': {...}, 'label': str, 'score': float, 'slice_index': int,
    'slice_array': np.ndarray} -- the last two identify the most-confident slice
    for representative-slice Grad-CAM (see gradcam_mri_3d).

    'path' may be a filesystem path (NIfTI needs real file access for gzip seeking,
    so a BytesIO caller should spool to a temp file first).
    """
    import nibabel as nib
    import numpy as np
    import torch
    from PIL import Image

    dev = _device(device)
    model, classes = _load_mri_model(model_path, dev)

    vol = nib.as_closest_canonical(nib.load(path)).get_fdata()
    if vol.ndim == 4:  # ponytail: some ANALYZE exports (e.g. OASIS-1) carry a trailing singleton dim
        vol = vol[..., 0]
    slices = _central_axial_slices(vol, n_slices)

    tf = _transform()
    batch = torch.stack([tf(Image.fromarray(s).convert("RGB")) for s in slices]).to(dev)
    with torch.no_grad():
        per_slice_probs = torch.softmax(model(batch), dim=1)
        probs = per_slice_probs.mean(dim=0)

    demented_idx = classes.index("Demented")
    slice_idx = int(per_slice_probs[:, demented_idx].argmax())

    label_idx = int(probs.argmax())
    return {
        "probs": {c: float(p) for c, p in zip(classes, probs.tolist())},
        "label": classes[label_idx],
        "score": float(probs[label_idx]),
        "slice_index": slice_idx,
        "slice_array": slices[slice_idx],
    }


def gradcam_mri_3d(path, model_path: str = DEFAULT_MODEL_PATH, n_slices: int = 16) -> dict:
    """Representative-slice Grad-CAM for a 3D volume: runs the existing 2D gradcam_mri
    on the single central axial slice with the highest P(Demented), so the same
    hook-based CAM logic applies unchanged. Not a true volumetric CAM (see TODO-2 in
    the 3D integration plan) -- label this in the UI as a representative slice.
    """
    import tempfile

    from PIL import Image

    result = predict_mri_probs_3d(path, model_path, n_slices)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        Image.fromarray(result["slice_array"]).save(tmp.name)
        cam_result = gradcam_mri(tmp.name, model_path)

    cam_result["slice_index"] = result["slice_index"]
    return cam_result
