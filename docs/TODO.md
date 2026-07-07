# To-Do

Based on README.md's stated scope (built / stubbed / skipped). See [PLAN.md](PLAN.md) for
the implementation plan and rationale.

## Ship the stated MVP seam
- [x] Add the FastAPI wrapper around `alz.predict()` — [app/api.py](app/api.py), see
      [tests/test_api.py](tests/test_api.py)
- [x] `data/download_oasis.py` documented in [data/README.md](data/README.md) as a
      separate phase-2 imaging fetch — it does not replace the tabular sample workflow

## Phase 2 (imaging) — implemented
- [x] Confirmed `requirements-imaging.txt` exists (now `torch`, `torchvision`, `pillow`,
      `kagglehub`)
- [x] Implement MRI severity classifier in
      [src/alz/imaging.py](src/alz/imaging.py) — the "needs GPU + data-use agreement"
      block was stale: the Kaggle `imagesoasis` mirror is 2D JPEG slices, publicly
      downloadable via `kagglehub` (no DUA), and a transfer-learned ResNet18 head trains
      fine on CPU (uses GPU automatically when available). See `train_mri.py`.

## Housekeeping
- [x] `.gitignore` was already committed and clean (only ignores `data/imagesoasis` and
      `models/*.joblib`) — no change needed
- [x] `data/download_oasis.py` was already tracked; kept as the phase-2 imaging helper and
      documented in [data/README.md](data/README.md); fixed it to print the real download
      path instead of a hardcoded string

## Deliberately deferred (per README) — leave alone unless requirements change
- Multi-modal fusion, auth, Docker, model registry, CI

Future:
1. Score integration accross modalities
2. more explainability for the image
3. Explainability for the EEG

## Phase 3 (3D MRI, slice-and-pool) — implemented

- [x] `predict_mri_probs_3d` + `_central_axial_slices` in
      [src/alz/imaging.py](../src/alz/imaging.py) — reuses the existing 2D model, mean-pools
      probs over central axial slices, same `{"probs","label","score"}` contract fusion
      already consumes
- [x] NIfTI upload + dispatch in the MRI tab —
      [app/streamlit_app.py](../app/streamlit_app.py): uploader accepts `.nii`/`.nii.gz`,
      spools to a temp file, branches on extension
- [x] Representative-slice Grad-CAM — `gradcam_mri_3d` in `imaging.py`, reuses 2D
      `gradcam_mri`/`explain_mri` on the most-affected central slice, no rewrite needed
- [x] `nibabel` added to `requirements-imaging.txt`, imported lazily inside the function
- [x] Domain-gap disclosure caption in the MRI tab (`streamlit_app.py:452-455`): "3D volume
      (v1): the existing 2D model is applied to central axial slices and pooled — not a
      validated volumetric classifier."
- [x] Unit self-check — [tests/test_imaging_3d.py](../tests/test_imaging_3d.py) (synthetic
      NIfTI volume, checks probs/label/score contract and slice-count clamping)
- [ ] TODO-1b — same disclosure line is still missing from
      [docs/fusion-methodology.md](fusion-methodology.md) and
      [docs/PRESENTATION.md](PRESENTATION.md); only the in-app caption is done
- [ ] End-to-end manual check with a real OASIS-3 `.nii.gz` — not yet run (no sample volumes
      pulled into `data/oasis3_samples/` yet; `.gitignore` entry is already in place)

Deferred, with trigger and upgrade path (each gets a `ponytail:` comment at its seam):

- TODO-2 — True volumetric Grad-CAM. Montage/overlay across all central slices instead of
  one representative slice. Add when clinicians ask for volumetric explainability. Upgrade
  path: loop `gradcam_mri` over the slice stack, tile the overlays.
- TODO-3 — Native 3D model (CNN / GNN / UNETR). Skipped: needs training and labelled 3D
  volumes. Add only if slice-and-pool underperforms on a real OASIS-3 eval. Path: MONAI 3D
  ResNet backbone + fine-tuned head, or a FreeSurfer-region GNN (aligns with prior GNN work).
- TODO-4 — Skull-stripping / MNI registration. Skipped. Add only if the domain gap
  measurably hurts — first compare pooled-3D vs 2D on the same subjects; if the gap is
  small, this preprocessing is not worth owning.
- TODO-5 — Imaging reliability re-weighting. The fusion reliability knob for the MRI
  modality stays at default 1.0. Skipped: no paired 3D validation data to tune it. Add when
  such data exists (temperature-scaling hook already noted in `docs/fusion-methodology.md`).
