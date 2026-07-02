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
