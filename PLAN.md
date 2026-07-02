# Plan: Implement TODO.md (ponytail)

## Context

`TODO.md` lists follow-ups from the README's stated scope. Exploration shows most
items are either **already done**, **blocked**, or **stale premises** — only one is
a real code task. This plan does them in priority order.

Key findings that shrink the work:
- `requirements-imaging.txt` **already exists** (TODO item is just a check-off).
- `.gitignore` is **already committed and clean** — it ignores `data/imagesoasis` and
  `models/*.joblib` only; it does *not* exclude `download_oasis.py`. The only untracked
  file is `TODO.md`. The TODO's "pending .gitignore change" is stale.
- `data/download_oasis.py` is **already tracked** (not untracked as TODO claims). It
  fetches the *imaging* Kaggle dataset (`ninadaithal/imagesoasis`) — a phase-2 helper,
  not tabular ingestion.
- MRI/CNN training is **blocked** (GPU + data-use agreement) — not a code task.

So the ordered work is: (1) FastAPI wrapper — the one real feature; (2) housekeeping/docs
for `download_oasis.py`; (3) close out the check-off / blocked / stale items in text.

## Priority 1 — FastAPI wrapper around `alz.predict()`

The README budgets "~15 lines" and names `alz.predict()` as the seam. `load_model` is
already exported from [src/alz/__init__.py](src/alz/__init__.py).

**New file: `app/api.py`** (~15 lines) mirroring the Streamlit record contract
([app/streamlit_app.py:34-39](app/streamlit_app.py)):
- `sys.path.insert` for `src` (same pattern as streamlit_app.py), `from alz import predict`.
- One `POST /predict` endpoint taking the record dict, returning `predict(record)`.
- A `GET /health` one-liner.
- Use a Pydantic model matching the 9 fields (`Age, EDUC, SES, MMSE, nWBV, ASF, Visit,
  MR Delay, M/F`) for input validation at the trust boundary — do **not** skip this.

**One-line perf fix:** `predict()` calls `load_model()` every invocation, which reloads
the joblib per request. Add `@lru_cache(maxsize=1)` (or `maxsize=None`) to `load_model`
in [src/alz/model.py](src/alz/model.py) so the server loads the model once.
`// ponytail: lru_cache, drop it if the model path ever needs to vary at runtime.`

**Deps:** add `fastapi` and `uvicorn[standard]` to
[requirements.txt](requirements.txt).

**Check:** a `tests/test_api.py` using FastAPI `TestClient` — one request through
`/predict` asserting `label in {"Normal","At-risk"}`, `0<=score<=1`, 3 drivers. This also
fills the existing gap: no current test exercises the `alz.predict()` dict seam end-to-end.
Requires a trained `models/model.joblib` (run `python train.py` first).

## Priority 2 — `download_oasis.py` + data docs (housekeeping)

Default (lazy, correct): **keep it as the phase-2 imaging helper it already is**, don't
fold into `train.py` (train.py consumes tabular CSV, not images — folding is wrong).

- Fix the misleading tail of [data/download_oasis.py](data/download_oasis.py): it ignores
  the returned `path` and prints a literal `"data"`. Print the real `path` and add a
  one-line docstring saying it fetches the **phase-2 imaging** dataset.
- Add `kagglehub` to [requirements-imaging.txt](requirements-imaging.txt) (it's imported
  but listed nowhere).
- Add a short note to [data/README.md](data/README.md): tabular workflow uses the bundled
  synthetic `oasis_longitudinal.csv` (swap for real OASIS-2 CSV, no code change);
  `download_oasis.py` is a **separate phase-2 imaging** fetch, not part of the tabular demo.

## Priority 3 — close out the trivial / blocked / stale items (mostly text)

- **requirements-imaging.txt exists** → confirmed; check it off (after Priority 2 adds
  `kagglehub`).
- **MRI/CNN training** → blocked on GPU + data-use agreement; leave
  [src/alz/imaging.py](src/alz/imaging.py) as the `NotImplementedError` stub. No code.
- **.gitignore / untracked download_oasis.py** → stale; nothing to change. Note in TODO
  that both premises are already resolved.
- Update `TODO.md`: check off shipped items, correct the stale notes.
- Deliberately deferred (fusion, auth, Docker, registry, CI) → leave alone.

## Verification (end-to-end)

1. `pip install -r requirements.txt` (now includes fastapi/uvicorn).
2. `python train.py` → writes `models/model.joblib`.
3. `pytest tests/` → smoke test + new `test_api.py` pass.
4. `uvicorn app.api:app --reload` then
   `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"Age":75,"EDUC":12,"SES":3,"MMSE":27,"nWBV":0.75,"ASF":1.2,"Visit":1,"MR Delay":0,"M/F":"F"}'`
   → returns `{score, label, drivers}`. `curl localhost:8000/health` → ok.
5. Cross-check the response matches what the Streamlit app shows for the same record.

## Skipped (ponytail)
- No `requirements-api.txt` split — fastapi/uvicorn go in the main file; add a split only
  if the API deploy target diverges from the demo.
- No Docker/auth/registry/CI — deferred per README, out of scope.
- No imaging implementation — blocked on hardware + data agreement.
