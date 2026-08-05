"""Patient database: links each OASIS-2 clinical subject to an MRI (OASIS-1) and an
EEG (ds004504) subject, and caches all-modality inference results in SQLite so a
cohort of patients only needs to be scored once.

No subject is shared across the three source datasets (see fusion-methodology.md),
so the link is synthesized: within each diagnosis class (demented vs not), a
clinical subject is paired with the nearest-age, same-sex MRI/EEG subject not yet
used, falling back to reuse once every candidate in a class is taken. This keeps
each patient's bundled modalities clinically consistent (a "Demented" patient never
gets a healthy-control EEG) even though it isn't a real co-registered patient.
"""
import glob
import json
import os
import re
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DB_PATH = os.path.join(_ROOT, "data", "patients.db")
OASIS2_CSV = os.path.join(_ROOT, "data", "oasis_longitudinal.csv")
OASIS1_RAW_DIR = os.path.join(_ROOT, "data", "OASIS1_raw")
IMAGESOASIS_DIR = os.path.join(_ROOT, "data", "imagesoasis", "versions", "1", "Data")
EEG_DERIV_DIR = os.path.join(_ROOT, "data", "ds004504", "derivatives")
EEG_PARTICIPANTS_TSV = os.path.join(_ROOT, "data", "ds004504", "participants.tsv")

MODALITIES = ["clinical", "mri_2d", "mri_3d", "eeg"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    mri_id TEXT,
    group_label TEXT,
    ehr_json TEXT
);
CREATE TABLE IF NOT EXISTS recordings (
    patient_id TEXT,
    modality TEXT,
    source_id TEXT,
    path TEXT,
    PRIMARY KEY (patient_id, modality, source_id)
);
CREATE TABLE IF NOT EXISTS results (
    patient_id TEXT,
    modality TEXT,
    input_key TEXT,
    result_json TEXT,
    created_at TEXT,
    PRIMARY KEY (patient_id, modality)
);
"""


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Connection + schema
# ---------------------------------------------------------------------------
def connect(path: str = DB_PATH) -> sqlite3.Connection:
    """Opens (creating if needed) the patient DB and auto-builds it on first use.

    check_same_thread=False: Streamlit reruns a page's script on whichever worker
    thread picks it up, but st.cache_resource (see app/streamlit_app.py) hands the
    same connection to every rerun -- sqlite3 connections are thread-affine by
    default, so this is required, not optional, for a cached connection.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    if conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 0:
        build(conn)
    return conn


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------
def _clinical_subjects(csv_path: str) -> list[dict]:
    """One row per OASIS-2 subject: latest visit's record, for alz.predict."""
    df = pd.read_csv(csv_path)
    ses_mean, mmse_mean = df["SES"].mean(), df["MMSE"].mean()
    subjects = []
    for subject_id, rows in df.groupby("Subject ID"):
        row = rows.loc[rows["Visit"].idxmax()]
        group = row["Group"]
        ehr = {
            "Visit": int(row["Visit"]),
            "MR Delay": int(row["MR Delay"]),
            "Age": int(row["Age"]),
            "EDUC": int(row["EDUC"]),
            "SES": float(row["SES"]) if pd.notna(row["SES"]) else float(ses_mean),
            "MMSE": float(row["MMSE"]) if pd.notna(row["MMSE"]) else float(mmse_mean),
            "nWBV": float(row["nWBV"]),
            "ASF": float(row["ASF"]),
            "M/F": row["M/F"],
        }
        subjects.append({
            "id": subject_id,
            "mri_id": row["MRI ID"],
            "group_label": group,
            "class": 0 if group == "Nondemented" else 1,
            "age": ehr["Age"],
            "sex": ehr["M/F"],
            "ehr": ehr,
        })
    return sorted(subjects, key=lambda s: s["id"])


_CDR_RE = re.compile(r"^CDR:\s*([\d.]*)", re.MULTILINE)
_AGE_RE = re.compile(r"^AGE:\s*(\d+)", re.MULTILINE)
_SEX_RE = re.compile(r"^M/F:\s*(\w+)", re.MULTILINE)


def _oasis1_subjects(raw_dir: str, imagesoasis_dir: str) -> list[dict]:
    """OASIS-1 MRI subjects: one per t88_masked_gfc volume, with CDR-derived class
    parsed from the subject's demographics .txt (stdlib regex; avoids adding an
    xlsx dependency for a single column)."""
    subjects = []
    pattern = os.path.join(raw_dir, "**", "*_t88_masked_gfc.hdr")
    for hdr_path in sorted(glob.glob(pattern, recursive=True)):
        sub_id = os.path.basename(hdr_path).split("_mpr")[0]
        subject_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(hdr_path))))
        txt_path = os.path.join(subject_dir, f"{sub_id}.txt")
        if not os.path.exists(txt_path):
            continue
        text = open(txt_path).read()
        cdr_match = _CDR_RE.search(text)
        age_match = _AGE_RE.search(text)
        sex_match = _SEX_RE.search(text)
        if not (age_match and sex_match):
            continue
        cdr = float(cdr_match.group(1)) if cdr_match and cdr_match.group(1) else 0.0
        sex = "M" if sex_match.group(1).startswith("M") else "F"

        jpgs = sorted(glob.glob(os.path.join(imagesoasis_dir, "**", f"{sub_id}_*.jpg"), recursive=True))
        jpg_path = jpgs[len(jpgs) // 2] if jpgs else None  # representative (median) slice

        subjects.append({
            "id": sub_id, "class": 1 if cdr > 0 else 0,
            "age": int(age_match.group(1)), "sex": sex,
            "hdr_path": hdr_path, "jpg_path": jpg_path,
        })
    return subjects


def _eeg_subjects(participants_tsv: str, deriv_dir: str) -> list[dict]:
    """ds004504 subjects, A/C only -- same filter as alz.eeg.build_dataset."""
    df = pd.read_csv(participants_tsv, sep="\t")
    subjects = []
    for _, row in df.iterrows():
        if row["Group"] not in ("A", "C"):
            continue
        sub_id = row["participant_id"]
        set_path = os.path.join(deriv_dir, sub_id, "eeg", f"{sub_id}_task-eyesclosed_eeg.set")
        if not os.path.exists(set_path) or os.path.getsize(set_path) < 1000:
            continue
        subjects.append({
            "id": sub_id, "class": 1 if row["Group"] == "A" else 0,
            "age": int(row["Age"]), "sex": row["Gender"], "set_path": set_path,
        })
    return subjects


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def greedy_match(patients: list[dict], candidates: list[dict]) -> dict[str, str]:
    """Assigns each patient (dicts with 'id'/'class'/'age'/'sex') the nearest-age
    candidate (same dict shape) of the same class, preferring same sex and an
    unused candidate, but reusing one rather than leaving a patient unmatched.
    Deterministic: patients and tie-broken candidate order are both sorted by id.
    """
    used = set()
    assignment = {}
    for patient in sorted(patients, key=lambda p: p["id"]):
        same_class = [c for c in candidates if c["class"] == patient["class"]]
        pool = same_class if same_class else candidates
        unused = [c for c in pool if c["id"] not in used]
        pool = unused if unused else pool
        same_sex = [c for c in pool if c["sex"] == patient["sex"]]
        pool = same_sex if same_sex else pool
        best = min(sorted(pool, key=lambda c: c["id"]), key=lambda c: abs(c["age"] - patient["age"]))
        assignment[patient["id"]] = best["id"]
        used.add(best["id"])
    return assignment


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(
    conn: sqlite3.Connection,
    oasis2_csv: str = OASIS2_CSV,
    oasis1_dir: str = OASIS1_RAW_DIR,
    imagesoasis_dir: str = IMAGESOASIS_DIR,
    eeg_participants_tsv: str = EEG_PARTICIPANTS_TSV,
    eeg_deriv_dir: str = EEG_DERIV_DIR,
) -> int:
    """(Re)populates patients + recordings from the source datasets. Idempotent.
    Returns the number of patients built."""
    clinical = _clinical_subjects(oasis2_csv)
    mri = _oasis1_subjects(oasis1_dir, imagesoasis_dir)
    eeg = _eeg_subjects(eeg_participants_tsv, eeg_deriv_dir)

    mri_match = greedy_match(clinical, mri) if mri else {}
    eeg_match = greedy_match(clinical, eeg) if eeg else {}
    mri_by_id = {m["id"]: m for m in mri}
    eeg_by_id = {e["id"]: e for e in eeg}

    conn.execute("DELETE FROM patients")
    conn.execute("DELETE FROM recordings")
    for patient in clinical:
        conn.execute(
            "INSERT INTO patients (patient_id, mri_id, group_label, ehr_json) VALUES (?, ?, ?, ?)",
            (patient["id"], patient["mri_id"], patient["group_label"], json.dumps(patient["ehr"])),
        )
        mri_subject = mri_by_id.get(mri_match.get(patient["id"]))
        if mri_subject:
            conn.execute(
                "INSERT OR REPLACE INTO recordings VALUES (?, 'mri_3d', ?, ?)",
                (patient["id"], mri_subject["id"], mri_subject["hdr_path"]),
            )
            if mri_subject["jpg_path"]:
                conn.execute(
                    "INSERT OR REPLACE INTO recordings VALUES (?, 'mri_2d', ?, ?)",
                    (patient["id"], mri_subject["id"], mri_subject["jpg_path"]),
                )
        eeg_subject = eeg_by_id.get(eeg_match.get(patient["id"]))
        if eeg_subject:
            conn.execute(
                "INSERT OR REPLACE INTO recordings VALUES (?, 'eeg', ?, ?)",
                (patient["id"], eeg_subject["id"], eeg_subject["set_path"]),
            )
    conn.commit()
    return len(clinical)


# ---------------------------------------------------------------------------
# Results cache + batch runner
# ---------------------------------------------------------------------------
def save_result(conn: sqlite3.Connection, patient_id: str, modality: str, input_key: str, result: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO results (patient_id, modality, input_key, result_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (patient_id, modality, input_key, json.dumps(result, default=_json_default), _now()),
    )
    conn.commit()


def get_result(conn: sqlite3.Connection, patient_id: str, modality: str) -> tuple[str, dict] | tuple[None, None]:
    row = conn.execute(
        "SELECT input_key, result_json FROM results WHERE patient_id = ? AND modality = ?",
        (patient_id, modality),
    ).fetchone()
    if row is None:
        return None, None
    result = json.loads(row["result_json"])
    if modality == "mri_3d" and isinstance(result.get("slice_array"), list):
        result["slice_array"] = np.array(result["slice_array"], dtype=np.uint8)
    return row["input_key"], result


def get_recordings(conn: sqlite3.Connection, patient_id: str) -> dict[str, str]:
    """{'mri_2d'/'mri_3d'/'eeg': path}, one path per modality (first if duplicates)."""
    rows = conn.execute(
        "SELECT modality, path FROM recordings WHERE patient_id = ?", (patient_id,)
    ).fetchall()
    return {row["modality"]: row["path"] for row in rows}


def list_patients(conn: sqlite3.Connection) -> list[dict]:
    return [dict(row) for row in conn.execute("SELECT patient_id, mri_id, group_label FROM patients ORDER BY patient_id")]


def run_patient(conn: sqlite3.Connection, patient_id: str, force: bool = False) -> dict:
    """Runs (or fetches cached) clinical + MRI 2D + MRI 3D + EEG inference for one
    patient, plus the fused score. Missing/failed modalities come back None; one
    bad file must not abort a cohort batch, so each modality is isolated."""
    from alz import predict as predict_clinical
    from alz.eeg import load_recording, predict_eeg_probs
    from alz.fusion import combine_mri, integrated_score
    from alz.imaging import predict_mri_probs, predict_mri_probs_3d

    patient_row = conn.execute("SELECT ehr_json FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
    if patient_row is None:
        raise ValueError(f"unknown patient_id: {patient_id}")
    ehr = json.loads(patient_row["ehr_json"])
    recordings = get_recordings(conn, patient_id)

    out = {}

    def _run(modality: str, input_key: str, fn):
        if not force:
            cached_key, cached = get_result(conn, patient_id, modality)
            if cached_key == input_key:
                out[modality] = cached
                return
        try:
            result = fn()
        except Exception as exc:
            out[modality] = {"error": str(exc)}
        else:
            out[modality] = result
            save_result(conn, patient_id, modality, input_key, result)

    _run("clinical", json.dumps(ehr, sort_keys=True), lambda: predict_clinical(ehr))
    if "mri_2d" in recordings:
        _run("mri_2d", recordings["mri_2d"], lambda: predict_mri_probs(recordings["mri_2d"]))
    if "mri_3d" in recordings:
        _run("mri_3d", recordings["mri_3d"], lambda: predict_mri_probs_3d(recordings["mri_3d"]))
    if "eeg" in recordings:
        _run("eeg", recordings["eeg"], lambda: predict_eeg_probs(load_recording(recordings["eeg"])))

    ok = {m: r for m, r in out.items() if r is not None and "error" not in r}
    mri = None
    if "mri_2d" in ok and "mri_3d" in ok:
        mri = combine_mri(ok["mri_2d"], ok["mri_3d"])
    elif "mri_3d" in ok:
        mri = ok["mri_3d"]
    elif "mri_2d" in ok:
        mri = ok["mri_2d"]
    fusion = integrated_score(clinical=ok.get("clinical"), mri=mri, eeg=ok.get("eeg"))
    if fusion:
        save_result(conn, patient_id, "fusion", "derived", fusion)
    out["fusion"] = fusion
    return out


def cohort_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per patient with per-modality + fusion scores, for the cohort table/charts."""
    rows = []
    for patient in list_patients(conn):
        pid = patient["patient_id"]
        ehr = json.loads(conn.execute("SELECT ehr_json FROM patients WHERE patient_id = ?", (pid,)).fetchone()[0])
        row = {"patient_id": pid, "group_label": patient["group_label"], "Age": ehr["Age"], "MMSE": ehr["MMSE"], "sex": ehr["M/F"]}
        for modality in ["clinical", "mri_2d", "mri_3d", "eeg", "fusion"]:
            _, result = get_result(conn, pid, modality)
            row[f"{modality}_score"] = result.get("score") if result and "error" not in result else None
        _, fusion = get_result(conn, pid, "fusion")
        row["fusion_label"] = fusion.get("score", 0) >= 0.5 if fusion else None
        rows.append(row)
    return pd.DataFrame(rows)


def cohort_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """patient_id + the 9 clinical feature columns, one row per patient, for clustering."""
    from alz import data

    frames = []
    for row in conn.execute("SELECT patient_id, ehr_json FROM patients ORDER BY patient_id"):
        frame = data.record_to_frame(json.loads(row["ehr_json"]))
        frame.insert(0, "patient_id", row["patient_id"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["patient_id"] + data.FEATURE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def cluster_labels(features: pd.DataFrame, k: int = 3) -> pd.Series:
    """KMeans phenotype cluster id per row of cohort_features(). Clusters come from the
    clinical features rather than the model scores, so "this cluster runs hotter than the
    population" stays a finding instead of a tautology.

    random_state is fixed because the UI filters on these ids: without it every Streamlit
    rerun would reshuffle which patients sit in "cluster 2".
    ponytail: fixed k from the UI, no silhouette search; add tuning if clusters look arbitrary.
    """
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = features.drop(columns=["patient_id"])
    k = max(1, min(k, len(X)))
    pipeline = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        KMeans(n_clusters=k, n_init=10, random_state=42),
    )
    return pd.Series(pipeline.fit_predict(X), index=features.index, name="cluster")


COMPARE_METRICS = ["Age", "MMSE", "clinical_score", "mri_2d_score", "mri_3d_score", "eeg_score", "fusion_score"]


def _percentile(others: pd.Series, value) -> float | None:
    """Midrank percentile of `value` against `others` (the patient's own row already
    excluded), ties counting half. With self excluded, the top scorer in a cohort of any
    size lands at exactly the 100th percentile."""
    others = others.dropna()
    if others.empty or value is None or pd.isna(value):
        return None
    return float(((others < value).sum() + 0.5 * (others == value).sum()) / len(others) * 100)


def compare_stats(df: pd.DataFrame, cohort_ids, patient_id: str | None = None) -> pd.DataFrame:
    """One row per metric: the cohort subset against the whole frame, plus where one patient
    sits in each. The same call answers all three comparisons the UI offers, by varying the
    two inputs: patient vs similar (cohort = that patient's cluster), patient vs whole
    population (cohort = everyone), cluster vs whole population (patient_id=None)."""
    cohort = df[df["patient_id"].isin(list(cohort_ids))]
    patient = df[df["patient_id"] == patient_id] if patient_id else df.iloc[:0]
    cohort_others = cohort[cohort["patient_id"] != patient_id]
    population_others = df[df["patient_id"] != patient_id]
    rows = []
    for metric in COMPARE_METRICS:
        if metric not in df.columns:
            continue
        value = patient[metric].iloc[0] if len(patient) else None
        rows.append({
            "metric": metric,
            "patient": value,
            "cohort_n": int(cohort[metric].notna().sum()),
            "cohort_median": cohort[metric].median(),
            "cohort_pct": _percentile(cohort_others[metric], value),
            "population_n": int(df[metric].notna().sum()),
            "population_median": df[metric].median(),
            "population_pct": _percentile(population_others[metric], value),
        })
    return pd.DataFrame(rows)


def demo():  # ponytail-required self-check for the matching + cache-hit logic
    patients = [
        {"id": "p1", "class": 1, "age": 80, "sex": "F"},
        {"id": "p2", "class": 0, "age": 40, "sex": "M"},
    ]
    candidates = [
        {"id": "c1", "class": 1, "age": 78, "sex": "F"},
        {"id": "c2", "class": 0, "age": 41, "sex": "M"},
        {"id": "c3", "class": 1, "age": 20, "sex": "F"},
    ]
    match = greedy_match(patients, candidates)
    assert match["p1"] == "c1", "nearest-age same-class-same-sex candidate should win"
    assert match["p2"] == "c2"
    assert greedy_match(patients, candidates) == match, "matching must be deterministic"

    # Reuse when a class runs out of candidates.
    forced_reuse = greedy_match(
        [{"id": "a", "class": 1, "age": 50, "sex": "M"}, {"id": "b", "class": 1, "age": 51, "sex": "M"}],
        [{"id": "only", "class": 1, "age": 50, "sex": "M"}],
    )
    assert forced_reuse == {"a": "only", "b": "only"}

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    save_result(conn, "p1", "eeg", "key-a", {"score": 0.5, "label": "x"})
    key, result = get_result(conn, "p1", "eeg")
    assert key == "key-a" and result["score"] == 0.5
    assert get_result(conn, "nope", "eeg") == (None, None)

    stats_df = pd.DataFrame({
        "patient_id": ["s1", "s2", "s3"],
        "Age": [60, 70, 80],
        "fusion_score": [0.1, 0.5, 0.9],
    })
    stats = compare_stats(stats_df, cohort_ids=["s2", "s3"], patient_id="s3")
    fusion_row = stats[stats["metric"] == "fusion_score"].iloc[0]
    assert fusion_row["patient"] == 0.9
    assert fusion_row["cohort_pct"] == 100.0, "the max value in its own cohort should rank at 100"
    assert fusion_row["population_pct"] == 100.0
    assert fusion_row["cohort_n"] == 2 and fusion_row["population_n"] == 3


demo()
