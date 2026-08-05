"""Patient DB: synthetic linking + results cache, against tiny in-memory fixtures
(no real datasets needed)."""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alz import db


def _fresh_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(db._SCHEMA)
    return conn


def _populate(conn, clinical, mri, eeg):
    """Mirrors db.build's insert logic without touching disk fixtures."""
    mri_match = db.greedy_match(clinical, mri) if mri else {}
    eeg_match = db.greedy_match(clinical, eeg) if eeg else {}
    mri_by_id = {m["id"]: m for m in mri}
    eeg_by_id = {e["id"]: e for e in eeg}
    for patient in clinical:
        conn.execute(
            "INSERT INTO patients (patient_id, mri_id, group_label, ehr_json) VALUES (?, ?, ?, ?)",
            (patient["id"], patient["id"] + "_MR1", patient["group_label"], "{}"),
        )
        m = mri_by_id.get(mri_match.get(patient["id"]))
        if m:
            conn.execute("INSERT INTO recordings VALUES (?, 'mri_3d', ?, ?)", (patient["id"], m["id"], "fake.hdr"))
        e = eeg_by_id.get(eeg_match.get(patient["id"]))
        if e:
            conn.execute("INSERT INTO recordings VALUES (?, 'eeg', ?, ?)", (patient["id"], e["id"], "fake.set"))
    conn.commit()
    return clinical, mri, eeg


CLINICAL = [
    {"id": "OAS2_0001", "class": 0, "age": 78, "sex": "F", "group_label": "Nondemented"},
    {"id": "OAS2_0002", "class": 0, "age": 65, "sex": "M", "group_label": "Nondemented"},
    {"id": "OAS2_0003", "class": 1, "age": 82, "sex": "F", "group_label": "Demented"},
    {"id": "OAS2_0004", "class": 1, "age": 70, "sex": "M", "group_label": "Demented"},
]
MRI = [
    {"id": "OAS1_A", "class": 0, "age": 79, "sex": "F", "hdr_path": "a.hdr", "jpg_path": None},
    {"id": "OAS1_B", "class": 0, "age": 60, "sex": "M", "hdr_path": "b.hdr", "jpg_path": None},
    {"id": "OAS1_C", "class": 1, "age": 81, "sex": "F", "hdr_path": "c.hdr", "jpg_path": None},
    {"id": "OAS1_D", "class": 1, "age": 71, "sex": "M", "hdr_path": "d.hdr", "jpg_path": None},
]
EEG = [
    {"id": "sub-001", "class": 0, "age": 77, "sex": "F"},
    {"id": "sub-002", "class": 0, "age": 66, "sex": "M"},
    {"id": "sub-003", "class": 1, "age": 83, "sex": "F"},
    {"id": "sub-004", "class": 1, "age": 69, "sex": "M"},
]


def test_every_patient_gets_at_least_one_mri_and_eeg():
    conn = _fresh_conn()
    _populate(conn, CLINICAL, MRI, EEG)
    counts = dict(conn.execute("SELECT modality, COUNT(*) FROM recordings GROUP BY modality").fetchall())
    assert counts["mri_3d"] == len(CLINICAL)
    assert counts["eeg"] == len(CLINICAL)


def test_pairing_respects_diagnosis_class():
    conn = _fresh_conn()
    _populate(conn, CLINICAL, MRI, EEG)
    rows = conn.execute(
        "SELECT p.group_label, r.modality, r.source_id FROM patients p JOIN recordings r USING (patient_id)"
    ).fetchall()
    class_by_source = {m["id"]: m["class"] for m in MRI} | {e["id"]: e["class"] for e in EEG}
    for row in rows:
        want_demented = row["group_label"] != "Nondemented"
        assert bool(class_by_source[row["source_id"]]) == want_demented


def test_matching_is_deterministic():
    assert db.greedy_match(CLINICAL, MRI) == db.greedy_match(CLINICAL, MRI)


def test_matching_reuses_rather_than_leaving_unmatched():
    patients = [{"id": "a", "class": 1, "age": 50, "sex": "M"}, {"id": "b", "class": 1, "age": 51, "sex": "M"}]
    only = [{"id": "only", "class": 1, "age": 50, "sex": "M"}]
    match = db.greedy_match(patients, only)
    assert match == {"a": "only", "b": "only"}


def test_save_and_get_result_round_trip():
    conn = _fresh_conn()
    payload = {"score": 0.42, "probs": {"a": 0.42, "b": 0.58}, "nested": {"x": [1, 2, 3]}}
    db.save_result(conn, "OAS2_0001", "clinical", "key1", payload)
    key, result = db.get_result(conn, "OAS2_0001", "clinical")
    assert key == "key1"
    assert result == payload


def test_get_result_missing_returns_none_pair():
    conn = _fresh_conn()
    assert db.get_result(conn, "nope", "clinical") == (None, None)


def _insert_patient_with_ehr(conn, patient_id, group_label, **ehr_overrides):
    ehr = {"Visit": 1, "MR Delay": 0, "Age": 78, "EDUC": 12, "SES": 2, "MMSE": 27, "nWBV": 0.7, "ASF": 1.1, "M/F": "F"}
    ehr.update(ehr_overrides)
    conn.execute(
        "INSERT INTO patients (patient_id, mri_id, group_label, ehr_json) VALUES (?, ?, ?, ?)",
        (patient_id, patient_id + "_MR1", group_label, __import__("json").dumps(ehr)),
    )


def test_cohort_features_returns_feature_columns_per_patient():
    from alz import data

    conn = _fresh_conn()
    _insert_patient_with_ehr(conn, "p1", "Nondemented", Age=70)
    _insert_patient_with_ehr(conn, "p2", "Demented", Age=85)
    conn.commit()

    features = db.cohort_features(conn)
    assert list(features["patient_id"]) == ["p1", "p2"]
    assert list(features.columns) == ["patient_id"] + data.FEATURE_COLUMNS
    assert list(features["Age"]) == [70, 85]


def test_cluster_labels_are_stable_across_calls():
    conn = _fresh_conn()
    for i, age in enumerate([60, 62, 61, 85, 88, 84]):
        _insert_patient_with_ehr(conn, f"p{i}", "Nondemented", Age=age)
    conn.commit()

    features = db.cohort_features(conn)
    labels = db.cluster_labels(features, k=2)
    assert len(labels) == len(features)
    assert labels.nunique() == 2
    assert list(db.cluster_labels(features, k=2)) == list(labels), "random_state must pin cluster ids across reruns"


def test_compare_stats_percentiles_and_counts():
    df = __import__("pandas").DataFrame({
        "patient_id": ["a", "b", "c", "d"],
        "Age": [60, 70, 80, 90],
        "fusion_score": [0.1, 0.4, 0.6, 0.9],
    })
    stats = db.compare_stats(df, cohort_ids=["a", "b", "c", "d"], patient_id="d")
    row = stats.set_index("metric").loc["fusion_score"]
    assert row["patient"] == 0.9
    assert 0 <= row["cohort_pct"] <= 100
    assert row["cohort_pct"] == 100.0, "max value in the full cohort should sit at the 100th percentile"
    assert row["population_n"] == 4

    # Same call, narrower cohort_ids: this is how "patient vs similar" and "patient vs
    # whole population" reuse the identical function.
    narrow = db.compare_stats(df, cohort_ids=["c", "d"], patient_id="d")
    narrow_row = narrow.set_index("metric").loc["fusion_score"]
    assert narrow_row["cohort_n"] == 2
    assert narrow_row["population_n"] == 4


def test_run_patient_caches_and_force_recomputes(monkeypatch):
    conn = _fresh_conn()
    ehr = {"Visit": 1, "MR Delay": 0, "Age": 78, "EDUC": 12, "SES": 2, "MMSE": 27, "nWBV": 0.7, "ASF": 1.1, "M/F": "F"}
    conn.execute(
        "INSERT INTO patients (patient_id, mri_id, group_label, ehr_json) VALUES (?, ?, ?, ?)",
        ("p1", "p1_MR1", "Nondemented", __import__("json").dumps(ehr)),
    )
    conn.commit()

    calls = {"n": 0}

    def fake_predict(record):
        calls["n"] += 1
        return {"score": 0.1, "label": "Normal", "drivers": []}

    monkeypatch.setattr("alz.predict", fake_predict)
    db.run_patient(conn, "p1")
    assert calls["n"] == 1
    db.run_patient(conn, "p1")
    assert calls["n"] == 1, "second run should hit the cache, not recompute"
    db.run_patient(conn, "p1", force=True)
    assert calls["n"] == 2, "force=True should recompute"
