"""Streamlit clinical workstation demo: patient triage across clinical, MRI and EEG signals.

No EHR/database exists yet -- patient context and cross-visit history live in
st.session_state with a seeded demo patient (ponytail: swap for a real fetch when
a patient store exists). Likewise there is no EEG model/data pipeline yet, so the
EEG tab is an illustrative placeholder (ponytail: wire up a real model + parser
when one exists).
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from alz import predict
from alz.data import FEATURE_META, load_population
from alz.explain import chat_about_case, explain_mri, synthesize_summary

st.set_page_config(page_title="Alzheimer's Early-Risk Triage", page_icon="🧠", layout="wide")
st.logo(os.path.join(os.path.dirname(__file__), "..", "TAO_logo.png"), size="large")
st.markdown(
    "<style>div[data-testid='stTabs'] button[role='tab']{padding:8px 18px;} "
    "div[data-testid='stMetricValue']{font-size:1.4rem;}</style>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Shared severity coding (green/amber/red) used across every tab + sidebar.
# ---------------------------------------------------------------------------
SEVERITY_COLORS = {
    "normal": ("#4CAF3D", "#FFFFFF"),
    "mild": ("#F5E27E", "#3A3000"),
    "high": ("#E12A26", "#FFFFFF"),
}


def chip(label: str, level: str) -> str:
    bg, fg = SEVERITY_COLORS.get(level, SEVERITY_COLORS["mild"])
    return (
        f'<span style="background-color:{bg}; color:{fg}; padding:2px 10px; '
        f'border-radius:12px; font-size:0.85em; font-weight:600;">{label}</span>'
    )


def band_for_score(score: float) -> tuple[str, str]:
    """(severity label, level) for a clinical risk score, matching the OASIS group names."""
    if score < 0.33:
        return "Non Demented", "normal"
    if score < 0.66:
        return "Mild Dementia", "mild"
    return "Moderate Dementia", "high"


MRI_LEVELS = {"Non Demented": "normal", "Mild Dementia": "mild", "Moderate Dementia": "high"}


def demo():  # ponytail-required self-check for the only nontrivial pure logic here
    assert band_for_score(0.1)[1] == "normal"
    assert band_for_score(0.5)[1] == "mild"
    assert band_for_score(0.9)[1] == "high"


demo()

# ---------------------------------------------------------------------------
# Session state: seeded demo patient + per-tab results (no real persistence).
# ---------------------------------------------------------------------------
if "patient" not in st.session_state:
    st.session_state.patient = {
        "id": "OASIS-10234",
        "name": "Jane Doe",
        "dob": "1950-03-14",
        "sex": "F",
        "history": "Hypertension; type 2 diabetes; former smoker (quit 2010).",
    }
st.session_state.setdefault("clinical_result", None)
st.session_state.setdefault("clinical_record", None)
st.session_state.setdefault("mri_scans", [])  # [{"name","bytes","result"}]
st.session_state.setdefault("mri_selected", None)
st.session_state.setdefault("eeg_result", None)
st.session_state.setdefault("mmse_history", [29, 28])  # demo prior-visit MMSE scores
st.session_state.setdefault("notes", "")

"""
# :material/neurology: Alzheimer's Early-Risk Triage
"""
""

# ---------------------------------------------------------------------------
# Persistent patient context (sidebar) -- visible across all tabs.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Patient")
    patient = st.session_state.patient
    with st.expander("Patient details", expanded=True):
        patient["name"] = st.text_input("Name", patient["name"])
        patient["id"] = st.text_input("Patient ID", patient["id"])
        c1, c2 = st.columns(2)
        patient["dob"] = c1.text_input("DOB", patient["dob"])
        patient["sex"] = c2.selectbox("Sex", ["F", "M"], index=["F", "M"].index(patient["sex"]))
        patient["history"] = st.text_area("Medical history", patient["history"], height=80)
    st.caption(f"{patient['name']} · {patient['sex']} · DOB {patient['dob']} · ID {patient['id']}")

    st.divider()
    st.caption("Latest assessments")
    if st.session_state.clinical_result:
        label, level = band_for_score(st.session_state.clinical_result["score"])
        st.markdown(f"Clinical risk: {chip(label, level)}", unsafe_allow_html=True)
    else:
        st.caption("Clinical risk: not yet assessed")
    if st.session_state.mri_selected:
        mri = st.session_state.mri_selected["result"]
        st.markdown(f"MRI severity: {chip(mri['label'], MRI_LEVELS.get(mri['label'], 'mild'))}", unsafe_allow_html=True)
    else:
        st.caption("MRI severity: not yet assessed")
    if st.session_state.eeg_result:
        eeg = st.session_state.eeg_result
        st.markdown(f"EEG: {chip(eeg['label'], eeg['level'])}", unsafe_allow_html=True)
    else:
        st.caption("EEG: not yet assessed")

tab_overview, tab_clinical, tab_mri, tab_eeg = st.tabs(
    ["Overview", "Clinical risk", "MRI records", "EEG records"]
)

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Assessment summary")
    cols = st.columns(4)

    with cols[0]:
        with st.container(border=True):
            st.caption("MMSE (latest)")
            record = st.session_state.clinical_record
            if record:
                prior = st.session_state.mmse_history[-1] if st.session_state.mmse_history else None
                delta = None if prior is None else record["MMSE"] - prior
                st.metric("MMSE score", record["MMSE"], delta)
            else:
                st.metric("MMSE score", "—")

    with cols[1]:
        with st.container(border=True):
            st.caption("Clinical risk")
            result = st.session_state.clinical_result
            if result:
                label, level = band_for_score(result["score"])
                st.markdown(chip(label, level), unsafe_allow_html=True)
                st.caption(f"Score: {result['score']:.0%}")
            else:
                st.caption("Not yet assessed")

    with cols[2]:
        with st.container(border=True):
            st.caption("MRI severity")
            selected = st.session_state.mri_selected
            if selected:
                mri = selected["result"]
                st.markdown(chip(mri["label"], MRI_LEVELS.get(mri["label"], "mild")), unsafe_allow_html=True)
                st.caption(f"Confidence: {mri['score']:.0%}")
            else:
                st.caption("Not yet assessed")

    with cols[3]:
        with st.container(border=True):
            st.caption("EEG classification")
            eeg = st.session_state.eeg_result
            if eeg:
                st.markdown(chip(eeg["label"], eeg["level"]), unsafe_allow_html=True)
                st.caption(f"Confidence: {eeg['score']:.0%}")
            else:
                st.caption("Not yet assessed")

    st.divider()
    st.subheader("Clinical conclusions")
    mri_result = selected["result"] if selected else None
    if st.button(":material/auto_awesome: Generate AI summary"):
        with st.spinner("Synthesizing..."):
            summary = synthesize_summary(patient, result, mri_result, eeg)
        if summary:
            st.session_state.notes = summary
        else:
            st.warning("LLM summary unavailable (check LLM provider configuration).")
    st.session_state.notes = st.text_area(
        "Synthesized notes",
        st.session_state.notes,
        height=150,
        placeholder="Summarize the combined AI assessments and clinical impression here...",
        label_visibility="collapsed",
    )

    st.subheader("Ask about this case")
    st.session_state.setdefault("chat_history", [])
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    if question := st.chat_input("Ask a question about this patient's case..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            reply = chat_about_case(st.session_state.chat_history, patient, result, mri_result, eeg)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply or "Sorry, I couldn't answer that (check LLM provider configuration)."}
        )
        st.rerun()

# ---------------------------------------------------------------------------
# Tab 2: Clinical risk
# ---------------------------------------------------------------------------
with tab_clinical:
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.container(border=True):
                st.markdown("**Demographics**")
                age = st.number_input("Age", min_value=18, max_value=110, value=75, help=FEATURE_META["Age"]["definition"])
                sex = st.selectbox("Sex", ["F", "M"], help=FEATURE_META["Sex_M"]["definition"])
                ses = st.number_input("Socioeconomic status (1=highest, 5=lowest)", min_value=1, max_value=5, value=3, help=FEATURE_META["SES"]["definition"])
                educ = st.number_input("Years of education", min_value=0, max_value=30, value=12, help=FEATURE_META["EDUC"]["definition"])
        with col2:
            with st.container(border=True):
                st.markdown("**Cognitive scores**")
                mmse = st.number_input(
                    "MMSE score (0-30)", min_value=0, max_value=30, value=27,
                    help="Mini-Mental State Exam: a bedside 30-point cognitive screening test "
                         "(memory, attention, language, orientation). Lower scores mean more impairment; "
                         "27-30 is normal, <24 typically flags concern.",
                )
                visit = st.number_input("Visit number", min_value=1, max_value=10, value=1, help=FEATURE_META["Visit"]["definition"])
        with col3:
            with st.container(border=True):
                st.markdown("**Brain volumetrics**")
                nwbv = st.number_input("Normalized whole-brain volume", min_value=0.5, max_value=0.95, value=0.75, format="%.3f", help=FEATURE_META["nWBV"]["definition"])
                asf = st.number_input(
                    "Atlas scaling factor", min_value=0.7, max_value=1.7, value=1.2, format="%.3f",
                    help="A head-size correction factor used to normalize MRI volume measurements "
                         "(like nWBV) so brains of different physical sizes can be compared fairly. "
                         "It reflects skull/head size, not disease severity.",
                )

        submitted = st.form_submit_button("Assess risk")

    if submitted:
        record = {
            "Age": age, "EDUC": educ, "SES": ses, "MMSE": mmse,
            "nWBV": nwbv, "ASF": asf, "Visit": visit, "MR Delay": 0,
            "M/F": sex,
        }
        result = predict(record)
        st.session_state.clinical_record = record
        st.session_state.clinical_result = result

    record = st.session_state.clinical_record
    result = st.session_state.clinical_result
    if result:
        severity_label, severity_level = band_for_score(result["score"])

        with st.container(border=True):
            top = st.columns([2, 2, 1])
            top[0].metric("Risk score", f"{result['score']:.0%}", result["label"])
            with top[1]:
                st.write("")
                st.markdown(chip(severity_label, severity_level), unsafe_allow_html=True)
            if record["Visit"] > 1 and st.session_state.mmse_history:
                prior = st.session_state.mmse_history[-1]
                with top[2]:
                    st.metric("MMSE vs. last visit", record["MMSE"], record["MMSE"] - prior)
                spark = go.Figure(go.Scatter(
                    y=st.session_state.mmse_history + [record["MMSE"]],
                    mode="lines+markers", line_color="#4C8BF5",
                ))
                spark.update_layout(height=100, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="MMSE")
                st.plotly_chart(spark, use_container_width=True)

            st.caption(
                "From a logistic regression trained on these 9 clinical features; score is the "
                "model's predicted probability of the higher-risk class. Drivers below show which "
                "inputs pushed it up or down."
            )
            st.subheader("Top contributing factors")
            population = load_population()
            for driver in result["drivers"]:
                meta = FEATURE_META.get(driver["feature"], {})
                label = meta.get("label", driver["feature"])
                tooltip = f"{meta.get('definition', '')} Usual range: {meta.get('range', 'n/a')}."
                is_sex = driver["feature"] == "Sex_M"
                if is_sex:
                    same_sex_ages = population.loc[population["Sex_M"] == driver["value"], "Age"]
                    age_percentile = (same_sex_ages < record["Age"]).mean() * 100
                    caption = (
                        f'(this patient: {"Male" if driver["value"] == 1 else "Female"}, '
                        f'age {age_percentile:.0f}th percentile for their sex)'
                    )
                else:
                    caption = f'(this patient: {driver["value"]:.2g}, {driver["percentile"]:.0f}th percentile)'
                st.markdown(
                    f'<span title="{tooltip}" style="text-decoration: underline dotted; cursor: help;">'
                    f'<b>{label}</b></span> — {driver["direction"]} {caption}',
                    unsafe_allow_html=True,
                )

            with st.expander("Population comparison charts"):
                for driver in result["drivers"]:
                    meta = FEATURE_META.get(driver["feature"], {})
                    label = meta.get("label", driver["feature"])
                    is_sex = driver["feature"] == "Sex_M"
                    fig = go.Figure()
                    if is_sex:
                        # ponytail: binary percentile is meaningless -- show Age split by sex instead
                        fig.add_trace(go.Histogram(x=population.loc[population["Sex_M"] == 1, "Age"], name="Male", marker_color="#636EFA", opacity=0.6))
                        fig.add_trace(go.Histogram(x=population.loc[population["Sex_M"] == 0, "Age"], name="Female", marker_color="#EF553B", opacity=0.6))
                        fig.update_layout(barmode="overlay")
                        fig.add_vline(x=record["Age"], line_color="crimson", line_width=2, annotation_text="This patient", annotation_position="top")
                        fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Age", showlegend=True)
                    else:
                        fig.add_trace(go.Histogram(x=population[driver["feature"]], name="Population", marker_color="#636EFA"))
                        fig.add_vline(x=driver["value"], line_color="crimson", line_width=2, annotation_text="This patient", annotation_position="top")
                        fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=label, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: MRI records
# ---------------------------------------------------------------------------
with tab_mri:
    with st.container(border=True):
        uploaded = st.file_uploader("Upload a new brain scan", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            uploaded_bytes = uploaded.getvalue()
            if not any(s["name"] == uploaded.name for s in st.session_state.mri_scans):
                st.session_state.mri_scans.append({"name": uploaded.name, "bytes": uploaded_bytes, "result": None})
            st.session_state.mri_selected_name = uploaded.name

        if st.session_state.mri_scans:
            names = [s["name"] for s in st.session_state.mri_scans]
            default_idx = names.index(st.session_state.get("mri_selected_name", names[-1])) if st.session_state.get("mri_selected_name") in names else len(names) - 1
            chosen_name = st.selectbox("Scan history", names, index=default_idx)
            scan = next(s for s in st.session_state.mri_scans if s["name"] == chosen_name)

            try:
                from alz.imaging import gradcam_mri, predict_mri_probs
            except ImportError:
                st.warning("MRI dependencies not installed — see requirements-imaging.txt")
            else:
                if scan["result"] is None:
                    scan["result"] = predict_mri_probs(io.BytesIO(scan["bytes"]))
                result = scan["result"]
                st.session_state.mri_selected = scan

                st.metric("Predicted severity", result["label"], f"{result['score']:.0%} confidence")

                probs = result["probs"]
                bar = go.Figure(go.Bar(
                    x=list(probs.values()), y=list(probs.keys()), orientation="h",
                    marker_color=["#4CAF3D", "#F5E27E", "#E12A26"][: len(probs)],
                    text=[f"{v:.0%}" for v in probs.values()], textposition="inside",
                ))
                bar.update_layout(height=140, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 1], tickformat=".0%"))
                st.plotly_chart(bar, use_container_width=True)

                img_col, cam_col = st.columns(2)
                with img_col:
                    st.image(scan["bytes"], caption="Uploaded scan", use_container_width=True)
                with cam_col:
                    try:
                        with st.spinner("Computing Grad-CAM heatmap..."):
                            cam = gradcam_mri(io.BytesIO(scan["bytes"]))
                        st.image(cam["overlay"], caption="Grad-CAM: regions driving the prediction", use_container_width=True)
                    except Exception:
                        st.info("Heatmap unavailable for this image.")

                with st.expander("What do these categories mean?"):
                    st.markdown(
                        "- **Non Demented** — no clinical signs of cognitive decline.\n"
                        "- **Mild Dementia** — early, noticeable memory/cognitive changes that still allow "
                        "mostly independent daily living.\n"
                        "- **Moderate Dementia** — more pronounced cognitive and functional decline, "
                        "typically requiring regular assistance with daily activities."
                    )

                st.subheader("Radiology report summary")
                with st.container(border=True):
                    with st.spinner("Generating explanation..."):
                        explanation = explain_mri(result)
                    st.markdown(f"**IMPRESSION:** {result['label']} ({result['score']:.0%} model confidence)")
                    st.markdown("**FINDINGS:**")
                    if explanation:
                        st.write(explanation)
                    else:
                        st.caption("LLM explanation unavailable (check LLM provider configuration).")
        else:
            st.caption("Upload a scan to begin.")

# ---------------------------------------------------------------------------
# Tab 4: EEG records (placeholder -- no EEG model/data pipeline exists yet)
# ---------------------------------------------------------------------------
with tab_eeg:
    st.caption("Illustrative preview only — no EEG model is wired up yet.")
    with st.container(border=True):
        st.file_uploader("Upload EEG recording", type=["edf", "csv"])
        recording = st.selectbox("Or select a prior recording", ["Visit 1 — 2024-03-02", "Visit 2 — 2024-09-14"])
        run = st.button("Run evaluation")

    if run:
        rng = np.random.default_rng(abs(hash(recording)) % (2**32))
        slowing_p = float(rng.uniform(0.1, 0.9))
        label, level = ("Generalized slowing suspected", "high") if slowing_p > 0.6 else (
            ("Mild abnormality", "mild") if slowing_p > 0.3 else ("Normal background rhythm", "normal")
        )
        st.session_state.eeg_result = {"label": label, "score": slowing_p, "level": level, "recording": recording}
        st.rerun()

    if st.session_state.eeg_result:
        eeg = st.session_state.eeg_result
        label, level, slowing_p = eeg["label"], eeg["level"], eeg["score"]

        rng = np.random.default_rng(abs(hash(eeg["recording"])) % (2**32))
        channels = ["Fp1", "Fp2", "C3", "C4", "O1", "O2"]
        t = np.linspace(0, 4, 1000)
        fig = go.Figure()
        for i, ch in enumerate(channels):
            wave = np.sin(2 * np.pi * (1 + i * 0.5) * t) + rng.normal(0, 0.3, t.shape)
            fig.add_trace(go.Scatter(x=t, y=wave + i * 4, mode="lines", name=ch, line=dict(width=1)))
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(showticklabels=False), xaxis_title="Time (s)")
        st.subheader("Signal viewer")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Evaluation results")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(chip(label, level), unsafe_allow_html=True)
        with m2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=slowing_p * 100,
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": SEVERITY_COLORS[level][0]}},
                title={"text": "Confidence (%)"},
            ))
            gauge.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(gauge, use_container_width=True)
