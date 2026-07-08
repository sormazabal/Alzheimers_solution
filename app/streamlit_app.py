"""Streamlit clinical workstation demo: patient triage across clinical, MRI and EEG signals.

No EHR/database exists yet -- patient context and cross-visit history live in
st.session_state with a seeded demo patient (ponytail: swap for a real fetch when
a patient store exists).
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
from alz.explain import chat_about_case, evidence_for_case, explain_eeg, explain_mri, synthesize_summary
from alz.fusion import integrated_score

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

# Display name + hover explanation for each EEG feature (alz.eeg.FEATURE_NAMES).
EEG_FEATURE_INFO = {
    "delta": ("Delta power", "0.5-4 Hz power. Elevated delta is part of the 'EEG slowing' pattern seen in Alzheimer's."),
    "theta": ("Theta power", "4-8 Hz power. Elevated theta is part of the 'EEG slowing' pattern seen in Alzheimer's."),
    "alpha": ("Alpha power", "8-13 Hz power. Reduced alpha (relative to delta/theta) is associated with Alzheimer's."),
    "beta": ("Beta power", "13-25 Hz power. Reduced beta is associated with Alzheimer's."),
    "gamma": ("Gamma power", "25-45 Hz power. Least specific band for dementia screening."),
    "theta_alpha_ratio": ("Theta / Alpha ratio", "Slow-to-fast power ratio. Higher values mean more EEG slowing, a strong Alzheimer's indicator."),
    "slowing_ratio": ("Slowing ratio", "(delta+theta) / (alpha+beta), an overall EEG-slowing index. Higher values are more consistent with Alzheimer's."),
}


def eeg_feature_display(name: str) -> tuple[str, str]:
    return EEG_FEATURE_INFO.get(name, (name.replace("_", " ").title(), ""))


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


MRI_LEVELS = {"Non Demented": "normal", "Demented": "high"}

def next_steps(level: str, pending: list[str]) -> str:
    """Recommendation text for a severity level, naming only assessments not yet run."""
    if not pending:
        return {
            "normal": "All assessments complete; no further workup indicated. Continue routine annual screening.",
            "mild": "All assessments complete. Recommend a follow-up visit in 6 months.",
            "high": "All assessments complete. Recommend referral to a memory/neurology specialist.",
        }[level]
    todo = "/".join(pending)
    if level == "normal":
        return f"Recommend completing the remaining assessment(s) ({todo}) to confirm this finding."
    if level == "mild":
        return f"Recommend completing the remaining assessment(s) ({todo}) and a follow-up visit in 6 months."
    return f"Recommend completing the remaining assessment(s) ({todo}) promptly and referral to a memory/neurology specialist."


def demo():  # ponytail-required self-check for the only nontrivial pure logic here
    assert band_for_score(0.1)[1] == "normal"
    assert band_for_score(0.5)[1] == "mild"
    assert band_for_score(0.9)[1] == "high"
    assert "MRI" in next_steps("mild", ["MRI", "EEG"])
    assert "remaining" not in next_steps("high", [])


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
st.session_state.setdefault("mri_selected", None)
st.session_state.setdefault("mri_2d_result", None)
st.session_state.setdefault("mri_3d_result", None)
st.session_state.setdefault("eeg_result", None)
st.session_state.setdefault("mmse_history", [29, 28])  # demo prior-visit MMSE scores
st.session_state.setdefault("notes", "")
st.session_state.setdefault("evidence", None)

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
    fused_sidebar = integrated_score(
        clinical=st.session_state.clinical_result,
        mri=(st.session_state.mri_selected or {}).get("result"),
        eeg=st.session_state.eeg_result,
    )
    if fused_sidebar:
        label, level = band_for_score(fused_sidebar["score"])
        st.markdown(f"Integrated prognosis: {chip(label, level)} ({fused_sidebar['score']:.0%})", unsafe_allow_html=True)
    else:
        st.caption("Integrated prognosis: not yet assessed")

    st.caption("Latest assessments")
    if st.session_state.clinical_result:
        label, level = band_for_score(st.session_state.clinical_result["score"])
        st.markdown(f"Clinical risk: {chip(label, level)}", unsafe_allow_html=True)
    else:
        st.caption("Clinical risk: not yet assessed")
    if st.session_state.mri_selected:
        mri = st.session_state.mri_selected["result"]
        st.markdown(f"MRI: {chip(mri['label'], MRI_LEVELS.get(mri['label'], 'high'))}", unsafe_allow_html=True)
    else:
        st.caption("MRI: not yet assessed")
    if st.session_state.eeg_result:
        eeg = st.session_state.eeg_result
        st.markdown(f"EEG: {chip(eeg['label'], eeg['level'])}", unsafe_allow_html=True)
    else:
        st.caption("EEG: not yet assessed")

tab_clinical, tab_mri, tab_eeg, tab_overview = st.tabs(
    ["Clinical risk", "MRI records", "EEG records", "Overview"]
)

# ---------------------------------------------------------------------------
# Tab 4: Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Assessment summary")

    fused = integrated_score(
        clinical=st.session_state.clinical_result,
        mri=(st.session_state.mri_selected or {}).get("result"),
        eeg=st.session_state.eeg_result,
    )
    with st.container(border=True):
        st.caption("Integrated prognosis")
        if fused:
            label, level = band_for_score(fused["score"])
            st.markdown(
                f'{chip(label, level)} &nbsp; <span style="font-size:1.4em; font-weight:600;">'
                f'{fused["score"]:.0%}</span> <span style="opacity:0.7;">posterior probability of dementia</span>',
                unsafe_allow_html=True,
            )
            st.caption(fused["note"])
            with st.expander("How this was combined"):
                st.caption("Each modality's own dementia-risk estimate (not shares of a total):")
                for c in fused["components"]:
                    st.caption(f"{c['modality']}: {c['p']:.0%}, weighted {c['weight']:.1f}x")
                st.caption(
                    "These are averaged in log-odds space (a logarithmic opinion pool) to "
                    "produce the combined probability above."
                )
        else:
            st.caption("Run at least one assessment to see the integrated score.")

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
            st.caption("MRI")
            selected = st.session_state.mri_selected
            if selected:
                mri = selected["result"]
                st.markdown(chip(mri["label"], MRI_LEVELS.get(mri["label"], "high")), unsafe_allow_html=True)
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

    st.subheader("Evidence-based suggestions")
    if st.button(":material/menu_book: Find evidence & trials"):
        with st.spinner("Searching PubMed and ClinicalTrials.gov..."):
            st.session_state.evidence = evidence_for_case(patient, result, mri_result, eeg)
    evidence = st.session_state.evidence
    if evidence:
        if evidence["pubmed"]:
            st.markdown("**Guidelines (PubMed)**")
            for a in evidence["pubmed"]:
                st.markdown(f"- [PMID {a['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/) — {a['title']} ({a['source']}, {a['pubdate']})")
        if evidence["trials"]:
            st.markdown("**Recruiting trials (ClinicalTrials.gov)**")
            for t in evidence["trials"]:
                st.markdown(f"- [{t['nct']}](https://clinicaltrials.gov/study/{t['nct']}) — {t['title']} ({t['status']})")
        if not evidence["pubmed"] and not evidence["trials"]:
            st.caption("No live results (check network access or API availability).")
        if evidence["recommendation"]:
            st.markdown("**Recommended approach**")
            st.write(evidence["recommendation"])
        else:
            st.caption("LLM recommendation unavailable (check LLM provider configuration).")
        with st.expander("Raw evidence data", expanded=False):
            st.json(evidence)

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
                mr_delay = st.number_input("MRI delay (days)", min_value=0, max_value=2000, value=0, help=FEATURE_META["MR Delay"]["definition"])

        submitted = st.form_submit_button("Assess risk")

    if submitted:
        record = {
            "Age": age, "EDUC": educ, "SES": ses, "MMSE": mmse,
            "nWBV": nwbv, "ASF": asf, "Visit": visit, "MR Delay": mr_delay,
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
            pending = [
                name for name, done in [("MRI", st.session_state.mri_selected), ("EEG", st.session_state.eeg_result)]
                if not done
            ]
            st.info(f"**Recommended next steps:** {next_steps(severity_level, pending)}")
            if record["Visit"] > 1 and st.session_state.mmse_history:
                prior = st.session_state.mmse_history[-1]
                with top[2]:
                    st.metric("MMSE vs. last visit", record["MMSE"], record["MMSE"] - prior)
                spark = go.Figure(go.Scatter(
                    y=st.session_state.mmse_history + [record["MMSE"]],
                    mode="lines+markers", line_color="#4C8BF5",
                ))
                spark.update_layout(height=100, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="MMSE")
                st.plotly_chart(spark, width="stretch")

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
                    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Tab 3: MRI records
# ---------------------------------------------------------------------------
IMAGESOASIS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "imagesoasis", "versions", "1", "Data")
OASIS1_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "OASIS1_raw")


@st.cache_data
def _mri_2d_examples() -> list[dict]:
    """A few example 2D slices per class from the imagesoasis mirror, for a picker
    like the EEG tab's example dropdown.

    ponytail: caps at 3 per class; raise if a bigger gallery is wanted.
    """
    import glob

    examples = []
    for cls in ["Non Demented", "Mild Dementia", "Moderate Dementia"]:
        paths = sorted(glob.glob(os.path.join(IMAGESOASIS_DIR, cls, "*.jpg")))[:3]
        for path in paths:
            examples.append({"name": f"{cls} — {os.path.basename(path)}", "path": path})
    return examples


@st.cache_data
def _oasis1_subjects() -> list[dict]:
    """OASIS-1 subjects available on disk under data/OASIS1_raw, using the
    atlas-registered, masked, gain-field-corrected volume for each subject.

    ponytail: fixed to the t88_masked_gfc variant; add a variant selector if asked.
    """
    import glob

    pattern = os.path.join(OASIS1_RAW_DIR, "**", "*_t88_masked_gfc.hdr")
    paths = sorted(glob.glob(pattern, recursive=True))
    return [{"name": os.path.basename(p).split("_mpr")[0], "path": p} for p in paths]


@st.cache_data(show_spinner=False)
def _cached_predict_2d(image_bytes: bytes) -> dict:
    from alz.imaging import predict_mri_probs
    return predict_mri_probs(io.BytesIO(image_bytes))


@st.cache_data(show_spinner=False)
def _cached_gradcam_2d(image_bytes: bytes) -> dict:
    from alz.imaging import gradcam_mri
    return gradcam_mri(io.BytesIO(image_bytes))


@st.cache_data(show_spinner=False)
def _cached_predict_3d(volume_path: str) -> dict:
    from alz.imaging import predict_mri_probs_3d
    return predict_mri_probs_3d(volume_path)


@st.cache_data(show_spinner=False)
def _cached_gradcam_3d(volume_path: str) -> dict:
    from alz.imaging import gradcam_mri_3d
    return gradcam_mri_3d(volume_path)


@st.cache_data(show_spinner=False)
def _cached_gradcam_volume_3d(volume_path: str) -> dict:
    from alz.imaging import gradcam_volume_3d
    return gradcam_volume_3d(volume_path)


# ponytail: unbounded cache; add max_entries=/ttl= if uploaded-temp-file churn grows memory.
@st.cache_data(show_spinner=False)
def _cached_volume_figure(volume_path: str, cam_volume=None):
    from alz.imaging import mri_volume_figure
    return mri_volume_figure(volume_path, cam_volume=cam_volume)


def _run_mri_2d(image_bytes: bytes, image_name: str):
    """Runs the 2D MRI model and renders results. Shared by upload/example/history."""
    result = _cached_predict_2d(image_bytes)
    st.session_state.mri_selected = {"name": image_name, "bytes": image_bytes, "result": result}

    st.metric("Dementia confirmation", result["label"], f"{result['score']:.0%} confidence")
    pending = [
        name for name, done in [("Clinical", st.session_state.clinical_result), ("EEG", st.session_state.eeg_result)]
        if not done
    ]
    st.info(f"**Recommended next steps:** {next_steps(MRI_LEVELS.get(result['label'], 'high'), pending)}")

    probs = result["probs"]
    bar = go.Figure(go.Bar(
        x=list(probs.values()), y=list(probs.keys()), orientation="h",
        marker_color=["#4CAF3D", "#E12A26"][: len(probs)],
        text=[f"{v:.0%}" for v in probs.values()], textposition="inside",
    ))
    bar.update_layout(height=140, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 1], tickformat=".0%"))
    st.plotly_chart(bar, width="stretch")

    img_col, cam_col = st.columns(2)
    cam = None
    with img_col:
        st.image(image_bytes, caption="Selected slice", width="stretch")
    with cam_col:
        try:
            with st.spinner("Computing Grad-CAM heatmap..."):
                cam = _cached_gradcam_2d(image_bytes)
            st.image(cam["overlay"], caption="Grad-CAM: regions driving the prediction", width="stretch")
        except Exception:
            st.info("Heatmap unavailable for this image.")

    st.session_state.mri_2d_result = {"result": result, "cam": (cam or {}).get("cam")}

    with st.expander("What do these categories mean?"):
        st.markdown(
            "- **Non Demented** — no clinical signs of cognitive decline.\n"
            "- **Demented** — MRI features consistent with dementia (mild or moderate); "
            "this model confirms presence, not severity."
        )

    st.subheader("Radiology report summary")
    with st.container(border=True):
        with st.spinner("Generating explanation..."):
            explanation = explain_mri(result, image_bytes=image_bytes, cam=(cam or {}).get("cam"))
        st.markdown(f"**IMPRESSION:** {result['label']} ({result['score']:.0%} model confidence)")
        st.markdown("**FINDINGS:**")
        if explanation:
            st.write(explanation)
        else:
            st.caption("LLM explanation unavailable (check LLM provider configuration).")


def _run_mri_3d(volume_path: str, volume_name: str):
    """Runs the 3D MRI model and renders results. Shared by upload/OASIS1_raw picker."""
    result = _cached_predict_3d(volume_path)
    st.session_state.mri_selected = {"name": volume_name, "bytes": None, "result": result}

    st.caption(
        "3D volume (v1): the existing 2D model is applied to central axial "
        "slices and pooled — not a validated volumetric classifier."
    )
    st.metric("Dementia confirmation", result["label"], f"{result['score']:.0%} confidence")
    pending = [
        name for name, done in [("Clinical", st.session_state.clinical_result), ("EEG", st.session_state.eeg_result)]
        if not done
    ]
    st.info(f"**Recommended next steps:** {next_steps(MRI_LEVELS.get(result['label'], 'high'), pending)}")

    probs = result["probs"]
    bar = go.Figure(go.Bar(
        x=list(probs.values()), y=list(probs.keys()), orientation="h",
        marker_color=["#4CAF3D", "#E12A26"][: len(probs)],
        text=[f"{v:.0%}" for v in probs.values()], textposition="inside",
    ))
    bar.update_layout(height=140, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 1], tickformat=".0%"))
    st.plotly_chart(bar, width="stretch")

    img_col, cam_col = st.columns(2)
    cam = None
    with img_col:
        st.image(
            result["slice_array"],
            caption=f"Central axial slice #{result['slice_index']} (most-affected)",
            width="stretch",
        )
    with cam_col:
        try:
            with st.spinner("Computing Grad-CAM heatmap..."):
                cam = _cached_gradcam_3d(volume_path)
            st.image(cam["overlay"], caption="Grad-CAM: representative slice", width="stretch")
        except Exception:
            st.info("Heatmap unavailable for this image.")

    st.session_state.mri_3d_result = {"result": result, "cam": (cam or {}).get("cam")}

    with st.expander("3D volume (rotable)"):
        try:
            with st.spinner("Computing volumetric Grad-CAM..."):
                cam_volume = _cached_gradcam_volume_3d(volume_path)["cam_volume"]
        except Exception:
            cam_volume = None
            st.info("Volumetric heatmap unavailable for this image.")
        with st.spinner("Rendering 3D volume..."):
            st.plotly_chart(_cached_volume_figure(volume_path, cam_volume=cam_volume), width="stretch")

    with st.expander("What do these categories mean?"):
        st.markdown(
            "- **Non Demented** — no clinical signs of cognitive decline.\n"
            "- **Demented** — MRI features consistent with dementia (mild or moderate); "
            "this model confirms presence, not severity."
        )

    st.subheader("Radiology report summary")
    with st.container(border=True):
        with st.spinner("Generating explanation..."):
            explanation = explain_mri(result, image_bytes=None, cam=(cam or {}).get("cam"))
        st.markdown(f"**IMPRESSION:** {result['label']} ({result['score']:.0%} model confidence)")
        st.markdown("**FINDINGS:**")
        if explanation:
            st.write(explanation)
        else:
            st.caption("LLM explanation unavailable (check LLM provider configuration).")


def _run_mri_combined():
    """Fuses the last 2D and 3D results (each stashed by _run_mri_2d/_run_mri_3d)
    into one combined score, classification, and radiology report."""
    from alz.explain import explain_mri_combined
    from alz.fusion import combine_mri

    r2d = st.session_state.mri_2d_result
    r3d = st.session_state.mri_3d_result
    if not r2d or not r3d:
        st.info("Run both the 2D slice and 3D volume analyses first; this tab combines their results.")
        return

    combined = combine_mri(r2d["result"], r3d["result"])
    st.session_state.mri_selected = {"name": "2D + 3D combined", "bytes": None, "result": combined}

    st.metric("Combined dementia confirmation", combined["label"], f"{combined['score']:.0%} confidence")
    pending = [
        name for name, done in [("Clinical", st.session_state.clinical_result), ("EEG", st.session_state.eeg_result)]
        if not done
    ]
    st.info(f"**Recommended next steps:** {next_steps(MRI_LEVELS.get(combined['label'], 'high'), pending)}")

    agree = r2d["result"]["label"] == r3d["result"]["label"]
    st.markdown(
        f"2D and 3D {'agree' if agree else 'disagree'} on the predicted label: "
        f"2D → {chip(r2d['result']['label'], MRI_LEVELS.get(r2d['result']['label'], 'high'))}, "
        f"3D → {chip(r3d['result']['label'], MRI_LEVELS.get(r3d['result']['label'], 'high'))}",
        unsafe_allow_html=True,
    )

    p2d = 1 - r2d["result"]["probs"]["Non Demented"]
    p3d = 1 - r3d["result"]["probs"]["Non Demented"]
    p_combined = combined["probs"]["Demented"]
    bar = go.Figure(go.Bar(
        x=[p2d, p3d, p_combined], y=["2D", "3D", "Combined"], orientation="h",
        marker_color=["#636EFA", "#636EFA", "#4C8BF5"],
        text=[f"{v:.0%}" for v in [p2d, p3d, p_combined]], textposition="inside",
    ))
    bar.update_layout(
        height=180, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 1], tickformat=".0%", title="P(Demented)"),
    )
    st.plotly_chart(bar, width="stretch")
    st.caption("Bars show each pathway's independent P(Demented); they don't have to sum to 100%.")

    st.subheader("Radiology report summary")
    with st.container(border=True):
        with st.spinner("Generating explanation..."):
            explanation = explain_mri_combined(
                r2d["result"], r3d["result"], combined, cam_2d=r2d["cam"], cam_3d=r3d["cam"]
            )
        st.markdown(f"**IMPRESSION:** {combined['label']} ({combined['score']:.0%} model confidence)")
        st.markdown("**FINDINGS:**")
        if explanation:
            st.write(explanation)
        else:
            st.caption("LLM explanation unavailable (check LLM provider configuration).")


with tab_mri:
    try:
        from alz import imaging as _imaging_check  # noqa: F401
    except ImportError:
        st.warning("MRI dependencies not installed — see requirements-imaging.txt")
    else:
        sub2d, sub3d, subcombined = st.tabs(["2D slice", "3D volume", "2D + 3D combined"])

        with sub2d:
            with st.container(border=True):
                uploaded = st.file_uploader("Upload a 2D slice", type=["jpg", "jpeg", "png"], key="mri_2d_uploader")
                examples = _mri_2d_examples()
                example_names = [e["name"] for e in examples]
                chosen_example = st.selectbox("Or select an example slice", example_names, key="mri_2d_example") if example_names else None
                if not example_names:
                    st.caption("No example slices available (imagesoasis dataset not downloaded).")

            if uploaded is not None:
                _run_mri_2d(uploaded.getvalue(), uploaded.name)
            elif chosen_example is not None:
                path = next(e["path"] for e in examples if e["name"] == chosen_example)
                with open(path, "rb") as f:
                    _run_mri_2d(f.read(), chosen_example)
            else:
                st.caption("Upload a slice or select an example to begin.")

        with sub3d:
            with st.container(border=True):
                uploaded_files = st.file_uploader(
                    "Upload a volume (.nii, .nii.gz, or a matching .hdr + .img pair)",
                    type=["nii", "gz", "hdr", "img"],
                    accept_multiple_files=True,
                    key="mri_3d_uploader",
                )
                subjects = _oasis1_subjects()
                subject_names = [s["name"] for s in subjects]
                chosen_subject = st.selectbox("Or select an OASIS1_raw subject", subject_names, key="mri_3d_subject") if subject_names else None
                if not subject_names:
                    st.caption("No OASIS1_raw subjects found on disk.")

            if uploaded_files:
                import tempfile

                names_lower = [f.name.lower() for f in uploaded_files]
                if any(n.endswith((".hdr", ".img")) for n in names_lower):
                    tmpdir = tempfile.mkdtemp()
                    hdr_path = None
                    for f in uploaded_files:
                        dest = os.path.join(tmpdir, f.name)
                        with open(dest, "wb") as out:
                            out.write(f.getvalue())
                        if f.name.lower().endswith(".hdr"):
                            hdr_path = dest
                    if hdr_path is None:
                        st.warning("Upload both the .hdr and .img files together.")
                    else:
                        _run_mri_3d(hdr_path, uploaded_files[0].name)
                else:
                    f = uploaded_files[0]
                    suffix = ".nii.gz" if f.name.lower().endswith(".nii.gz") else ".nii"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(f.getvalue())
                        tmp_path = tmp.name
                    _run_mri_3d(tmp_path, f.name)
            elif chosen_subject is not None:
                path = next(s["path"] for s in subjects if s["name"] == chosen_subject)
                _run_mri_3d(path, chosen_subject)
            else:
                st.caption("Upload a volume or select a subject to begin.")

        with subcombined:
            _run_mri_combined()

# ---------------------------------------------------------------------------
# Tab 4: EEG records
# ---------------------------------------------------------------------------
EEG_EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ds004504", "derivatives")
EEG_PARTICIPANTS_TSV = os.path.join(os.path.dirname(__file__), "..", "data", "ds004504", "participants.tsv")


@st.cache_data
def _eeg_examples() -> list[dict]:
    """A handful of real ds004504 subjects (2 AD + 2 healthy) for the demo dropdown."""
    import pandas as pd

    df = pd.read_csv(EEG_PARTICIPANTS_TSV, sep="\t")
    examples = []
    for group, group_label, n in [("A", "AD", 2), ("C", "CN", 2)]:
        for _, row in df[df["Group"] == group].head(n).iterrows():
            sub_id = row["participant_id"]
            set_path = os.path.join(EEG_EXAMPLES_DIR, sub_id, "eeg", f"{sub_id}_task-eyesclosed_eeg.set")
            if os.path.exists(set_path) and os.path.getsize(set_path) > 1000:
                examples.append({
                    "name": f"{sub_id} — {group_label} (MMSE {row['MMSE']})",
                    "path": set_path,
                })
    return examples


with tab_eeg:
    with st.container(border=True):
        uploaded_eeg = st.file_uploader("Upload EEG recording", type=["set"])
        examples = _eeg_examples()
        example_names = [e["name"] for e in examples]
        chosen_example = st.selectbox("Or select a prior recording", example_names) if example_names else None
        if not example_names:
            st.caption("No example recordings available (dataset not downloaded).")
        run = st.button("Run evaluation")

    if run:
        try:
            from alz.eeg import eeg_explanation, load_recording, predict_eeg_probs
        except ImportError:
            st.warning("EEG dependencies not installed — see requirements-eeg.txt")
        else:
            if uploaded_eeg is not None:
                recording_source, recording_name = uploaded_eeg, uploaded_eeg.name
            elif chosen_example is not None:
                recording_source = next(e["path"] for e in examples if e["name"] == chosen_example)
                recording_name = chosen_example
            else:
                recording_source = None

            if recording_source is None:
                st.warning("Upload a recording or select an example first.")
            else:
                with st.spinner("Extracting band-power features and scoring..."):
                    raw = load_recording(recording_source)
                    result = predict_eeg_probs(raw)
                    explanation = eeg_explanation(raw)
                level = band_for_score(result["score"])[1]
                st.session_state.eeg_result = {
                    **result, "level": level, "recording": recording_name, "drivers": explanation["contributions"],
                }
                st.session_state.eeg_explain = explanation
                st.session_state.eeg_raw_signal = raw.get_data()[:6, : int(raw.info["sfreq"] * 4)]
                st.session_state.eeg_raw_sfreq = raw.info["sfreq"]
                st.session_state.eeg_raw_channels = raw.info["ch_names"][:6]
                st.rerun()

    if st.session_state.eeg_result:
        eeg = st.session_state.eeg_result
        label, level, score = eeg["label"], eeg["level"], eeg["score"]

        if st.session_state.get("eeg_raw_signal") is not None:
            signal = st.session_state.eeg_raw_signal
            sfreq = st.session_state.eeg_raw_sfreq
            t = np.arange(signal.shape[1]) / sfreq
            fig = go.Figure()
            for i, ch in enumerate(st.session_state.eeg_raw_channels):
                fig.add_trace(go.Scatter(x=t, y=signal[i] * 1e6 + i * 100, mode="lines", name=ch, line=dict(width=1)))
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(showticklabels=False), xaxis_title="Time (s)")
            st.subheader("Signal viewer")
            st.plotly_chart(fig, width="stretch")

        st.subheader("Evaluation results")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(chip(label, level), unsafe_allow_html=True)
        with m2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": SEVERITY_COLORS[level][0]}},
                title={"text": "P(Alzheimer's pattern) (%)"},
            ))
            gauge.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(gauge, width="stretch")

        explanation = st.session_state.get("eeg_explain")
        if explanation:
            from alz.eeg import BANDS

            st.subheader("Why this score")
            c1, c2 = st.columns(2)

            with c1:
                st.caption("Band contribution to score (hover a bar for details)")
                drivers = explanation["contributions"]
                raw_names = [d["feature"] for d in drivers][::-1]
                display = [eeg_feature_display(n) for n in raw_names]
                names = [d[0] for d in display]
                tooltips = [d[1] for d in display]
                values = [d["contribution"] for d in drivers][::-1]
                colors = [SEVERITY_COLORS["high"][0] if v > 0 else SEVERITY_COLORS["normal"][0] for v in values]
                bar = go.Figure(go.Bar(
                    x=values, y=names, orientation="h", marker_color=colors,
                    customdata=tooltips,
                    hovertemplate="<b>%{y}</b><br>%{customdata}<br>Contribution: %{x:.3f}<extra></extra>",
                ))
                bar.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Contribution to log-odds")
                st.plotly_chart(bar, width="stretch")

            with c2:
                st.caption("Relative band power vs cohort")
                band_names = [name.title() for name, _, _ in BANDS]
                spectrum = go.Figure()
                spectrum.add_trace(go.Bar(x=band_names, y=explanation["relative_bands"], name="This patient"))
                cohort = explanation["cohort_bands"]
                if cohort:
                    spectrum.add_trace(go.Bar(x=band_names, y=cohort["healthy_bands"], name="Healthy mean"))
                    spectrum.add_trace(go.Bar(x=band_names, y=cohort["ad_bands"], name="AD mean"))
                spectrum.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), barmode="group", yaxis_title="Relative power")
                st.plotly_chart(spectrum, width="stretch")

            st.caption(
                "Scalp topography: redder regions have more slow-wave (delta + theta) power, "
                "the 'EEG slowing' pattern associated with Alzheimer's. See scale at right."
            )
            try:
                import matplotlib.pyplot as plt
                import mne

                slowing = explanation["per_channel_bands"][:, 0] + explanation["per_channel_bands"][:, 1]
                info = mne.create_info(explanation["ch_names"], sfreq=1.0, ch_types="eeg")
                info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
                fig, (ax, cax) = plt.subplots(1, 2, figsize=(4, 3), gridspec_kw={"width_ratios": [1, 0.06]})
                im, _ = mne.viz.plot_topomap(slowing, info, axes=ax, show=False, cmap="Reds", contours=4)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label("Slow-wave power (a.u.)", color="white")
                cbar.ax.yaxis.set_tick_params(color="white")
                plt.setp(cbar.ax.get_yticklabels(), color="white")
                fig.patch.set_alpha(0)
                ax.set_facecolor("none")
                st.pyplot(fig, width="content")
            except Exception:
                st.caption("Topomap unavailable for this recording's channel layout.")

            with st.spinner("Drafting clinical readout..."):
                readout = explain_eeg(eeg, explanation["contributions"])
            if readout:
                st.info(readout)
