"""Streamlit demo: enter a patient record, see risk score + drivers via alz.predict()."""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import plotly.graph_objects as go
import streamlit as st

from alz import predict
from alz.data import FEATURE_META, load_population
from alz.explain import explain_mri

st.set_page_config(page_title="Alzheimer's Early-Risk Triage", page_icon="🧠")
st.title("Alzheimer's Early-Risk Triage")


tab_clinical, tab_mri = st.tabs(["Clinical risk", "MRI severity"])

with tab_clinical:
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=110, value=75, help=FEATURE_META["Age"]["definition"])
            educ = st.number_input("Years of education", min_value=0, max_value=30, value=12, help=FEATURE_META["EDUC"]["definition"])
            ses = st.number_input("Socioeconomic status (1=highest, 5=lowest)", min_value=1, max_value=5, value=3, help=FEATURE_META["SES"]["definition"])
            sex = st.selectbox("Sex", ["F", "M"], help=FEATURE_META["Sex_M"]["definition"])
        with col2:
            mmse = st.number_input(
                "MMSE score (0-30)", min_value=0, max_value=30, value=27,
                help="Mini-Mental State Exam: a bedside 30-point cognitive screening test "
                     "(memory, attention, language, orientation). Lower scores mean more impairment; "
                     "27-30 is normal, <24 typically flags concern.",
            )
            nwbv = st.number_input("Normalized whole-brain volume", min_value=0.5, max_value=0.95, value=0.75, format="%.3f", help=FEATURE_META["nWBV"]["definition"])
            asf = st.number_input(
                "Atlas scaling factor", min_value=0.7, max_value=1.7, value=1.2, format="%.3f",
                help="A head-size correction factor used to normalize MRI volume measurements "
                     "(like nWBV) so brains of different physical sizes can be compared fairly. "
                     "It reflects skull/head size, not disease severity.",
            )
            visit = st.number_input("Visit number", min_value=1, max_value=10, value=1, help=FEATURE_META["Visit"]["definition"])

        submitted = st.form_submit_button("Assess risk")

    if submitted:
        record = {
            "Age": age, "EDUC": educ, "SES": ses, "MMSE": mmse,
            "nWBV": nwbv, "ASF": asf, "Visit": visit, "MR Delay": 0,
            "M/F": sex,
        }
        result = predict(record)

        st.metric("Risk score", f"{result['score']:.0%}", result["label"])
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

            fig = go.Figure()
            if is_sex:
                # ponytail: binary percentile is meaningless — show Age split by sex instead
                fig.add_trace(go.Histogram(x=population.loc[population["Sex_M"] == 1, "Age"], name="Male", marker_color="#636EFA", opacity=0.6))
                fig.add_trace(go.Histogram(x=population.loc[population["Sex_M"] == 0, "Age"], name="Female", marker_color="#EF553B", opacity=0.6))
                fig.update_layout(barmode="overlay")
                fig.add_vline(
                    x=record["Age"], line_color="crimson", line_width=2,
                    annotation_text="This patient", annotation_position="top",
                )
                fig.update_layout(
                    height=220, margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Age", showlegend=True,
                )
            else:
                fig.add_trace(go.Histogram(x=population[driver["feature"]], name="Population", marker_color="#636EFA"))
                fig.add_vline(
                    x=driver["value"], line_color="crimson", line_width=2,
                    annotation_text="This patient", annotation_position="top",
                )
                fig.update_layout(
                    height=220, margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title=label, showlegend=False,
                )
            st.plotly_chart(fig, use_container_width=True)

with tab_mri:
    uploaded = st.file_uploader("Brain scan", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.image(uploaded, caption="Uploaded scan", width=300)
        try:
            from alz.imaging import gradcam_mri, predict_mri_probs
        except ImportError:
            st.warning("MRI dependencies not installed — see requirements-imaging.txt")
        else:
            uploaded_bytes = uploaded.getvalue()
            result = predict_mri_probs(io.BytesIO(uploaded_bytes))
            st.metric("Predicted severity", result["label"], f"{result['score']:.0%} confidence")

            probs = result["probs"]
            fig = go.Figure(go.Pie(
                labels=list(probs.keys()), values=list(probs.values()),
                textinfo="percent", textposition="inside", sort=False,
            ))
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("What do these categories mean?"):
                st.markdown(
                    "- **Non Demented** — no clinical signs of cognitive decline.\n"
                    "- **Mild Dementia** — early, noticeable memory/cognitive changes that still allow "
                    "mostly independent daily living.\n"
                    "- **Moderate Dementia** — more pronounced cognitive and functional decline, "
                    "typically requiring regular assistance with daily activities."
                )

            st.subheader("Where the model is looking")
            try:
                with st.spinner("Computing Grad-CAM heatmap..."):
                    cam = gradcam_mri(io.BytesIO(uploaded_bytes))
                st.image(cam["overlay"], caption="Regions driving the prediction (Grad-CAM)", width=300)
            except Exception:
                st.info("Heatmap unavailable for this image.")

            st.subheader("Why this assessment")
            with st.spinner("Generating explanation..."):
                explanation = explain_mri(result)
            if explanation:
                st.write(explanation)
            else:
                st.caption("LLM explanation unavailable (check LLM provider configuration).")
