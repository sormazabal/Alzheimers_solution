"""Streamlit demo: enter a patient record, see risk score + drivers via alz.predict()."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import plotly.graph_objects as go
import streamlit as st

from alz import predict
from alz.data import FEATURE_META, load_population

st.set_page_config(page_title="Alzheimer's Early-Risk Triage", page_icon="🧠")
st.title("Alzheimer's Early-Risk Triage")


tab_clinical, tab_mri = st.tabs(["Clinical risk", "MRI severity"])

with tab_clinical:
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=110, value=75)
            educ = st.number_input("Years of education", min_value=0, max_value=30, value=12)
            ses = st.number_input("Socioeconomic status (1=highest, 5=lowest)", min_value=1, max_value=5, value=3)
            sex = st.selectbox("Sex", ["F", "M"])
        with col2:
            mmse = st.number_input("MMSE score (0-30)", min_value=0, max_value=30, value=27)
            nwbv = st.number_input("Normalized whole-brain volume", min_value=0.5, max_value=0.95, value=0.75, format="%.3f")
            asf = st.number_input("Atlas scaling factor", min_value=0.7, max_value=1.7, value=1.2, format="%.3f")
            visit = st.number_input("Visit number", min_value=1, max_value=10, value=1)

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
            st.markdown(
                f'<span title="{tooltip}" style="text-decoration: underline dotted; cursor: help;">'
                f'<b>{label}</b></span> — {driver["direction"]} '
                f'(this patient: {driver["value"]:.2g}, {driver["percentile"]:.0f}th percentile)',
                unsafe_allow_html=True,
            )

            fig = go.Figure()
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
            from alz.imaging import predict_mri_probs
        except ImportError:
            st.warning("MRI dependencies not installed — see requirements-imaging.txt")
        else:
            result = predict_mri_probs(uploaded)
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
