"""Streamlit demo: enter a patient record, see risk score + drivers via alz.predict()."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st

from alz import predict

st.set_page_config(page_title="Alzheimer's Early-Risk Triage", page_icon="🧠")
st.title("Alzheimer's Early-Risk Triage — MVP")
st.caption(
    "Proof of capability: screens routine clinical/cognitive data for dementia risk, "
    "before imaging. Phase 2 (MRI confirmation) is scaffolded but not built — see src/alz/imaging.py."
)

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
    for driver in result["drivers"]:
        st.write(f"- **{driver['feature']}** — {driver['direction']}")
