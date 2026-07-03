### Phase 1: workflow and trust (quick wins)

These features build directly on the existing architecture in `streamlit_app.py` and immediately improve the tool's usability for a clinician.

* [x] **Doctor view overall summary** — DONE: `synthesize_summary` in `src/alz/explain.py` now generates an LLM narrative from clinical/MRI/EEG results, wired into `app/streamlit_app.py`.

* [x] **Improve MRI explainability** — DONE: Grad-CAM overlay (`gradcam_mri`) plus `explain_mri` LLM narrative in plain language, with confidence and uncertainty noted.

* [x] **Doctor Q&A** — DONE: `chat_about_case` in `src/alz/explain.py` grounds replies in the patient/case context (`_case_context`), wired into the chat tab.



---

### Phase 2: core clinical decisions (the Siemens differentiators)

These capabilities represent the unique value proposition of the platform and align with the stated goals in the documentation.

* [x] **Integrated score**: Both `PRESENTATION.md` and `README.md` explicitly note multi-modal fusion (combining tabular and imaging data) as the primary SHS differentiator for the future. Creating a single, unified prognosis metric is a critical milestone that proves the system is greater than the sum of its parts.


* [x] **Evidence-based recommended actions (real-time RAG APIs)** — DONE: `evidence_for_case` in `src/alz/explain.py` queries PubMed and ClinicalTrials.gov live and injects results into the clinical recommendation.
* [x] **Clinical guidelines via PubMed (E-utilities API)** — DONE: implemented as part of `evidence_for_case`, results shown under "Guidelines (PubMed)" in the Overview tab.
* [x] **Trial mapping via ClinicalTrials.gov (v2 JSON API)** — DONE: implemented as part of `evidence_for_case`, recruiting trials shown under "Recruiting trials (ClinicalTrials.gov)".
* [x] **Source provenance** — DONE: PMIDs and NCT IDs are rendered as clickable links (`app/streamlit_app.py` evidence section).



---

### Phase 3: expanding modalities (high effort, strategic)

These features require substantial new data pipelines but will position the tool as a comprehensive suite.

* [x]**Incorporate EEG**: Replaced the random generator placeholder in `streamlit_app.py` with a real classifier (`src/alz/eeg.py`) trained on the ds004504 dataset -- band-power features (neurokit2) into a logistic regression, mirroring the tabular model.



---

### System and architecture additions to consider

Here are a few technical capabilities that would make the MVP more robust for a Siemens integration.

* **Real EHR integration**: The `streamlit_app.py` notes that patient context currently lives in session state and flags the need for a real fetch when a patient store exists. Building a FHIR-compliant connector would allow the tool to automatically pull patient demographics and cognitive scores directly from hospital systems.


* **Staging severity refinement**: `README.md` references an external repository for CDR-based severity bucketing. While excluded from the current binary-risk MVP to avoid leakage, incorporating this logic could be highly relevant if staging disease progression becomes a formal requirement.



### Proposed execution roadmap

| Priority | Feature | Category | Effort | Status |
| --- | --- | --- | --- | --- |
| 1 | Doctor view overall summary | Workflow | Low | Done |
| 2 | Improve MRI explainability | Trust | Medium | Done |
| 3 | Integrated multi-modal score | Core strategy | Medium | Not started |
| 4 | Doctor Q&A | Workflow | Medium | Done |
| 5 | Real-time RAG (PubMed/ClinicalTrials APIs) | Decision support | High | Done |
| 6 | Incorporate EEG | Advanced modality | High | Not started |
