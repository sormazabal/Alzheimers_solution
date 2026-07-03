### Phase 1: workflow and trust (quick wins)

These features build directly on the existing architecture in `streamlit_app.py` and immediately improve the tool's usability for a clinician.

* **Doctor view overall summary**: The `streamlit_app.py` file already has a placeholder for this using the `synthesize_summary` function. Upgrading this to a robust, LLM-generated synthesis of all available data points provides immediate triage value, turning raw scores into a clinical narrative.


* **Improve MRI explainability**: The current Grad-CAM implementation in `streamlit_app.py` is an excellent start. Enhancing the radiological report to explicitly describe physical characteristics (such as ventricular enlargement) and explain the confidence score in plain text will help clinicians weigh the AI's opinion appropriately and build trust.


* **Doctor Q&A**: Expanding the existing chat interface (`chat_about_case`) into a fully robust query system allows clinicians to interrogate the patient's history. This requires solid grounding in the patient record to avoid hallucinations.



---

### Phase 2: core clinical decisions (the Siemens differentiators)

These capabilities represent the unique value proposition of the platform and align with the stated goals in the documentation.

* **Integrated score**: Both `PRESENTATION.md` and `README.md` explicitly note multi-modal fusion (combining tabular and imaging data) as the primary SHS differentiator for the future. Creating a single, unified prognosis metric is a critical milestone that proves the system is greater than the sum of its parts.


* **Evidence-based recommended actions (real-time RAG APIs)**: To eliminate infrastructure overhead and ensure the most up-to-date information, the tool will query live external databases via REST APIs, parsing and injecting the results into the clinical recommendation engine.
* **Clinical guidelines via PubMed (E-utilities API)**: Based on the patient's demographics and calculated risk level, the backend will programmatically construct queries to the PubMed ESearch API. The tool will pull the highest-ranking, recent clinical guidelines using the EFetch utility (handling the native XML/JSON conversion) to match the clinical profile.
* **Trial mapping via ClinicalTrials.gov (v2 JSON API)**: Using the modernized v2 API endpoint (`[https://clinicaltrials.gov/api/v2/studies](https://clinicaltrials.gov/api/v2/studies)`), the system will programmatically query conditions and interventions. For example, a patient flagged with high risk could trigger a targeted call filtering for `query.cond=Alzheimer Disease` and `filter.overallStatus=RECRUITING` to match open trials. The pipeline will extract the deeply nested `protocolSection` fields (like eligibility criteria and trial sites) to present clean, actionable recruitment opportunities.
* **Source provenance**: To guarantee clinical auditability, the RAG output must explicitly map its recommendations back to the source IDs. The interface will display clickable PMIDs (PubMed IDs) and NCT IDs (ClinicalTrials.gov unique identifiers) alongside the text so the physician can instantly verify the source.



---

### Phase 3: expanding modalities (high effort, strategic)

These features require substantial new data pipelines but will position the tool as a comprehensive suite.

* **Incorporate EEG**: Replacing the current random generator placeholder in `streamlit_app.py` with a genuine EEG pipeline introduces a highly valuable diagnostic signal. This requires a new data acquisition strategy and a dedicated time-series model, making it a heavier technical lift.



---

### System and architecture additions to consider

Here are a few technical capabilities that would make the MVP more robust for a Siemens integration.

* **Real EHR integration**: The `streamlit_app.py` notes that patient context currently lives in session state and flags the need for a real fetch when a patient store exists. Building a FHIR-compliant connector would allow the tool to automatically pull patient demographics and cognitive scores directly from hospital systems.


* **Staging severity refinement**: `README.md` references an external repository for CDR-based severity bucketing. While excluded from the current binary-risk MVP to avoid leakage, incorporating this logic could be highly relevant if staging disease progression becomes a formal requirement.



### Proposed execution roadmap

| Priority | Feature | Category | Effort |
| --- | --- | --- | --- |
| 1 | Doctor view overall summary | Workflow | Low |
| 2 | Improve MRI explainability | Trust | Medium |
| 3 | Integrated multi-modal score | Core strategy | Medium |
| 4 | Doctor Q&A | Workflow | Medium |
| 5 | Real-time RAG (PubMed/ClinicalTrials APIs) | Decision support | High |
| 6 | Incorporate EEG | Advanced modality | High |