# Gemini AI Diagram Prompt — Slide 6 Architecture

Copy the prompt below as-is into Gemini AI's image generation.

---

Create a clean, professional infographic-style architecture diagram, widescreen 16:9 aspect ratio, flat vector illustration only (no photographic imagery, no 3D rendering), legible when projected at slide size.

**Overall style (match this reference style exactly):** a light-gray horizontal title bar across the top holding a bold black all-caps title. Below it, one horizontal row of five boxed panels of equal size, each a black-outlined rounded rectangle on a white background, with a bold black header at the top of the panel. Inside each panel, stack a small icon (database cylinder, gauge/confidence meter, funnel, mini bar chart, or waveform, matching the reference's icon vocabulary) above short labeled text blocks and simple tables where noted. Thin black arrows connect the panels left to right, showing data flowing through the pipeline. Use a muted, restrained color palette: black outlines and text, white panel backgrounds, and accent fills only in muted blue, gray, and green (for bars, tables, and icons) — no bright or saturated colors. Below the row of panels, add a full-width horizontal footer band (light gray background) containing a one-line traceability note on the left and a row of text-based data-source badges on the right, separated by a vertical divider, mirroring the "Traceability Assurance" footer style of the reference image.

**Title bar text:** "ALZHEIMER'S MULTIMODAL DIAGNOSIS SYSTEM: CLINICAL, MRI & EEG PIPELINE"

**Panel 1 — USER INPUT / INTERFACE**
- Icon: a simple monitor/dashboard icon
- Text blocks: "Streamlit Clinical Workstation — 4 tabs: Clinical Risk, MRI Records, EEG Records, Overview"
- "FastAPI service — POST /predict, GET /health"
- "CLIs: train.py, train_mri.py, evaluate_mri.py"
- Arrow points right into Panel 2, 3, and 4 (fan-out, since input feeds all three models)

**Panel 2 — CLINICAL RISK MODEL**
- Icon: a small gauge/confidence meter
- Text: "9 tabular features → mean imputation → standardization → logistic regression"
- Small horizontal mini bar-chart labeled "top_drivers(): coefficient × value ranking"
- Output label: "score = P(at risk)"

**Panel 3 — MRI SEVERITY MODEL**
- Icon: a simple brain-slice/image icon
- Text: "Brain slice image → ResNet18 (frozen backbone, retrained final layer) → softmax over severity classes"
- Small heatmap-overlay icon labeled "Grad-CAM: pixel attribution for winning class"
- Output label: "score = winning class probability"

**Panel 4 — EEG MODEL**
- Icon: a small waveform/signal icon
- Text: "Resting-state recording → relative band power (delta/theta/alpha/beta/gamma) + 2 slowing ratios → standardized logistic regression"
- Output label: "score = P(AD pattern)"

**Panel 5 — FUSION & OUTPUT**
- Icon: a confidence dial combined with a merge/funnel icon
- Formula block (render as clean typeset math, boxed):
  "logit(P_fused) = ( Σ rᵢ · logit(pᵢ) ) / ( Σ rᵢ )"
  "P_fused = sigmoid(logit(P_fused))"
- Text: "Log-odds pooling — a confident modality (p near 0 or 1) dominates; an uncertain modality (p≈0.5) barely moves the result"
- Output label: "Final: score, label, drivers, AI case summary, live evidence"
- Arrows from Panels 2, 3, and 4 converge into Panel 5

**Footer band (full width, light gray):**
- Left text: "Every score traceable to its source model and input features."
- Right side, text-based data-source badges in a row: "OASIS-2 CSV" · "Kaggle imagesoasis" · "OpenNeuro ds004504" · "PubMed E-utilities" · "ClinicalTrials.gov v2"

Keep all text short and legible, consistent font weight for headers vs. body text, generous white space inside each panel, and consistent panel sizing across the row.
