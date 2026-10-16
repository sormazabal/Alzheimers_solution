# Q&A: How the Models and Functions Work

---

## 1. Basics

**Q: What does the system do?**
A: It takes a patient's clinical data, an MRI scan and an EEG recording. Each goes to its own model. The three risk scores are fused into one integrated prognosis. A language model then writes plain-language explanations of the results.

**Q: Is it all AI?**
A: Three layers, three different answers:
- **Risk scores:** small, classic machine learning (logistic regression, one ResNet18). No language model is involved.
- **Fusion:** a fixed formula. Nothing is trained.
- **Explanations:** a language model (LLM). It only writes text. It never changes a score.

**Q: Does it diagnose?**
A: No. It outputs risk estimates for a clinician to review. The integrated score is a probability of dementia, shown as a tier (Low, Intermediate, High), not a diagnosis or a severity stage.

**Q: How do other systems call it?**
A: One function: `from alz import predict`. It takes a patient record and returns a score, a label and the top drivers. A small FastAPI wrapper (`/health`, `/predict`) exists, but covers the clinical model only.

---

## 2. Clinical risk model

**Q: What is the model?**
A: A scikit-learn pipeline: fill missing values with the mean, standardize, then logistic regression. Output is P(at-risk) and a label (Normal or At-risk).

**Q: Why logistic regression and not something stronger?**
A: Its coefficients give a direct, honest explanation of each prediction. The reference notebook used a random forest. We swapped it for this reason.

**Q: What does it read?**
A: Nine inputs: visit number, MRI delay, age, years of education, socioeconomic status, MMSE, whole-brain volume (nWBV), atlas scaling factor (ASF) and sex.

**Q: Why is CDR left out?**
A: CDR (Clinical Dementia Rating) is itself the diagnosis. Using it would leak the answer into a tool meant to screen before diagnosis.

**Q: How are the "top drivers" computed?**
A: Each standardized input is multiplied by its model coefficient. The three largest contributions are shown, each marked "raises risk" or "lowers risk", with the patient's percentile against the training population. This is exact for a linear model. No SHAP is needed.

**Q: How was it trained and tested?**
A: 5-fold cross-validation on 82 rows. Reported metrics come from out-of-fold predictions. The final model is then refit on all rows. The dataset is a synthetic OASIS-shaped sample, so the numbers are indicative only.

---

## 3. MRI model

**Q: What is the model?**
A: A ResNet18 pretrained on ImageNet. The backbone is frozen, and only a new final layer is trained. It classifies Non Demented vs Demented and outputs a probability for each.

**Q: Why transfer learning?**
A: The data is about 94 subjects, which is too few to train a deep network from scratch. A frozen backbone limits overfitting.

**Q: How was it trained?**
A: 3 epochs by default, Adam optimizer, class-weighted loss because Demented subjects are rare. The split is 70/15/15 **by subject**, so slices from one patient never land in both train and test. An earlier per-slice split leaked patients and gave a meaningless ~1.00 accuracy.

**Q: What is the "3D" pathway?**
A: It is not a true 3D network. It takes the 16 central axial slices of a NIfTI volume, runs the same 2D model on each and averages the probabilities. Treat it as a v1 approximation. The model learned JPEG slices, not raw scanner intensities, so a domain gap remains.

**Q: How are 2D and 3D combined?**
A: Each pathway gives P(Demented). The two probabilities are converted to log-odds, averaged with equal weight, and converted back. Opposite results cancel to about 50%.

**Q: Why log-odds and not a plain average of probabilities?**
A: Log-odds treat each result as a unit of evidence. A confident result (for example 95%) pulls the answer harder than a near-50% one, which adds almost nothing. A plain average treats 95% and 60% as only a linear gap. Log-odds also keep the result between 0 and 1. A symmetric pair such as 90% and 10% gives exactly 50%.

**Q: Why equal weights?**
A: No data exists to say that 2D or 3D is more reliable, so any other weight would be a guess. Both pathways also use the same model (see below), so they have no known difference in quality.

**Q: Are the 2D and 3D models both ResNet?**
A: Yes, and they are the same model. There is one ResNet18 with one set of weights. The 2D pathway runs it on one image. The "3D" pathway runs it on 16 central axial slices of a NIfTI volume and averages the results. No 3D network (such as a 3D ResNet) is used. Because both pathways share the model, they are not fully independent, and averaging them reduces slice-level noise more than it adds new evidence.

**Q: What is the attention map?**
A: Grad-CAM on the last convolutional block. It highlights the image regions that pushed the model toward its answer. For 3D, one map is made for the most confident slice, plus an optional full-volume map overlaid on a rotatable 3D render.

**Q: Does the map show anatomy?**
A: No. A short text summary describes where the **model** looked (superior or inferior, central or peripheral, left or right, focal or diffuse). No anatomical atlas is used.

**Q: What hardware does it run on?**
A: Picks automatically: NVIDIA GPU, Apple Silicon (MPS), DirectML, then CPU.

---

## 4. EEG model

**Q: What is the model?**
A: Logistic regression on 7 numbers per recording. It outputs P("Alzheimer's pattern").

**Q: What are the 7 numbers?**
A: Relative power in five bands (delta, theta, alpha, beta, gamma), averaged over all channels, plus two ratios: theta/alpha and (delta+theta)/(alpha+beta). Power comes from Welch's method. This captures the classic "EEG slowing" in Alzheimer's: more slow-wave power, less fast-wave power.

**Q: Why not a deep learning model?**
A: Only 65 usable subjects (Alzheimer's vs healthy; frontotemporal cases are dropped). That is too few to adapt a deep model honestly. Band power is a well-established, explainable signal that suits a small model.

**Q: What input does it need?**
A: A resting-state, eyes-closed EEGLAB `.set` file, read with MNE.

**Q: How is it explained?**
A: Same method as the clinical model (standardized feature times coefficient). The panel also shows the patient's spectrum against the healthy and Alzheimer's cohort averages, and a scalp map of per-channel band power.

**Q: How was it tested?**
A: 5-fold cross-validation with balanced class weights. Metrics are out-of-fold.

---

## 5. Fusion (integrated prognosis)

**Q: How are the three scores combined?**
A: A **logarithmic opinion pool**: a weighted average in log-odds space, converted back to a probability. Each modality contributes P(dementia): clinical score, 1 minus P(Non Demented) for MRI, and the EEG score.

**Q: Why not a trained fusion model?**
A: No dataset pairs clinical, MRI and EEG data with one shared diagnosis, so a fusion model cannot be trained or validated. The pool needs no training data.

**Q: Does one modality dominate?**
A: Only through confidence. A score near 50% adds almost nothing to the pooled result. A score near 0% or 100% pulls it strongly. No hand-set weights are used.

**Q: Can weights be changed?**
A: Yes. A per-modality reliability weight exists and defaults to 1.0 for all. It is deliberately untuned. Change it only once validation data shows one modality is more reliable.

**Q: What if a modality is missing?**
A: It is skipped. With only one modality, the fused score equals that modality's own score.

**Q: What if the modalities disagree?**
A: Averaging hides conflict, so the app shows the spread (highest minus lowest probability) and warns when it exceeds 0.4.

**Q: Where do the tiers come from?**
A: Low below 0.33, Intermediate to 0.66, High above. Both the tier cutoffs and the 0.4 warning are display heuristics, not tuned on data.

**Q: Known weaknesses?**
A: It assumes each input is well calibrated, and small fine-tuned models can be overconfident. It also assumes the modalities are independent, but they reflect the same disease. A trained fusion model is the long-term fix.

---

## 6. LLM explanations

**Q: What do the LLM features do?**
A: Five things:
- Radiology-style MRI findings (2D, and a combined 2D+3D version).
- A short EEG readout.
- A draft clinical note across all modalities (about 90 words).
- A chat box for questions about the case.
- Evidence retrieval: live PubMed guidelines and recruiting ClinicalTrials.gov trials, plus a recommendation that may cite only those sources.

**Q: Which model is used?**
A: Configurable. The provider is picked from environment settings (Ollama, Gemini, OpenAI, Groq, AWS Bedrock, or Anthropic as the fallback). All sit behind one `complete()` interface, so swapping is a settings change. MRI image reading uses the vision model `gemma-3-27b-it` through HuggingFace.

**Q: Can the LLM change a score?**
A: No. It receives the finished numbers and writes text about them.

**Q: What does the LLM see?**
A: A text digest of the case: sex, date of birth, history, each model's label, score and top drivers, and the fused score. The vision call also receives the scan image.

**Q: How is hallucination limited?**
A: Prompts say to use only the supplied facts and never invent timeframes, percentages or outcomes. Evidence recommendations may cite only the sources that were actually retrieved. Replies are requested as JSON and checked. This reduces errors, but does not remove them.

**Q: What if the LLM is down or no key is set?**
A: Each panel returns nothing and the app carries on. The scores never depend on it. Evidence retrieval degrades part by part: PubMed, trials and recommendation fail independently.



---

## 7. Patient database and cohort tools


**Q: What does the database do?**
A: SQLite stores the links and caches every model result. A result is reused while its input is unchanged, so each patient is scored once. One failing file does not stop a batch.

**Q: What are the cohort tools?**
A: A sortable table and charts across patients. Patients are grouped into 3 phenotype clusters (KMeans on the clinical inputs only, not on model scores). Any patient can be compared by percentile with their cluster or the whole cohort.

---

## 8. Accuracy and limits

**Q: How accurate is it?**
A: Preliminary, on small data (out-of-fold or held-out):

| Model | AUROC | Sensitivity | Specificity | Data |
|---|---|---|---|---|
| Clinical | 0.73 | 0.47 | 0.84 | 82 synthetic rows |
| MRI | 0.91 | 0.70 | 0.91 | about 94 subjects, small subsample |
| EEG | 0.82 | 0.75 | 0.76 | 65 subjects |
| Fusion | not measured | not measured | not measured | no paired ground truth |

**Q: What is the weakest part?**
A: The clinical model's sensitivity, which is below 50% on the current sample. Next are the 3D pathway (an approximation) and the fused score (unvalidated).

**Q: Has it been tested on our patients?**
A: No. Every number comes from public research datasets or synthetic data.

**Q: Does it handle other scanners, populations or ages?**
A: Unknown. Expect a drop until it is validated and retrained on your data.

**Q: Is a human still needed?**
A: Yes. The design is machine draft, clinician decision.

---

## 9. Security and privacy

**Q: Where does data go?**
A: Scoring is fully local. Three things can leave the machine: LLM prompts (with sex, date of birth and history), MRI images sent to the HuggingFace vision model, and search terms sent to PubMed and ClinicalTrials.gov. The search terms are fixed ("Alzheimer Disease"), not patient data. Use a local provider (Ollama) to keep all patient data on site.

**Q: Is it production-secure?**
A: No, this is a prototype. There is no authentication, no audit log and no access control. The API is a stub.

---

## 10. Roadmap and tough questions

| Question | Safe answer | Do not promise |
|---|---|---|
| Can you validate on our data? | Yes. Send a labeled sample set and we will report results and retrain needs. | Performance before testing. |
| Can you make 3D a true volumetric model? | Yes, it is the natural upgrade. It needs full 3D training data. | A date, or a gain in accuracy. |
| Can you train the fusion step? | Yes, once paired multi-modal data with a shared diagnosis exists. | Any improvement over the current pool. |
| Can it run fully on premises? | Scoring already does. Use a local LLM provider for the explanations. | Equal text quality from a small local model. |
| Can you add other modalities (blood biomarkers, PET)? | Each is a new model plus one more input to the pool. | A timeline. |
| What is your accuracy? | Preliminary figures above, with the data caveats. | Any clinical-grade claim. |
| Is it ready for clinical use? | No. It is a prototype. | Regulatory status of any kind. |
