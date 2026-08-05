# 2-Minute Demo Script: Clinical Triage Workflow

A TTS-narration source and shot list for a 2-minute screen recording of the app, from the
perspective of a clinician working through one patient's visit. Audience is a clinician end
user, so the narration talks the way a physician would chart-side: it names the actual risk
scores, the MMSE numbers, and the EEG band findings driving each decision. It still leaves out
validation metrics like AUROC and any mention of model training, that context belongs to the
app's own documentation, not a bedside narration.

This replaces `docs/demo.md` for recording purposes. That file is a longer prose walkthrough for
reading, not a timed script, and three of its details no longer match the shipped app (a seeded
"Jane Doe" patient that only renders when `data/oasis_longitudinal.csv` is missing; a three-class
MRI severity claim when the shipped model is binary Non Demented vs Demented; no guidance on
where the app stalls). Use this file for the recording; use `docs/demo.md` if you want the
longer narrated explanation of why each step matters for Siemens.

Total target runtime: **2:00 to 2:10**. Six recorded takes, cut together in the editor — see
Part 3 for why a single continuous pass will not work.

---

## Part 1 — Prep (all off camera, before you press record)

### Step A — Pick a patient whose four assessments agree

The sidebar's patient dropdown and the cohort database hold 40 synthetic patients, most of them
partially cached. Not every patient tells one consistent story across clinical, MRI, and EEG —
picking one at random risks a scene where the clinical score reads low-risk while the EEG reads
Alzheimer's pattern, which undercuts a clinician-facing narrative.

**Default pick for this script: `OAS2_0007`.** Verified directly against the database:

| Modality | Result |
|---|---|
| Clinical | At-risk, high band |
| MRI, 2D slice | Demented |
| MRI, 3D volume | Demented |
| EEG | Alzheimer's pattern |
| Fused | Demented (positive) |

All four agree, all four are already cached, and the patient's clinical record has `Visit = 3`
with a prior MMSE of 28 dropping to 19 — so the MMSE sparkline in Segment 2 is guaranteed to
appear and to show a real decline, not a flat line.

To verify this is still true, or to pick a different patient, run (PowerShell, from the repo
root, with the project's virtualenv active):

```powershell
$env:PYTHONPATH="src"; uv run python -c "from alz import db; import pandas as pd; pd.set_option('display.width',250); c=db.connect(); print(db.cohort_frame(c).to_string())"
```

Look for a row where `clinical_score`, `mri_2d_score`, `mri_3d_score`, and `eeg_score` are all
present (not blank) and all point the same direction — either all comfortably above 0.5 for a
"demented" story, or all comfortably below for a "reassuring" story. A mixed row is not wrong,
just a different, more nuanced narrative than this script is written for.

### Step B — Know which on-screen selections you can actually make

This matters and is easy to get wrong: **the "MRI records" and "EEG records" pages do not know
which patient is selected in the sidebar.** Their dropdowns are short, fixed example lists,
independent of the sidebar patient entirely:

- MRI "2D slice" dropdown: 9 fixed slices, 3 per class (Non Demented / Mild Dementia /
  Moderate Dementia). It does **not** contain every patient's actual linked slice.
- MRI "3D volume" dropdown: **all 77** `OASIS1_raw` subjects by ID — this one does let you pick
  a specific patient's real linked volume, if you know its ID.
- EEG dropdown: only 4 fixed recordings (2 AD, 2 control). It does **not** contain every
  patient's actual linked recording.

The only way to see a given patient's *actual* linked MRI slice and EEG file, exactly as the
database paired them, is the sidebar's `▶ Run all modalities` button. But that button does not
render the 2D slice image, the 3D rotatable volume, or the EEG signal viewer / "Why this score"
panel at all (see Part 3 for why). So for those three visuals you will manually pick from the
pages' own dropdowns — pick from the **same diagnostic class**, not the literal linked file, to
keep the story consistent:

For `OAS2_0007` specifically (already confirmed against the database):

| Page | What to select |
|---|---|
| MRI "2D slice" | Any **Mild Dementia** example, e.g. `Mild Dementia — OAS1_0028_MR1_mpr-1_100.jpg` |
| MRI "3D volume" | `OAS1_0053_MR1` — this **is** the patient's real linked volume |
| EEG | `sub-001 — AD (MMSE 16)` (or `sub-002`) — same AD group as the real linked recording |

If you swap patients in Step A, look up the new patient's `group_label` from `cohort_frame` and
its real MRI-3D subject ID with:

```powershell
$env:PYTHONPATH="src"; uv run python -c "from alz import db; c=db.connect(); print(c.execute('SELECT modality, source_id FROM recordings WHERE patient_id=?', ('OAS2_0007',)).fetchall())"
```

then pick the matching-class 2D and EEG examples from the two short dropdowns above, and select
the real subject ID in the 3D dropdown.

### Step C — Pre-warm every page, in this exact order

Order matters: `▶ Run all modalities` resets the EEG explainability panel, so it must run
**first** or it will wipe out the manual EEG pre-warm in step 5.

1. Sidebar: select `OAS2_0007` (or your chosen patient), click **`▶ Run all modalities`**. Wait
   for the sidebar chips to fill in.
2. MRI records → **2D slice**: select the Mild Dementia example above, click **Run evaluation**.
   Wait for the Grad-CAM image and the radiology report to finish.
3. MRI records → **3D volume**: select `OAS1_0053_MR1`, click **Run evaluation**. Wait through
   all four spinners — "Computing Grad-CAM heatmap...", "Computing volumetric Grad-CAM...",
   "Rendering 3D volume...", "Generating explanation...".
4. MRI records → **2D + 3D combined**: just open the tab and let it settle.
5. EEG records: select `sub-001 — AD (MMSE 16)`, click **Run evaluation**. Wait for the topomap
   and the clinical readout box to finish.
6. Overview: click **`✨ Generate AI summary`**, wait, then click **`📖 Find evidence & trials`**,
   wait.

After this, every page you will film renders completely on navigation alone — no button press
on camera, no spinner on camera except the one an LLM re-fires on render (see Part 3).

### Step D — Capture setup

Browser at 100% zoom, window sized so the sidebar and the main column both fit without
horizontal scroll. Keep Streamlit's top-right "Running" indicator out of frame where you can.

---

## Part 2 — The script

Each segment below is **one separate recording take**. Start filming only once the page has
fully rendered from Part 1's pre-warm — do not record the page loading.

The **Narration** blocks are plain prose: no stage directions, no brackets, no markdown
emphasis. Copy each one straight into the TTS tool as-is.

---

### Segment 1 — Patient context (sidebar) · ~0:12

**On screen:** Sidebar `Patient` subheader, the `Patient (linked EHR + MRI + EEG)` dropdown
showing the selected ID, the filled `Integrated prognosis:` chip and percentage, and the three
`Latest assessments` chips underneath it, all filled in.

**Narration:**

Here's a patient presenting with early memory concerns, already worked up with a clinical history, an MRI, and an EEG on file. One click runs all three through the model and returns an integrated risk assessment.

---

### Segment 2 — Clinical risk · ~0:21

**On screen:** `Clinical risk` page. The `Risk score` metric and its label, the severity chip,
the blue `Recommended next steps:` box, then scroll down to the `MMSE vs. last visit` metric and
its sparkline (this appears because the patient is on Visit 3), then to `Top contributing
factors`, and open the `Population comparison charts` expander for a beat before moving on.

**Narration:**

Her clinical risk score comes back at eighty percent, in the high risk band. The biggest driver is her Mini-Mental State exam score, which dropped from twenty eight to nineteen since her last visit, a decline large enough to warrant a specialist referral rather than routine follow up. The model lists each contributing factor, so the physician can see exactly what pushed that score, not just take the number on faith.

---

### Segment 3 — MRI, 2D slice · ~0:18

**On screen:** MRI records → "2D slice" tab. The `Dementia confirmation` metric, the two-bar
probability chart, then the paired `Selected slice` and `Grad-CAM: regions driving the
prediction` images — dwell on the Grad-CAM for a moment — then scroll to `Radiology report
summary` and hold on the `IMPRESSION:` / `FINDINGS:` text.

**Narration:**

Her MRI slice classifies as consistent with dementia, with the model around sixty three percent confident. The heatmap marks the specific regions driving that classification, so a radiologist can check the model's read against the actual anatomy instead of trusting a label blind. Below it, a radiology style summary spells out the structural findings in plain text.

---

### Segment 4 — MRI, 3D volume · ~0:22

**On screen:** MRI records → "3D volume" tab. Briefly show the central slice and its Grad-CAM,
then open the **`3D volume (rotable)`** expander and **drag-rotate the translucent brain shell
for several seconds** so the hot-colored activation region inside reads clearly on camera. This
is the strongest single visual in the app — give it real screen time, even in silence.

**Watch for:** the on-screen caption naming this a "v1" pathway is intentionally not read aloud
(see the caveats decision above) but stays visible in frame — don't crop it out.

**Narration:**

One axial slice can only show where atrophy shows up in that single cut. Rotating the full volume lets us see the model's attention across the entire brain, still landing around sixty nine percent confidence on dementia.

---

### Segment 5 — MRI, 2D + 3D combined · ~0:13

**On screen:** MRI records → "2D + 3D combined" tab. The `Combined dementia confirmation`
metric, the agree/disagree sentence (for this patient, both pathways agree), and the three-bar
`P(Demented)` chart comparing 2D, 3D, and Combined.

**Narration:**

The two dimensional and three dimensional reads agree, both landing on dementia, so the combined confirmation carries the same conclusion with less uncertainty than either read alone, one reconciled call instead of two to weigh separately.

---

### Segment 6 — EEG · ~0:28

**On screen:** EEG records page. `Signal viewer` trace, then the chip and the `P(Alzheimer's
pattern) (%)` gauge under `Evaluation results`, then scroll to `Why this score`: hover one bar in
`Band contribution to score` to show its tooltip, pan across `Relative band power vs cohort`,
then hold on the scalp topomap, and finish on the blue LLM readout box underneath it.

**Narration, part one (gauge):**

Now a third, independent modality, her EEG. The model is scoring for the classic Alzheimer's signature, more slow wave activity relative to alpha, and here it comes back positive, around eighty four percent.

**Narration, part two (why this score, after the pan/hover):**

That score breaks down by frequency band, delta, theta, alpha, beta, and gamma power, each compared against a healthy cohort and an Alzheimer's cohort. The scalp map localizes where that slowing is strongest, so instead of one gauge reading, the physician gets a channel by channel picture to review.

---

### Segment 7 — Overview · ~0:14

**On screen:** Overview page. The `Integrated prognosis` card with its chip and percentage, open
`How this was combined` briefly to show the per-modality weights, then the four-column metric
row, then scroll to the pre-filled `Clinical conclusions` note and the `Guidelines (PubMed)` and
`Recruiting trials (ClinicalTrials.gov)` link lists.

**Narration:**

Combining all three modalities, clinical, MRI, and EEG, produces one fused risk score around seventy seven percent, pooling each modality's own probability rather than just averaging labels. The system also drafts a clinical note and pulls relevant guideline citations and open trials, so the visit ends with a referral decision backed by evidence, not a single test result.

---

## Part 3 — Assembly and fallbacks

### Why this is six takes, not one continuous recording

Verified against the code, not assumed:

- **`▶ Run all modalities` does not populate the pages with the strongest visuals.**
  `_load_patient_into_session` (`app/streamlit_app.py:133-165`) sets `cam: None` and
  `eeg_explain = None`, and never sets `mri_2d_confirmed`, `mri_3d_confirmed`, or
  `eeg_raw_signal`. Without the manual re-runs in Step C, the MRI "2D slice" and "3D volume" tabs
  stay on their idle caption, and the EEG page shows only the chip and gauge — no signal viewer,
  no "Why this score" panel, no topomap.
- **No LLM call is cached.** `explain_mri`, `explain_mri_combined`, `explain_eeg`,
  `synthesize_summary`, and `evidence_for_case` (`src/alz/explain.py`) all run live on every page
  render, and Streamlit executes all three MRI sub-tab bodies on every visit to that page — so
  **navigating to MRI records fires three live network calls every time**, and EEG fires one.
  Pre-warming removes the wait for the numbers and images; it cannot remove this. Expect a
  several-second pause on your recording the first time you land on MRI records or EEG in this
  take, even after Step C.

Because of the second point, record one take per segment above, starting the take a beat after
you land on the page, and cut together in the editor rather than attempting one unbroken pass.

### If a segment runs long

Trim words inside that segment's Narration block, not whole beats — every segment above was
approved as required content. Segment 4 (3D volume) is the one place to protect: the silent
rotation needs real screen time to read on video, so if total runtime is tight, shave a few
words from Segment 2's or Segment 6's narration instead.

### If an LLM panel fails to render on camera

The app degrades visibly rather than erroring: a grey caption like "LLM explanation unavailable
(check LLM provider configuration)" appears in place of the generated text. If this happens
during a take, stop, check `.env` for `LLM_PROVIDER` / API keys, and re-record that segment — do
not narrate content that isn't actually on screen.

### Final check before publishing

1. Play back all seven narration blocks through the TTS tool once, unedited, and confirm nothing
   reads as stray punctuation.
2. Time the assembled cut. Target 2:00 to 2:10.
3. Confirm every verbatim label named above (`Risk score`, `Dementia confirmation`,
   `P(Alzheimer's pattern) (%)`, `Integrated prognosis`, etc.) is visible in the frame at the
   moment the narration references it.

---

## Full narration, one block (for the TTS tool)

Everything below is the complete voiceover in order, nothing else. Paste this whole block into
the TTS tool for a single continuous audio file, then split it at the seven paragraph breaks
during editing to line up with the recorded segments above.

```text
Meet a patient coming in with early memory concerns. Her history, a brain scan, and an EEG are already linked to her file. Watch what happens with just one click.

Right away, her risk score comes back high, and her memory test score is the biggest reason why. It has dropped sharply since her last visit. Instead of just a number, we get a clear next step, refer her to a specialist, and we can see exactly why.

Her brain scan backs that up. This slice reads as consistent with dementia, and it highlights exactly where in the brain that reading comes from. Underneath, a short written summary spells it out in plain language, there to support the read, not to replace it.

One slice can only show so much. Let's rotate the full volume and see where the changes really sit, across the whole brain, not just a single cut.

The slice and the volume do not have to stand alone. Put them side by side, and here, they agree, giving one combined read instead of two separate opinions.

Now for a third, independent read, her EEG. We are looking for the slowing pattern often seen in Alzheimer's, and here, it shows up.

A number alone is not enough. Break it down by brain wave band, compare her against a healthy group and an affected group, and map where on her scalp the slowing runs strongest. Now it is something we can actually check, not just trust.

Pull it all together, and her three assessments become one fused risk score. A clinic