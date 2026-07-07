### Aims

This project develops an integrated prognosis framework for Alzheimer's disease that fuses longitudinal clinical and cognitive records with 3D structural MRI volumes, targeting early-risk triage and severity prediction. The primary objective is a calibrated, interpretable risk score suitable for clinical decision support, evaluated on both prediction accuracy and diagnostic transparency.

Our pilot work uses the OASIS-1 2D cross-sectional slice mirror with a transfer-learned ResNet18 classifier. This pipeline exposes a concrete limitation that OASIS-3 access would resolve: the 2D slices force us to approximate volumetric structure by pooling predictions over central axial slices rather than modeling the brain as a 3D volume, discarding most of the anatomical information a full T1-weighted scan carries. Access to OASIS-3 longitudinal T1-weighted volumes and paired clinical data would let us replace this slice-based proxy with a native 3D pipeline and test whether volumetric and longitudinal signal improves prediction over our current 2D baseline.

### Proposed research methods

MRI volumes will be preprocessed with intensity normalization, spatial resampling, and skull-stripping, then passed through a 3D convolutional network (a 3D ResNet) to extract volumetric features. We will additionally evaluate a graph neural network over regional structural embeddings to capture anatomical relationships between brain regions.

Imaging features will be combined with longitudinal tabular records through a probabilistic fusion layer, either a log-opinion pool or a trained stacking meta-classifier, producing a single calibrated prognosis score. For interpretability, we will apply 3D Grad-CAM to project voxel-level importance back onto anatomical cross-sections, so clinicians can check which regions drive each prediction.

### Variables of interest

**Imaging:** raw and preprocessed 3D T1-weighted structural MRI volumes.

**Clinical and cognitive:** age, biological sex, years of education, socioeconomic status, and Mini-Mental State Examination score.

**Longitudinal:** visit number, normalized whole-brain volume, estimated total intracranial volume, and atlas scaling factor, used to track structural change over time.

**Target:** Clinical Dementia Rating score, requested solely as the ground-truth label for model training and validation.
