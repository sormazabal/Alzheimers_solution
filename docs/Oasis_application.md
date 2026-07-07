

### Aims

To investigate multimodal data fusion techniques that combine longitudinal clinical, cognitive, and structural neuroimaging data for early-risk triage and severity prediction of Alzheimer's disease. The primary objective is to develop and validate an integrated prognosis framework that combines tabular electronic health records with 3D structural MRI volumes, optimizing prediction calibration and diagnostic transparency for clinical decision support systems.

### Proposed research methods

The project will employ advanced machine learning and deep learning methodologies to process and integrate longitudinal multi-modal datasets. The structural 3D MRI volumes will undergo preprocessing, including intensity normalisation, spatial resampling, and skull-stripping. Volumetric feature extraction will be conducted using 3D convolutional neural networks, such as 3D ResNet architectures, alongside graph neural networks utilising regional structural brain embeddings to map anatomical relationships.

These extracted imaging features will be combined with longitudinal tabular records through a probabilistic fusion layer, specifically a logarithmic opinion pool or a trained stacking meta-classifier, to output a single, calibrated prognosis score. To ensure clinical interpretability, feature attribution methods such as 3D Grad-CAM will be implemented to project voxel-level importance metrics back onto anatomical cross-sections, allowing clinical verification of the regions driving the automated prediction.

### Variables of interest

The primary variables required include raw and preprocessed 3D structural MRI data, specifically T1-weighted scans. From the clinical and cognitive profiles, the variables of interest include patient age, biological sex, years of education, socioeconomic status, and Mini-Mental State Examination scores. Longitudinal metrics such as visit number, normalised whole-brain volume, estimated total intracranial volume, and atlas scaling factors will be utilised to track structural changes over time. Clinical Dementia Rating scores will be requested exclusively to serve as the ground-truth target for model training and validation.