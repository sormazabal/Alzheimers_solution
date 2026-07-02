"""Phase-2 helper: fetches the OASIS imaging dataset for the (stubbed) MRI/CNN pipeline.

Not part of the tabular demo -- that uses the bundled data/oasis_longitudinal.csv.
Requires kagglehub (see requirements-imaging.txt).
"""
import kagglehub

path = kagglehub.dataset_download("ninadaithal/imagesoasis")

print("Path to dataset files:", path)
