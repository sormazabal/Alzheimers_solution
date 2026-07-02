"""FastAPI wrapper around alz.predict() -- the ~15-line seam the README promised."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from alz import predict

app = FastAPI(title="Alzheimer's Early-Risk Triage API")


class PatientRecord(BaseModel):
    Age: int
    EDUC: int
    SES: int
    MMSE: int
    nWBV: float
    ASF: float
    Visit: int
    mr_delay: int = Field(alias="MR Delay")
    sex: str = Field(alias="M/F")

    model_config = {"populate_by_name": True}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(record: PatientRecord):
    return predict(record.model_dump(by_alias=True))
