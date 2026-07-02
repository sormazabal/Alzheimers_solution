"""One runnable check: _case_context degrades gracefully across missing assessments."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alz.explain import _case_context

PATIENT = {"sex": "F", "dob": "1950-03-14", "history": "Hypertension"}


def test_case_context_all_missing():
    ctx = _case_context(PATIENT, None, None, None)
    assert ctx.count("not yet assessed") == 3


def test_case_context_partial():
    clinical_result = {"label": "At-risk", "score": 0.72, "drivers": [{"feature": "MMSE", "direction": "raises risk"}]}
    ctx = _case_context(PATIENT, clinical_result, None, {"label": "Mild abnormality", "score": 0.4})
    assert "At-risk (72%)" in ctx
    assert "MMSE (raises risk)" in ctx
    assert "MRI severity classifier: not yet assessed." in ctx
    assert "Mild abnormality (40% confidence)" in ctx


if __name__ == "__main__":
    test_case_context_all_missing()
    test_case_context_partial()
    print("ok")
