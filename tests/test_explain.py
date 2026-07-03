"""One runnable check: _case_context degrades gracefully across missing assessments."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alz.explain import _case_context, _format_recommendation, _parse_pubmed, _parse_trials

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


def test_parse_pubmed():
    payload = {
        "result": {
            "uids": ["111", "222"],
            "111": {"title": "Guideline A", "source": "Journal X", "pubdate": "2024 Jan"},
            "222": {"title": "Guideline B", "source": "Journal Y", "pubdate": "2023 Dec"},
        }
    }
    parsed = _parse_pubmed(payload)
    assert parsed == [
        {"pmid": "111", "title": "Guideline A", "source": "Journal X", "pubdate": "2024 Jan"},
        {"pmid": "222", "title": "Guideline B", "source": "Journal Y", "pubdate": "2023 Dec"},
    ]


def test_parse_trials():
    payload = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT001", "briefTitle": "Trial A"},
                    "statusModule": {"overallStatus": "RECRUITING"},
                }
            }
        ]
    }
    parsed = _parse_trials(payload)
    assert parsed == [{"nct": "NCT001", "title": "Trial A", "status": "RECRUITING"}]


def test_format_recommendation_str():
    assert _format_recommendation("Consider a follow-up MRI.") == "Consider a follow-up MRI."


def test_format_recommendation_list():
    value = [
        {"action": "Consult cardiology", "rationale": "Manage brain-heart multimorbidity."},
        {"action": "Discuss trial enrollment", "rationale": "Patient meets recruiting criteria."},
    ]
    formatted = _format_recommendation(value)
    assert "- **Consult cardiology**: Manage brain-heart multimorbidity." in formatted
    assert "- **Discuss trial enrollment**: Patient meets recruiting criteria." in formatted


def test_format_recommendation_fallback():
    assert _format_recommendation({"unexpected": "shape"}) == "{'unexpected': 'shape'}"


if __name__ == "__main__":
    test_case_context_all_missing()
    test_case_context_partial()
    test_parse_pubmed()
    test_parse_trials()
    test_format_recommendation_str()
    test_format_recommendation_list()
    test_format_recommendation_fallback()
    print("ok")
