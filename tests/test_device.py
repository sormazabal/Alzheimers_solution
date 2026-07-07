"""One runnable check: _device()'s cuda -> directml -> cpu fallback chain.

Skips cleanly if torch isn't installed.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")

from alz.imaging import _device


def test_explicit_device_wins(monkeypatch):
    assert _device("cpu") == "cpu"


def test_cuda_preferred_over_directml(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _device() == "cuda"


def test_directml_used_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fake_directml = types.SimpleNamespace(is_available=lambda: True, device=lambda: "dml")
    monkeypatch.setitem(sys.modules, "torch_directml", fake_directml)
    assert _device() == "dml"


def test_cpu_when_neither_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "torch_directml", None)  # simulates ImportError
    assert _device() == "cpu"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
