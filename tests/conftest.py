"""Shared fixtures for laya-service tests."""
import os
import sys
from pathlib import Path

import pytest

# Make src/ importable when not already installed via pip install -e .
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

MODEL_DIR = os.environ.get("LAYA_MODEL_DIR", "/MASTERFOLDER/Services/laya/onnx")

STATE = {
    "subject": "Refund not received",
    "body": (
        "I cancelled my subscription two weeks ago and I still have not received my refund. "
        "This is the third time I am writing. If this is not resolved I will "
        "dispute the charge with my bank."
    ),
}

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "payments, refunds, invoices",
            "support": "product help and bugs",
            "sales": "new purchases and upgrades",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"],
    },
    "churn_risk": {"type": "noul", "instructions": "Is the customer likely to cancel or dispute?"},
}


@pytest.fixture(scope="module")
def worker():
    """LayaWorker with CUDA (skips if GPU unavailable)."""
    from src.worker import LayaWorker

    try:
        w = LayaWorker(MODEL_DIR)
    except (ImportError, FileNotFoundError, OSError) as e:
        pytest.skip(f"CUDA/model unavailable: {e}")
    yield w
    w.release()
