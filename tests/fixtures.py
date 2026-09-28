"""Shared test fixtures for laya-service.

Internal to this repo — not part of the installable package.
Imported by tests/conftest.py, tests/test_parity.py, and bench.py.
"""

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
