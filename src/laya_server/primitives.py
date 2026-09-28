"""Typed primitives over the Laya /system_one HTTP API.

Three native wrappers (laya_choice, laya_score, laya_noul), a heterogeneous batch
primitive (laya_batch), and three composed helpers (classify, multi_select, rank).

All primitives accept an optional ``endpoint`` argument and resolve it via
``client.resolve_endpoint`` (explicit → LAYA_ENDPOINT → default).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .client import resolve_endpoint, system_one


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoutingSignal:
    act_probability: float
    escalate_probability: float
    experimental: bool = True


@dataclass(frozen=True)
class ChoiceResult:
    choice: str
    probabilities: Dict[str, float]
    confidence: float
    routing_signal: RoutingSignal


@dataclass(frozen=True)
class ScoreResult:
    score: float
    legend: Dict[str, str]
    probabilities: Dict[str, float]
    confidence: float
    routing_signal: RoutingSignal


@dataclass(frozen=True)
class NoulResult:
    noul: float
    routing_signal: RoutingSignal


@dataclass(frozen=True)
class BatchResult:
    answers: Dict[str, Any]
    usage: Dict[str, int]
    cached: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rs(raw: Dict[str, Any]) -> RoutingSignal:
    return RoutingSignal(
        act_probability=raw.get("act_probability", 0.0),
        escalate_probability=raw.get("escalate_probability", 0.0),
        experimental=raw.get("experimental", False),
    )


def _unpack_choice(a: Dict[str, Any]) -> ChoiceResult:
    return ChoiceResult(
        choice=a["choice"],
        probabilities=dict(a["probabilities"]),
        confidence=a["confidence"],
        routing_signal=_rs(a["rl_agent"]),
    )


def _unpack_score(a: Dict[str, Any]) -> ScoreResult:
    return ScoreResult(
        score=a["score"],
        legend=dict(a["legend"]),
        probabilities=dict(a["probabilities"]),
        confidence=a["confidence"],
        routing_signal=_rs(a["rl_agent"]),
    )


def _unpack_noul(a: Dict[str, Any]) -> NoulResult:
    return NoulResult(
        noul=a["noul"],
        routing_signal=_rs(a["rl_agent"]),
    )


def _unpack(qid: str, answers: Dict[str, Any]) -> Any:
    a = answers[qid]
    t = a["type"]
    if t == "choice":
        return _unpack_choice(a)
    if t == "score":
        return _unpack_score(a)
    return _unpack_noul(a)


# ---------------------------------------------------------------------------
# Native wrappers
# ---------------------------------------------------------------------------

def laya_choice(
    state: Any,
    instructions: str,
    options: Dict[str, Any] | List[str],
    *,
    endpoint: str | None = None,
) -> ChoiceResult:
    """One choice question → ChoiceResult.

    ``options`` may be a ``{label: description}`` dict or a plain list of labels
    (descriptions default to None).
    """
    ep = resolve_endpoint(endpoint)
    if isinstance(options, list):
        criteria: Any = {label: None for label in options}
    else:
        criteria = options
    questions = {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}}
    resp = system_one(ep, state, questions)
    return _unpack_choice(resp["answers"]["q"])


def laya_score(
    state: Any,
    instructions: str,
    levels: List[str],
    *,
    endpoint: str | None = None,
) -> ScoreResult:
    """One score question → ScoreResult.  ``levels`` ordered, index 0 = lowest."""
    ep = resolve_endpoint(endpoint)
    questions = {"q": {"type": "score", "instructions": instructions, "criteria": levels}}
    resp = system_one(ep, state, questions)
    return _unpack_score(resp["answers"]["q"])


def laya_noul(
    state: Any,
    statement: str,
    *,
    true_desc: str | None = None,
    false_desc: str | None = None,
    endpoint: str | None = None,
) -> NoulResult:
    """One noul (yes/no) question → NoulResult.  Returns P(true)."""
    ep = resolve_endpoint(endpoint)
    criteria: Dict[str, Any] = {}
    if true_desc:
        criteria["true"] = true_desc
    if false_desc:
        criteria["false"] = false_desc
    questions = {"q": {"type": "noul", "instructions": statement, "criteria": criteria}}
    resp = system_one(ep, state, questions)
    return _unpack_noul(resp["answers"]["q"])


# ---------------------------------------------------------------------------
# Batch primitive
# ---------------------------------------------------------------------------

def laya_batch(
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    *,
    endpoint: str | None = None,
) -> BatchResult:
    """One forward pass for many mixed-type questions about one state.

    Each value in ``questions`` is a question spec::

        {"type": "choice", "instructions": ..., "criteria": {...}}
        {"type": "score",  "instructions": ..., "criteria": [...]}
        {"type": "noul",   "instructions": ..., "criteria": {...}}

    Returns a ``BatchResult`` with answers keyed by qid, each unpacked into the
    appropriate dataclass.
    """
    ep = resolve_endpoint(endpoint)
    resp = system_one(ep, state, questions)
    answers = {qid: _unpack(qid, resp["answers"]) for qid in resp["answers"]}
    return BatchResult(
        answers=answers,
        usage=resp.get("usage", {}),
        cached=resp.get("cached", False),
    )


# ---------------------------------------------------------------------------
# Composed helpers
# ---------------------------------------------------------------------------

def classify(
    state: Any,
    instructions: str,
    labels: List[str],
    *,
    endpoint: str | None = None,
) -> ChoiceResult:
    """N-classification pattern — sugar over ``laya_choice``.

    Semantically identical to ``laya_choice``; the name signals intent.
    """
    return laya_choice(state, instructions, labels, endpoint=endpoint)


def multi_select(
    state: Any,
    instructions: str,
    candidates: List[str],
    *,
    threshold: float = 0.5,
    endpoint: str | None = None,
) -> Dict[str, float]:
    """One noul per candidate, batched into ONE forward pass.

    Returns ``{candidate: P(true)}``.  NOT a native model type — the model
    cannot multi-select natively, so this is a composition of binary noul
    questions.

    ``threshold`` is accepted for caller convenience but NOT applied — the full
    probability distribution is preserved so the caller decides the cut-off::

        selected = {c for c, p in result.items() if p >= threshold}
    """
    questions = {
        f"c{i}": {
            "type": "noul",
            "instructions": instructions,
            "criteria": {
                "true": f"Applies: {c}",
                "false": f"Does not apply: {c}",
            },
        }
        for i, c in enumerate(candidates)
    }
    batch = laya_batch(state, questions, endpoint=endpoint)
    return {c: batch.answers[f"c{i}"].noul for i, c in enumerate(candidates)}


def rank(
    state: Any,
    instructions: str,
    items: List[str],
    rubric: List[str],
    *,
    endpoint: str | None = None,
) -> List[Tuple[str, ScoreResult]]:
    """Independent score per item against a **shared** ordered rubric.

    Every candidate is evaluated against the same ``rubric`` (index 0 = lowest).
    The item name is embedded into the per-item instructions so the model scores
    each candidate against an identical scale — making the expected-value scores
    genuinely comparable.

    Returns ``[(item, ScoreResult), ...]`` sorted descending by score.  Each
    ``ScoreResult`` preserves the full per-level probability distribution and
    confidence — not just the sorted label.

    ``rubric`` is required.  Without a caller-supplied scale there is no
    guarantee that independent scores are comparable across items.

    Pairwise preference ranking (O(n²) calls) is a future strategy if real
    usage demonstrates that independent scoring is insufficient for a domain.
    """
    questions = {
        f"item_{i}": {
            "type": "score",
            "instructions": f"{instructions} — {it}",
            "criteria": rubric,
        }
        for i, it in enumerate(items)
    }
    batch = laya_batch(state, questions, endpoint=endpoint)
    scored = [(it, batch.answers[f"item_{i}"]) for i, it in enumerate(items)]
    return sorted(scored, key=lambda pair: pair[1].score, reverse=True)
