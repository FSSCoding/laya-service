"""Laya GPU inference service."""

__version__ = "0.3.0"

from .primitives import (
    BatchResult,
    ChoiceResult,
    NoulResult,
    RoutingSignal,
    ScoreResult,
    classify,
    laya_batch,
    laya_choice,
    laya_noul,
    laya_score,
    multi_select,
    rank,
)

__all__ = [
    "__version__",
    "BatchResult",
    "ChoiceResult",
    "NoulResult",
    "RoutingSignal",
    "ScoreResult",
    "classify",
    "laya_batch",
    "laya_choice",
    "laya_noul",
    "laya_score",
    "multi_select",
    "rank",
]
