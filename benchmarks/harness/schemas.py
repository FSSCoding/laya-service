"""Case schema (pydantic), loader, and validation for benchmarks/v1/cases.jsonl."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class CaseFormulation(BaseModel):
    """The question(s) to ask about the state. Mirrors /system_one request shape.
    
    For choice/score/noul: questions is a dict of qid -> question spec
    For multi_select: questions has instructions + candidates
    For rank: questions has instructions + items + rubric
    """
    type: Literal["choice", "score", "noul", "multi_select", "rank"]
    questions: dict[str, Any]  # Flexible structure based on type


class CaseGold(BaseModel):
    """Ground truth for each question. Shape depends on question type."""
    answer: Any = None  # choice: str, noul: bool, score: int
    acceptable: list[str] = Field(default_factory=list)  # alternative correct answers
    score: int | None = None  # for score type: ordinal 0..K-1
    tolerance: int = 0  # acceptable deviation for score
    labels: list[str] | None = None  # for multi_select: set of correct labels
    ordering: list[str] | None = None  # for rank: gold ordering of item ids


class Case(BaseModel):
    """A single benchmark case."""
    id: str
    family: str
    tier: Literal["deterministic", "objective", "subjective"]
    difficulty: Literal["easy", "medium", "hard"]
    split: Literal["dev", "lock"]
    state: dict[str, Any]
    formulation: CaseFormulation
    gold: dict[str, CaseGold]
    variant_of: str | None = None
    tags: list[str] = Field(default_factory=list)
    provenance: str = ""
    notes: str = ""

    @field_validator("gold", mode="before")
    @classmethod
    def parse_gold(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {k: CaseGold(**v2) if isinstance(v2, dict) else v2 for k, v2 in v.items()}
        return v


class Manifest(BaseModel):
    """Corpus manifest — version, counts, split assignment."""
    version: str
    date: str
    split_seed: int
    counts: dict[str, Any]
    provenance_notes: str = ""


def load_cases(path: Path) -> list[Case]:
    """Load and validate cases.jsonl. Returns list of Case objects."""
    cases = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                case = Case(**data)
                cases.append(case)
            except Exception as e:
                raise ValueError(f"Line {line_num}: {e}") from e
    return cases


def load_manifest(path: Path) -> Manifest:
    """Load manifest.json."""
    with open(path) as f:
        data = json.load(f)
    return Manifest(**data)


def split_cases(cases: list[Case]) -> tuple[list[Case], list[Case]]:
    """Partition cases into (dev, lock) by their split field."""
    dev = [c for c in cases if c.split == "dev"]
    lock = [c for c in cases if c.split == "lock"]
    return dev, lock
