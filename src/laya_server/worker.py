"""
Laya ONNX inference worker (GPU).

Loads the ONNX bundle once (resident), builds inputs for any number of typed questions about one
state, runs a single forward pass via onnxruntime-gpu, and shapes the calibrated answers exactly as
/MasterFOLDER/Services/laya/src/laya.ts does. The request/response shape follows Jev's
system_one API.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .sequence import (
    QTYPES,
    SpecialIds,
    _round4,
    build_sequence,
    confidence_from_probs,
    render_options,
    softmax,
    temp_bucket,
    to_internal,
)


class LayaWorker:
    """Holds one resident onnxruntime-gpu InferenceSession + tokenizer.

    Thread-safe for one call at a time.
    """

    def __init__(self, model_dir: str, execution_providers=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer as _Tkz

        self._ort = ort
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"model dir not found: {self.model_dir}")

        with open(self.model_dir / "laya_config.json") as f:
            self.config = json.load(f)

        with open(self.model_dir / "tokenizer" / "tokenizer.json") as f:
            self._tok = _Tkz.from_file(str(self.model_dir / "tokenizer" / "tokenizer.json"))

        # special token ids (raise if missing, matching laya.ts)
        def tid(t: str) -> int:
            v = self._tok.token_to_id(t)
            if v is None:
                raise ValueError(f"special token {t} missing from tokenizer")
            return v

        self.ids = SpecialIds(
            cls=tid("[CLS]"),
            sep=tid("[SEP]"),
            mask=tid("[MASK]"),
            pad=tid("[PAD]"),
            mask_tok="[MASK]",
        )

        providers = execution_providers or [
            ("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {}),
        ]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_dir / "laya.onnx"), sess_options=so, providers=providers,
        )
        self._lock = threading.Lock()

    @property
    def active_providers(self) -> List[str]:
        try:
            return [p[0] if isinstance(p, tuple) else p for p in self._session.get_providers()]
        except Exception:
            return []

    def _encode(self, text: str) -> List[int]:
        # add_special_tokens: False (TS encode)
        ids = self._tok.encode(text, add_special_tokens=False).ids
        return ids

    def system_one(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Answer every question about `state` in one forward pass."""
        qids = list(questions.keys())
        if not qids:
            raise ValueError("systemOne: at least one question is required")

        items = []
        any_truncated = False
        original_input_tokens = 0
        for qid in qids:
            q = to_internal(questions[qid])
            # Pre-ONNX validation — client errors that should be 400, not 500.
            if q["t"] not in QTYPES:
                raise ValueError(f"unknown question type: {q['t']!r}")
            if q["t"] == "score":
                crit = q["crit"] or []
                if len(crit) < 2:
                    raise ValueError(
                        f"question {json.dumps(qid)}: score requires at least "
                        f"2 levels, got {len(crit)}"
                    )
                if len(crit) > 10:
                    raise ValueError(
                        f"question {json.dumps(qid)}: score supports 2–10 "
                        f"levels, got {len(crit)}"
                    )
            out = build_sequence(
                self._encode, self.ids, state, q,
                self.config["max_len"], self.config["head_max_len"],
            )
            if len(out["markers"]) != len(render_options(q)):
                raise ValueError(
                    f"question {json.dumps(qid)}: options do not fit in "
                    f"head_max_len={self.config['head_max_len']} tokens"
                )
            if out.get("truncated"):
                any_truncated = True
                original_input_tokens += out.get("original_state_tokens", 0)
            items.append(
                {"q": q, "ids": out["ids"], "markers": out["markers"], "qtype": QTYPES[q["t"]]}
            )

        n = len(items)
        L = max(len(it["ids"]) for it in items)
        K = max(len(it["markers"]) for it in items)

        input_ids = np.full((n, L), self.ids.pad, dtype=np.int64)
        attention = np.zeros((n, L), dtype=np.int64)
        marker_pos = np.zeros((n, K), dtype=np.int64)
        marker_mask = np.zeros((n, K), dtype=np.bool_)
        qtype = np.zeros((n,), dtype=np.int64)
        n_tokens = 0
        for i, it in enumerate(items):
            for j, v in enumerate(it["ids"]):
                input_ids[i, j] = v
                attention[i, j] = 1
            n_tokens += len(it["ids"])
            for j, m in enumerate(it["markers"]):
                marker_pos[i, j] = m
                marker_mask[i, j] = True
            qtype[i] = it["qtype"]

        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "marker_pos": marker_pos,
            "marker_mask": marker_mask,
            "qtype": qtype,
        }
        with self._lock:
            logits, act = self._session.run(["logits", "act_probs"], feeds)
        logits = np.asarray(logits, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32)
        n_act = act.shape[1] if act.ndim > 1 else 1

        answers: Dict[str, Any] = {}
        for r, it in enumerate(items):
            qid = qids[r]
            k = len(it["markers"])
            temp = (
                self.config["temperature_by_options"].get(temp_bucket(it["qtype"], k))
                or self.config["temperature"][it["qtype"]]
                or 1.0
            )
            p = softmax([v / temp for v in logits[r, :k].tolist()])
            routing = {
                "act_probability": _round4(float(act[r, 0])) if n_act else 0.0,
                "escalate_probability": (
                    _round4(float(act[r, 1])) if n_act >= 2 else 0.0
                ),
                "experimental": True,
            }
            qt = it["q"]["t"]
            if qt == "choice":
                keys = list(dict(it["q"]["crit"]).keys())
                best = max(range(k), key=lambda i: p[i])
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[best],
                    "probabilities": {kk: _round4(p[i]) for i, kk in enumerate(keys)},
                    "confidence": _round4(confidence_from_probs(p)),
                    "rl_agent": routing,
                }
            elif qt == "score":
                crit = it["q"]["crit"]
                answers[qid] = {
                    "type": "score",
                    "score": _round4(sum(i * v for i, v in enumerate(p))),
                    "legend": {str(i): c for i, c in enumerate(crit)},
                    "probabilities": {str(i): _round4(v) for i, v in enumerate(p)},
                    "confidence": _round4(confidence_from_probs(p)),
                    "rl_agent": routing,
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": _round4(p[1]),
                    "rl_agent": routing,
                }

        usage: Dict[str, Any] = {"input_tokens": n_tokens, "output_tokens": 0}
        if any_truncated:
            usage["truncated"] = True
            usage["original_input_tokens"] = original_input_tokens

        return {
            "model": "laya",
            "answers": answers,
            "usage": usage,
        }

    def system_many(
        self, pairs: List[tuple]
    ) -> List[Dict[str, Any]]:
        """Answer questions for multiple (state, questions) pairs in one forward pass.

        Each pair is (state, questions_dict) — the same shape as a single
        ``system_one`` call.  All questions across all pairs are merged into
        one batched ONNX call, then split back into per-pair results.

        Returns a list of result dicts, one per input pair, in the same order.
        """
        if not pairs:
            return []

        # Flatten: collect (pair_index, qid, state, question) for every question.
        flat: List[tuple] = []  # (pair_idx, qid, state, q_internal)
        for pair_idx, (state, questions) in enumerate(pairs):
            for qid, q_raw in questions.items():
                q = to_internal(q_raw)
                if q["t"] not in QTYPES:
                    raise ValueError(f"unknown question type: {q['t']!r}")
                if q["t"] == "score":
                    crit = q["crit"] or []
                    if len(crit) < 2:
                        raise ValueError(
                            f"question {json.dumps(qid)}: score requires at "
                            f"least 2 levels, got {len(crit)}"
                        )
                    if len(crit) > 10:
                        raise ValueError(
                            f"question {json.dumps(qid)}: score supports "
                            f"2–10 levels, got {len(crit)}"
                        )
                flat.append((pair_idx, qid, state, q))

        # Build sequences for all items.
        items = []
        pair_truncated: Dict[int, bool] = {}
        pair_original_tokens: Dict[int, int] = {}
        for pair_idx, qid, state, q in flat:
            out = build_sequence(
                self._encode, self.ids, state, q,
                self.config["max_len"], self.config["head_max_len"],
            )
            if len(out["markers"]) != len(render_options(q)):
                raise ValueError(
                    f"question {json.dumps(qid)}: options do not fit in "
                    f"head_max_len={self.config['head_max_len']} tokens"
                )
            if out.get("truncated"):
                pair_truncated[pair_idx] = True
                pair_original_tokens[pair_idx] = (
                    pair_original_tokens.get(pair_idx, 0)
                    + out.get("original_state_tokens", 0)
                )
            items.append({
                "pair_idx": pair_idx,
                "qid": qid,
                "q": q,
                "ids": out["ids"],
                "markers": out["markers"],
                "qtype": QTYPES[q["t"]],
            })

        n = len(items)
        L = max(len(it["ids"]) for it in items)
        K = max(len(it["markers"]) for it in items)

        input_ids = np.full((n, L), self.ids.pad, dtype=np.int64)
        attention = np.zeros((n, L), dtype=np.int64)
        marker_pos = np.zeros((n, K), dtype=np.int64)
        marker_mask = np.zeros((n, K), dtype=np.bool_)
        qtype = np.zeros((n,), dtype=np.int64)
        pair_tokens: Dict[int, int] = {}
        for i, it in enumerate(items):
            for j, v in enumerate(it["ids"]):
                input_ids[i, j] = v
                attention[i, j] = 1
            pair_tokens[it["pair_idx"]] = pair_tokens.get(it["pair_idx"], 0) + len(it["ids"])
            for j, m in enumerate(it["markers"]):
                marker_pos[i, j] = m
                marker_mask[i, j] = True
            qtype[i] = it["qtype"]

        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "marker_pos": marker_pos,
            "marker_mask": marker_mask,
            "qtype": qtype,
        }
        with self._lock:
            logits, act = self._session.run(["logits", "act_probs"], feeds)
        logits = np.asarray(logits, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32)
        n_act = act.shape[1] if act.ndim > 1 else 1

        # Group answers back by pair index.
        pair_answers: Dict[int, Dict[str, Any]] = {i: {} for i in range(len(pairs))}
        for r, it in enumerate(items):
            pair_idx = it["pair_idx"]
            qid = it["qid"]
            k = len(it["markers"])
            temp = (
                self.config["temperature_by_options"].get(temp_bucket(it["qtype"], k))
                or self.config["temperature"][it["qtype"]]
                or 1.0
            )
            p = softmax([v / temp for v in logits[r, :k].tolist()])
            routing = {
                "act_probability": _round4(float(act[r, 0])) if n_act else 0.0,
                "escalate_probability": (
                    _round4(float(act[r, 1])) if n_act >= 2 else 0.0
                ),
                "experimental": True,
            }
            qt = it["q"]["t"]
            if qt == "choice":
                keys = list(dict(it["q"]["crit"]).keys())
                best = max(range(k), key=lambda i: p[i])
                pair_answers[pair_idx][qid] = {
                    "type": "choice",
                    "choice": keys[best],
                    "probabilities": {kk: _round4(p[i]) for i, kk in enumerate(keys)},
                    "confidence": _round4(confidence_from_probs(p)),
                    "rl_agent": routing,
                }
            elif qt == "score":
                crit = it["q"]["crit"]
                pair_answers[pair_idx][qid] = {
                    "type": "score",
                    "score": _round4(sum(i * v for i, v in enumerate(p))),
                    "legend": {str(i): c for i, c in enumerate(crit)},
                    "probabilities": {str(i): _round4(v) for i, v in enumerate(p)},
                    "confidence": _round4(confidence_from_probs(p)),
                    "rl_agent": routing,
                }
            else:
                pair_answers[pair_idx][qid] = {
                    "type": "noul",
                    "noul": _round4(p[1]),
                    "rl_agent": routing,
                }

        # Build per-pair results.
        results = []
        for pair_idx in range(len(pairs)):
            usage: Dict[str, Any] = {
                "input_tokens": pair_tokens.get(pair_idx, 0),
                "output_tokens": 0,
            }
            if pair_truncated.get(pair_idx):
                usage["truncated"] = True
                usage["original_input_tokens"] = pair_original_tokens.get(pair_idx, 0)
            results.append({
                "model": "laya",
                "answers": pair_answers[pair_idx],
                "usage": usage,
            })
        return results

    def release(self) -> None:
        with self._lock:
            try:
                self._session.close()
            except Exception:
                pass
