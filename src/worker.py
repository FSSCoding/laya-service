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
from typing import Any, Dict, List, Optional

import numpy as np

from .sequence import (
    QTYPES,
    SpecialIds,
    build_sequence,
    confidence_from_probs,
    render_options,
    softmax,
    temp_bucket,
    to_internal,
    _round4,
)


class LayaWorker:
    """Holds one resident onnxruntime-gpu InferenceSession + tokenizer. Thread-safe for one call at a time."""

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
            cls=tid("[CLS]"), sep=tid("[SEP]"), mask=tid("[MASK]"), pad=tid("[PAD]"), mask_tok="[MASK]"
        )

        providers = execution_providers or [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(self.model_dir / "laya.onnx"), sess_options=so, providers=providers)
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
        """Answer every question about `state` in one forward pass (Jev's system_one request/response)."""
        qids = list(questions.keys())
        if not qids:
            raise ValueError("systemOne: at least one question is required")

        items = []
        for qid in qids:
            q = to_internal(questions[qid])
            out = build_sequence(self._encode, self.ids, state, q, self.config["max_len"], self.config["head_max_len"])
            if len(out["markers"]) != len(render_options(q)):
                raise ValueError(
                    f"question {json.dumps(qid)}: options do not fit in head_max_len={self.config['head_max_len']} tokens"
                )
            items.append({"q": q, "ids": out["ids"], "markers": out["markers"], "qtype": QTYPES[q["t"]]})

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
            ext = {"act_probability": float(act[r, 0]) if n_act else 0.0}
            qt = it["q"]["t"]
            if qt == "choice":
                keys = list(dict(it["q"]["crit"]).keys())
                best = max(range(k), key=lambda i: p[i])
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[best],
                    "probabilities": {kk: _round4(p[i]) for i, kk in enumerate(keys)},
                    "confidence": _round4(confidence_from_probs(p)),
                    "rl_agent": ext,
                }
            elif qt == "score":
                crit = it["q"]["crit"]
                answers[qid] = {
                    "type": "score",
                    "score": _round4(sum(i * v for i, v in enumerate(p))),
                    "legend": {str(i): c for i, c in enumerate(crit)},
                    "probabilities": {str(i): _round4(v) for i, v in enumerate(p)},
                    "confidence": _round4(confidence_from_probs(p)),
                    "rl_agent": ext,
                }
            else:
                answers[qid] = {"type": "noul", "noul": _round4(p[1]), "rl_agent": ext}

        return {
            "model": "laya",
            "answers": answers,
            "usage": {"input_tokens": n_tokens, "output_tokens": 0},
        }

    def release(self) -> None:
        with self._lock:
            try:
                self._session.close()
            except Exception:
                pass
