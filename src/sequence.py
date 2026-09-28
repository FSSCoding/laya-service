"""
Pure (model-free) Python port of the request rendering in the checkpoint's rl_common.py —
a faithful port of /MASTERFOLDER/Services/laya/src/sequence.ts.

Kept separate from the ONNX session so it can be unit-tested without the weights. Every
function mirrors its TypeScript counterpart exactly so that GPU outputs match the CPU
reference to 4 decimals.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

# choice -> 0, score -> 1, noul -> 2
QTYPES: Dict[str, int] = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES: List[str] = ["choice", "score", "noul"]


def to_internal(q: Dict[str, Any]) -> Dict[str, Any]:
    """RLAgent._to_internal. Normalise criteria for the given question type."""
    crit = q.get("criteria")
    if q["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = q.get("instructions")
    if isinstance(ins, str):
        ins_s = ins
    else:
        ins_s = json_dumps(ins)
    return {"t": q["type"], "ins": ins_s, "crit": crit}


def render_options(q: Dict[str, Any]) -> List[str]:
    """Option texts in label-index order. Noul is always [false, true] so p[1] == noul."""
    t = q["t"]
    crit = q["crit"]
    if t == "choice":
        # JS Object.entries over a Record<string, string|null>: insertion key order
        out: List[str] = []
        for k, v in dict(crit).items():
            out.append(f"{k}: {v}" if v else k)
        return out
    if t == "score":
        return [f"level {i}: {c}" for i, c in enumerate(crit)]
    c = crit or {}
    return [
        "false: " + (c.get("false") or "no, the statement does not hold"),
        "true: " + (c.get("true") or "yes, the statement holds"),
    ]


def json_dumps(v: Any) -> str:
    """Python's json.dumps(obj, ensure_ascii=False): ', ' / ': ' separators, insertion key order.

    Mirrors sequence.ts pyJsonDumps. For the supported scalar/list/dict shapes this is exactly
    json.dumps with separators=(", ", ": ").
    """
    return json.dumps(v, ensure_ascii=False, separators=(", ", ": "))


def serialize_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json_dumps(state)


def size_bucket(k: int) -> str:
    if k <= 2:
        return "2"
    if k <= 5:
        return "3-5"
    if k <= 10:
        return "6-10"
    return "11+"


def temp_bucket(qtype: int, k: int) -> str:
    """Key for the per-cardinality temperature: a 2-option noul and a 20-option choice need different scaling."""  # noqa: E501
    return f"{QTYPE_NAMES[qtype]}:{size_bucket(k)}"


def confidence_from_probs(p: Sequence[float]) -> float:
    """Jev-style confidence: 1 - normalized entropy of the answer distribution."""
    k = len(p)
    if k < 2:
        return 1.0
    ent = 0.0
    for x in p:
        ent -= x * math_log(max(x, 1e-12))
    return 1.0 - ent / math_log(k)


def softmax(z: Sequence[float]) -> List[float]:
    zmax = max(z)
    e = [math_exp(v - zmax) for v in z]
    s = sum(e)
    return [v / s for v in e]


def _round4(x: float) -> float:
    # JS Math.round(x*1e4)/1e4. Python round() uses banker's rounding; JS uses half-up.
    # Match JS: round away from zero on .5 ties.
    import math

    r = math.floor(x * 1e4 + 0.5) if x >= 0 else math.ceil(x * 1e4 - 0.5)
    return r / 1e4


# local math aliases to keep call sites readable and match the TS names
def math_log(x: float) -> float:
    import math

    return math.log(x)


def math_exp(x: float) -> float:
    import math

    return math.exp(x)


class SpecialIds:
    def __init__(self, cls: int, sep: int, mask: int, pad: int, mask_tok: str = "[MASK]") -> None:
        self.cls = cls
        self.sep = sep
        self.mask = mask
        self.pad = pad
        self.mask_tok = mask_tok


def build_sequence(
    encode,
    ids: SpecialIds,
    state: Any,
    q: Dict[str, Any],
    max_len: int,
    head_max_len: int,
) -> Dict[str, List[int]]:
    """
    rl_common.build_sequence:
      [CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
    Returns the ids and the position of each option's [MASK] marker.
    """

    def scrub(s: str) -> str:
        return s.replace(ids.mask_tok, " ")

    opts = render_options(q)
    head_ids = encode(f"{q['t']} question: {scrub(q['ins'])}")
    # opt0 = [MASK] + encode(" " + opt)[:48]  (leading space consumed by the " " prefix in TS)
    opt_ids = [[ids.mask] + encode(" " + scrub(o))[:48] for o in opts]
    def _total(xs):
        return sum(len(o) for o in xs)

    opt_budget = head_max_len - _total(opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - _total(opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    seq = [ids.cls] + head_ids + [ids.sep]
    markers: List[int] = []
    for o in opt_ids:
        markers.append(len(seq))
        seq.extend(o)
    seq.append(ids.sep)
    room = max(0, max_len - len(seq) - 1)
    st = encode(scrub(serialize_state(state)))[:room]
    seq.extend(st)
    seq.append(ids.sep)
    ids_out = seq[:max_len]
    markers_out = [m for m in markers if m < max_len]
    return {"ids": ids_out, "markers": markers_out}
