"""Model-quality metrics: per-family, per-tier, per-split. All §5 metrics, hand-rolled.

No scipy/numpy — pure Python math (~100 lines for AUC/Spearman/Kendall/ECE/Brier).
All metrics have applicability guards: they return None with a reason when conditions
aren't met (insufficient N, no variance, missing classes, etc.).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

from .schemas import Case, load_cases, split_cases


# ── Metric validity ───────────────────────────────────────────────────────────

class MetricResult(NamedTuple):
    """A metric value or N/A with reason."""
    value: float | None
    reason: str | None = None  # None means valid
    
    @property
    def is_valid(self) -> bool:
        return self.reason is None
    
    def to_json(self) -> Any:
        if self.is_valid:
            return self.value
        return {"value": None, "reason": self.reason}


def require_min_n(n: int, min_n: int, metric_name: str) -> MetricResult | None:
    """Return N/A if n < min_n."""
    if n < min_n:
        return MetricResult(None, f"{metric_name}: insufficient N ({n} < {min_n})")
    return None


def require_both_classes(labels: list[int], metric_name: str) -> MetricResult | None:
    """Return N/A if not both 0 and 1 are present."""
    if len(set(labels)) < 2:
        return MetricResult(None, f"{metric_name}: only one class represented")
    return None


def require_variance(vals: list[float], metric_name: str) -> MetricResult | None:
    """Return N/A if all values are identical (no variance)."""
    if len(set(vals)) < 2:
        return MetricResult(None, f"{metric_name}: no variance in input")
    return None


# ── Statistical helpers (hand-rolled, zero deps) ──────────────────────────────

def auc_roc(labels: list[int], scores: list[float]) -> float:
    """Area under ROC curve (trapezoidal rule). labels: 0/1, scores: higher = more positive."""
    if len(labels) != len(scores) or len(labels) < 2:
        return 0.5
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5
    prev_fpr = prev_tpr = 0.0
    auc = 0.0
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / total_neg
        tpr = tp / total_pos
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr
    return auc


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - (6 * d_sq) / (n * (n ** 2 - 1))


def kendall_tau(x: list[float], y: list[float]) -> float:
    """Kendall τ rank correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total if total > 0 else 0.0


def _rank(vals: list[float]) -> list[float]:
    """Assign ranks (1-based, average ties)."""
    indexed = sorted(enumerate(vals), key=lambda p: p[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def ece(probs: list[float], correct: list[int], n_bins: int = 10) -> float:
    """Expected calibration error (uniform bins)."""
    if not probs:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for p, c in zip(probs, correct):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, c))
    ece_val = 0.0
    total = len(probs)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(p for p, _ in b) / len(b)
        avg_acc = sum(c for _, c in b) / len(b)
        ece_val += len(b) / total * abs(avg_conf - avg_acc)
    return ece_val


def brier_score(probs: list[float], outcomes: list[int]) -> float:
    """Brier score (lower is better). probs: P(true), outcomes: 0/1."""
    if not probs:
        return 0.0
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


# ── Correctness extraction ────────────────────────────────────────────────────

def extract_choice_correct(answer: dict, gold: Any) -> tuple[bool, float, float]:
    """Returns (correct, P(gold), confidence)."""
    choice = answer.get("choice", "")
    probs = answer.get("probabilities", {})
    gold_answer = gold.answer if hasattr(gold, 'answer') else gold["answer"]
    gold_acceptable = gold.acceptable if hasattr(gold, 'acceptable') else gold.get("acceptable", [])
    p_gold = probs.get(gold_answer, 0.0)
    confidence = answer.get("confidence", 0.0)
    correct = choice == gold_answer or choice in gold_acceptable
    return correct, p_gold, confidence


def extract_noul_correct(answer: dict, gold: Any) -> tuple[bool, float, float]:
    """Returns (correct, P(true), confidence).
    
    P(true) is the model's raw output, not flipped by gold. Gold determines correctness only.
    """
    noul = answer.get("noul", 0.5)
    gold_answer = gold.answer if hasattr(gold, 'answer') else gold["answer"]
    p_true = noul  # Always the model's P(true), regardless of gold
    confidence = answer.get("confidence", 0.0)
    correct = (noul >= 0.5) == gold_answer
    return correct, p_true, confidence


def extract_score_correct(answer: dict, gold: Any) -> tuple[bool, float, float]:
    """Returns (correct_within_tolerance, predicted_ordinal, confidence)."""
    score = answer.get("score", 0.0)
    probs = answer.get("probabilities", {})
    confidence = answer.get("confidence", 0.0)
    # Predicted ordinal = argmax of level probabilities
    pred_ordinal = max(range(len(probs)), key=lambda i: probs.get(str(i), 0.0))
    gold_score = gold.score if hasattr(gold, 'score') else gold["score"]
    gold_tolerance = gold.tolerance if hasattr(gold, 'tolerance') else gold.get("tolerance", 0)
    correct = abs(pred_ordinal - gold_score) <= gold_tolerance
    return correct, pred_ordinal, confidence


def extract_routing(answer: dict) -> tuple[float, float]:
    """Extract (act_probability, escalate_probability) from rl_agent."""
    rl = answer.get("rl_agent", {})
    return rl.get("act_probability", 0.0), rl.get("escalate_probability", 0.0)


# ── Per-family metrics ────────────────────────────────────────────────────────

def compute_choice_metrics(records: list[dict], cases: dict[str, Case]) -> dict:
    """Choice metrics: top-1, top-2, P(gold), confidence, option-count sensitivity, ECE."""
    correct_list = []
    p_gold_list = []
    conf_list = []
    option_counts = []
    for rec in records:
        case = cases.get(rec["case_id"])
        if not case or case.formulation.type != "choice":
            continue
        answers = rec.get("response", {}).get("answers", {})
        for qid, gold in case.gold.items():
            ans = answers.get(qid, {})
            ok, pg, conf = extract_choice_correct(ans, gold)
            correct_list.append(int(ok))
            p_gold_list.append(pg)
            conf_list.append(conf)
            n_opts = len(case.formulation.questions.get(qid, {}).get("criteria", {}))
            option_counts.append(n_opts)

    if not correct_list:
        return {"n_cases": 0}

    # Option-count sensitivity
    bins = {"2": [], "3-5": [], "6-10": [], "11-20": [], "25+": []}
    for ok, n in zip(correct_list, option_counts):
        if n <= 2:
            bins["2"].append(ok)
        elif n <= 5:
            bins["3-5"].append(ok)
        elif n <= 10:
            bins["6-10"].append(ok)
        elif n <= 20:
            bins["11-20"].append(ok)
        else:
            bins["25+"].append(ok)
    opt_sensitivity = {k: sum(v) / len(v) if v else None for k, v in bins.items()}

    return {
        "n_cases": len(correct_list),
        "top1_accuracy": sum(correct_list) / len(correct_list),
        "p_gold_mean": sum(p_gold_list) / len(p_gold_list),
        "p_gold_median": sorted(p_gold_list)[len(p_gold_list) // 2],
        "confidence_mean": sum(conf_list) / len(conf_list),
        "ece": ece(conf_list, correct_list),
        "option_count_sensitivity": opt_sensitivity,
    }


def compute_noul_metrics(records: list[dict], cases: dict[str, Case]) -> dict:
    """Noul metrics: accuracy at thresholds, AUC, Brier, phrasing symmetry."""
    p_true_list = []
    outcome_list = []
    conf_list = []
    correct_list = []
    for rec in records:
        case = cases.get(rec["case_id"])
        if not case or case.formulation.type != "noul":
            continue
        answers = rec.get("response", {}).get("answers", {})
        for qid, gold in case.gold.items():
            ans = answers.get(qid, {})
            ok, pt, conf = extract_noul_correct(ans, gold)
            p_true_list.append(pt)
            gold_answer = gold.answer if hasattr(gold, 'answer') else gold["answer"]
            outcome_list.append(int(gold_answer))
            conf_list.append(conf)
            correct_list.append(int(ok))

    if not p_true_list:
        return {"n_cases": 0}

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    thresh_metrics = {}
    for t in thresholds:
        pred = [1 if p >= t else 0 for p in p_true_list]
        tp = sum(1 for p, o in zip(pred, outcome_list) if p == 1 and o == 1)
        fp = sum(1 for p, o in zip(pred, outcome_list) if p == 1 and o == 0)
        fn = sum(1 for p, o in zip(pred, outcome_list) if p == 0 and o == 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = sum(1 for p, o in zip(pred, outcome_list) if p == o) / len(outcome_list)
        thresh_metrics[str(t)] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # AUC: requires both classes + min N
    auc_result = require_both_classes(outcome_list, "AUC")
    if auc_result is None:
        auc_result = require_min_n(len(outcome_list), 5, "AUC")
    if auc_result is None:
        auc_val = auc_roc(outcome_list, p_true_list)
    else:
        auc_val = auc_result.to_json()

    # Brier: needs at least 1 sample (robust)
    brier_val = brier_score(p_true_list, outcome_list) if p_true_list else None

    # ECE: needs meaningful sample
    ece_result = require_min_n(len(p_true_list), 10, "ECE")
    if ece_result is None:
        ece_val = ece(p_true_list, outcome_list)
    else:
        ece_val = ece_result.to_json()

    return {
        "n_cases": len(p_true_list),
        "auc": auc_val,
        "brier": brier_val,
        "ece": ece_val,
        "thresholds": thresh_metrics,
        "accuracy_at_0.5": sum(correct_list) / len(correct_list),
    }


def compute_score_metrics(records: list[dict], cases: dict[str, Case]) -> dict:
    """Score metrics: MAE, Spearman, Kendall, monotonicity."""
    pred_ordinals = []
    gold_ordinals = []
    conf_list = []
    for rec in records:
        case = cases.get(rec["case_id"])
        if not case or case.formulation.type != "score":
            continue
        answers = rec.get("response", {}).get("answers", {})
        for qid, gold in case.gold.items():
            ans = answers.get(qid, {})
            ok, pred, conf = extract_score_correct(ans, gold)
            pred_ordinals.append(pred)
            gold_score = gold.score if hasattr(gold, 'score') else gold["score"]
            gold_ordinals.append(gold_score)
            conf_list.append(conf)

    if not pred_ordinals:
        return {"n_cases": 0}

    mae = sum(abs(p - g) for p, g in zip(pred_ordinals, gold_ordinals)) / len(pred_ordinals)
    exact = sum(1 for p, g in zip(pred_ordinals, gold_ordinals) if p == g) / len(pred_ordinals)

    # Spearman/Kendall: need min N + variance in both inputs
    gold_floats = [float(x) for x in gold_ordinals]
    pred_floats = [float(x) for x in pred_ordinals]
    
    spearman_result = require_min_n(len(pred_ordinals), 3, "Spearman")
    if spearman_result is None:
        spearman_result = require_variance(gold_floats, "Spearman")
    if spearman_result is None:
        spearman_result = require_variance(pred_floats, "Spearman")
    if spearman_result is None:
        spearman_val = spearman_rho(gold_floats, pred_floats)
    else:
        spearman_val = spearman_result.to_json()

    kendall_result = require_min_n(len(pred_ordinals), 3, "Kendall")
    if kendall_result is None:
        kendall_result = require_variance(gold_floats, "Kendall")
    if kendall_result is None:
        kendall_result = require_variance(pred_floats, "Kendall")
    if kendall_result is None:
        kendall_val = kendall_tau(gold_floats, pred_floats)
    else:
        kendall_val = kendall_result.to_json()

    return {
        "n_cases": len(pred_ordinals),
        "mae": mae,
        "exact_hit": exact,
        "spearman": spearman_val,
        "kendall": kendall_val,
    }


def compute_multi_select_metrics(records: list[dict], cases: dict[str, Case]) -> dict:
    """Multi-select metrics: P/R/F1 at threshold sweep."""
    all_preds = []  # list of (predicted_set, gold_set)
    for rec in records:
        case = cases.get(rec["case_id"])
        if not case or case.formulation.type != "multi_select":
            continue
        answers = rec.get("response", {}).get("answers", {})
        gold_labels = set()
        for qid, gold in case.gold.items():
            gold_labels_val = gold.labels if hasattr(gold, 'labels') else gold.get("labels")
            if gold_labels_val:
                gold_labels.update(gold_labels_val)
        # Each answer is a noul; collect P(true) per candidate
        pred_scores = {}
        for qid, ans in answers.items():
            pred_scores[qid] = ans.get("noul", 0.5)
        all_preds.append((pred_scores, gold_labels))

    if not all_preds:
        return {"n_cases": 0}

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    thresh_metrics = {}
    for t in thresholds:
        tp = fp = fn = 0
        for pred_scores, gold_labels in all_preds:
            pred_set = {k for k, v in pred_scores.items() if v >= t}
            tp += len(pred_set & gold_labels)
            fp += len(pred_set - gold_labels)
            fn += len(gold_labels - pred_set)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        thresh_metrics[str(t)] = {"precision": prec, "recall": rec, "f1": f1}

    return {
        "n_cases": len(all_preds),
        "thresholds": thresh_metrics,
    }


def compute_rank_metrics(records: list[dict], cases: dict[str, Case]) -> dict:
    """Rank metrics: Spearman/Kendall vs gold ordering, pairwise accuracy."""
    gold_orderings = []
    pred_orderings = []
    for rec in records:
        case = cases.get(rec["case_id"])
        if not case or case.formulation.type != "rank":
            continue
        answers = rec.get("response", {}).get("answers", {})
        # Get the first gold entry (rank cases have one gold with ordering)
        first_gold_key = list(case.gold.keys())[0] if case.gold else None
        if not first_gold_key:
            continue
        gold_obj = case.gold[first_gold_key]
        gold_order = gold_obj.ordering if hasattr(gold_obj, 'ordering') else gold_obj.get("ordering")
        if not gold_order:
            continue
        # Predicted ordering: sort items by score descending
        item_scores = []
        for item_id, ans in answers.items():
            item_scores.append((item_id, ans.get("score", 0.0)))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        pred_order = [item for item, _ in item_scores]
        gold_orderings.append(gold_order)
        pred_orderings.append(pred_order)

    if not gold_orderings:
        return {"n_cases": 0}

    # Convert orderings to rank vectors for correlation
    all_spearman = []
    all_kendall = []
    pairwise_correct = 0
    pairwise_total = 0
    for gold, pred in zip(gold_orderings, pred_orderings):
        gold_ranks = {item: i for i, item in enumerate(gold)}
        pred_ranks = {item: i for i, item in enumerate(pred)}
        items = set(gold) & set(pred)
        if len(items) < 3:
            continue
        g = [gold_ranks[i] for i in items]
        p = [pred_ranks[i] for i in items]
        all_spearman.append(spearman_rho([float(x) for x in g], [float(x) for x in p]))
        all_kendall.append(kendall_tau([float(x) for x in g], [float(x) for x in p]))
        # Pairwise accuracy
        items_list = list(items)
        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                a, b = items_list[i], items_list[j]
                gold_order_ab = gold_ranks.get(a, 0) < gold_ranks.get(b, 0)
                pred_order_ab = pred_ranks.get(a, 0) < pred_ranks.get(b, 0)
                pairwise_total += 1
                if gold_order_ab == pred_order_ab:
                    pairwise_correct += 1

    return {
        "n_cases": len(gold_orderings),
        "spearman_mean": sum(all_spearman) / len(all_spearman) if all_spearman else 0.0,
        "kendall_mean": sum(all_kendall) / len(all_kendall) if all_kendall else 0.0,
        "pairwise_accuracy": pairwise_correct / pairwise_total if pairwise_total > 0 else 0.0,
    }


# ── Routing-head experiment ───────────────────────────────────────────────────

def compute_routing_metrics(records: list[dict], cases: dict[str, Case]) -> dict:
    """Routing head: AUC of escalate vs confidence for predicting incorrectness.
    
    Validity guards:
    - Minimum N per class (correct and incorrect)
    - At least 2 distinct signal values
    - Signal range above epsilon
    - Show raw statistics (min/median/max/std/unique count)
    - Smoke corpus never issues keep/kill — only "insufficient evidence"
    """
    escalate_scores = []
    confidence_scores = []
    incorrect_labels = []  # 1 = incorrect
    for rec in records:
        case = cases.get(rec["case_id"])
        if not case:
            continue
        answers = rec.get("response", {}).get("answers", {})
        for qid, gold in case.gold.items():
            ans = answers.get(qid, {})
            act, esc = extract_routing(ans)
            conf = ans.get("confidence", 0.5)
            # Determine correctness
            qtype = case.formulation.questions.get(qid, {}).get("type", case.formulation.type)
            if qtype == "choice":
                ok, _, _ = extract_choice_correct(ans, gold)
            elif qtype == "noul":
                ok, _, _ = extract_noul_correct(ans, gold)
            elif qtype == "score":
                ok, _, _ = extract_score_correct(ans, gold)
            else:
                continue
            escalate_scores.append(esc)
            confidence_scores.append(conf)
            incorrect_labels.append(0 if ok else 1)

    n_total = len(incorrect_labels)
    n_correct = sum(1 for l in incorrect_labels if l == 0)
    n_incorrect = sum(1 for l in incorrect_labels if l == 1)
    
    # Basic statistics
    esc_unique = len(set(escalate_scores))
    esc_min = min(escalate_scores) if escalate_scores else 0.0
    esc_max = max(escalate_scores) if escalate_scores else 0.0
    esc_range = esc_max - esc_min
    esc_mean = sum(escalate_scores) / n_total if n_total > 0 else 0.0
    esc_std = (sum((e - esc_mean) ** 2 for e in escalate_scores) / n_total) ** 0.5 if n_total > 0 else 0.0
    
    conf_unique = len(set(confidence_scores))
    conf_min = min(confidence_scores) if confidence_scores else 0.0
    conf_max = max(confidence_scores) if confidence_scores else 0.0
    conf_range = conf_max - conf_min
    
    # Validity checks
    validity_issues = []
    if n_correct < 3:
        validity_issues.append(f"insufficient correct cases ({n_correct} < 3)")
    if n_incorrect < 3:
        validity_issues.append(f"insufficient incorrect cases ({n_incorrect} < 3)")
    if esc_unique < 2:
        validity_issues.append(f"escalate has no variance ({esc_unique} unique values)")
    if esc_range < 1e-6:
        validity_issues.append(f"escalate range too small ({esc_range:.2e} < 1e-6)")
    
    # Compute AUC only if valid
    if validity_issues:
        esc_auc = {"value": None, "reason": "; ".join(validity_issues)}
        conf_auc = {"value": None, "reason": "; ".join(validity_issues)}
        verdict = "insufficient_evidence"
    else:
        # AUC: higher score = more likely incorrect
        # For escalate: higher P(escalate) should predict incorrectness
        esc_auc = auc_roc(incorrect_labels, escalate_scores)
        # For confidence: LOWER confidence should predict incorrectness → invert
        conf_auc = auc_roc(incorrect_labels, [1 - c for c in confidence_scores])
        # Verdict only if we have enough evidence
        if n_total < 20:
            verdict = "insufficient_evidence"
        else:
            verdict = "keep" if esc_auc > 0.55 and esc_auc > conf_auc else "kill"

    return {
        "n_cases": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "escalate_auc": esc_auc,
        "confidence_auc": conf_auc,
        "incorrect_rate": n_incorrect / n_total if n_total > 0 else 0.0,
        "escalate_stats": {
            "min": esc_min,
            "max": esc_max,
            "mean": esc_mean,
            "std": esc_std,
            "range": esc_range,
            "unique_count": esc_unique,
        },
        "confidence_stats": {
            "min": conf_min,
            "max": conf_max,
            "range": conf_range,
            "unique_count": conf_unique,
        },
        "verdict": verdict,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_all_metrics(
    raw_path: Path,
    cases_path: Path,
    output_path: Path | None = None,
) -> dict:
    """Compute all metrics from raw.jsonl + cases.jsonl. Returns metrics dict."""
    cases = load_cases(cases_path)
    cases_by_id = {c.id: c for c in cases}
    dev_cases, lock_cases = split_cases(cases)
    dev_ids = {c.id for c in dev_cases}
    lock_ids = {c.id for c in lock_cases}

    # Load raw records
    records = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Partition by split
    dev_records = [r for r in records if r.get("split") == "dev"]
    lock_records = [r for r in records if r.get("split") == "lock"]

    # Group by family
    families = defaultdict(list)
    for rec in records:
        families[rec["family"]].append(rec)
    dev_families = defaultdict(list)
    for rec in dev_records:
        dev_families[rec["family"]].append(rec)
    lock_families = defaultdict(list)
    for rec in lock_records:
        lock_families[rec["family"]].append(rec)

    # Compute metrics per family, per split
    family_metrics = {}
    for fam in families:
        dev_recs = dev_families.get(fam, [])
        lock_recs = lock_families.get(fam, [])
        fam_metrics = {}

        # Determine primary type for this family
        sample_case = cases_by_id.get(families[fam][0]["case_id"])
        if not sample_case:
            continue
        ptype = sample_case.formulation.type

        for split_name, recs in [("dev", dev_recs), ("lock", lock_recs)]:
            if ptype == "choice":
                m = compute_choice_metrics(recs, cases_by_id)
            elif ptype == "noul":
                m = compute_noul_metrics(recs, cases_by_id)
            elif ptype == "score":
                m = compute_score_metrics(recs, cases_by_id)
            elif ptype == "multi_select":
                m = compute_multi_select_metrics(recs, cases_by_id)
            elif ptype == "rank":
                m = compute_rank_metrics(recs, cases_by_id)
            else:
                m = {"n_cases": len(recs), "type": ptype, "note": "no metrics implemented"}
            fam_metrics[split_name] = m

        family_metrics[fam] = fam_metrics

    # Routing head (across all cases)
    routing = compute_routing_metrics(records, cases_by_id)

    # Build output
    metrics = {
        "corpus_version": "v1",
        "n_total_cases": len(records),
        "n_dev": len(dev_records),
        "n_lock": len(lock_records),
        "families": family_metrics,
        "routing_head": routing,
    }

    # Write output
    if output_path is None:
        output_path = raw_path.parent / (raw_path.stem.replace("-raw", "") + "-metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote metrics to {output_path}")
    return metrics


def main():
    """CLI: python -m benchmarks.harness.metrics <raw.jsonl> <cases.jsonl>"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    compute_all_metrics(args.raw, args.cases, args.output)


if __name__ == "__main__":
    main()
