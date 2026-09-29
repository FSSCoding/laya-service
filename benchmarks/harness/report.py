"""Report generator: markdown + JSON from metrics.json + systems.json + provenance.

Supports --diff mode for regression detection against a baseline.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _fmt(val: Any, fmt: str = ".3f", na_label: str = "N/A") -> str:
    """Format a metric value that may be a float, None, or a dict with value/reason."""
    if val is None:
        return na_label
    if isinstance(val, dict):
        if val.get("value") is None:
            reason = val.get("reason", "unknown")
            return f"{na_label} ({reason})"
        val = val["value"]
    if isinstance(val, (int, float)):
        return f"{val:{fmt}}"
    return str(val)


def generate_report(
    metrics_path: Path,
    systems_path: Path | None = None,
    output_path: Path | None = None,
    baseline_path: Path | None = None,
) -> str:
    """Generate markdown report from metrics + systems. Returns report text."""
    with open(metrics_path) as f:
        metrics = json.load(f)

    systems = {}
    if systems_path and systems_path.exists():
        with open(systems_path) as f:
            systems = json.load(f)

    # Build report
    lines = []
    
    # Detect smoke/pipeline-validation runs
    corpus_version = metrics.get("corpus_version", "unknown")
    n_total = metrics.get("n_total_cases", 0)
    is_smoke = "smoke" in corpus_version.lower() or n_total < 100
    
    if is_smoke:
        lines.append("# ⚠️ SMOKE / PIPELINE VALIDATION — NOT MODEL CHARACTERISATION\n")
        lines.append(f"**This report is from a smoke corpus ({n_total} cases). It validates the harness pipeline, not the model.**\n")
        lines.append("**Do not draw conclusions about Laya's capabilities from this report.**\n")
        lines.append("---\n")
    else:
        lines.append("# Laya Evaluation Report\n")
    
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n")

    # Header: provenance
    lines.append("## Provenance\n")
    prov = metrics.get("provenance", systems.get("provenance", {}))
    lines.append(f"- **Corpus version:** {metrics.get('corpus_version', 'unknown')}")
    lines.append(f"- **Total cases:** {metrics.get('n_total_cases', 0)} (dev: {metrics.get('n_dev', 0)}, lock: {metrics.get('n_lock', 0)})")
    if prov:
        lines.append(f"- **Service version:** {prov.get('service_version', 'unknown')}")
        lines.append(f"- **Git commit:** {prov.get('service_git_commit', 'unknown')[:8]}")
        lines.append(f"- **Provider:** {prov.get('provider', 'unknown')}")
        lines.append(f"- **Package version:** {prov.get('package_version', 'unknown')}")
    lines.append("")

    # Per-family quality tables
    lines.append("## Model Quality (per family, per split)\n")
    families = metrics.get("families", {})
    for fam, splits in sorted(families.items()):
        lines.append(f"### {fam}\n")
        for split_name, m in splits.items():
            lines.append(f"**{split_name.upper()} split** (n={m.get('n_cases', 0)})\n")
            if m.get("n_cases", 0) == 0:
                lines.append("_No cases in this split._\n")
                continue
            # Type-specific metrics
            if "top1_accuracy" in m:
                lines.append(f"- Top-1 accuracy: **{_fmt(m['top1_accuracy'], '.1%')}**")
                lines.append(f"- P(gold) mean: {_fmt(m.get('p_gold_mean', 0))}")
                lines.append(f"- Confidence mean: {_fmt(m.get('confidence_mean', 0))}")
                lines.append(f"- ECE: {_fmt(m.get('ece', 0))}")
                opt_sens = m.get("option_count_sensitivity", {})
                if opt_sens:
                    parts = []
                    for k, v in opt_sens.items():
                        if v is not None:
                            parts.append(f"{k}={_fmt(v, '.1%')}")
                        else:
                            parts.append(f"{k}=n/a")
                    lines.append(f"- Option-count sensitivity: {', '.join(parts)}")
            elif "auc" in m:
                lines.append(f"- AUC: **{_fmt(m['auc'])}**")
                lines.append(f"- Brier score: {_fmt(m.get('brier', 0))}")
                lines.append(f"- ECE: {_fmt(m.get('ece', 0))}")
                lines.append(f"- Accuracy @0.5: {_fmt(m.get('accuracy_at_0.5', 0), '.1%')}")
            elif "mae" in m:
                lines.append(f"- MAE: **{_fmt(m['mae'], '.2f')}**")
                lines.append(f"- Exact hit: {_fmt(m.get('exact_hit', 0), '.1%')}")
                lines.append(f"- Spearman ρ: {_fmt(m.get('spearman', 0))}")
                lines.append(f"- Kendall τ: {_fmt(m.get('kendall', 0))}")
            elif "thresholds" in m and "precision" in m["thresholds"].get("0.5", {}):
                # multi_select
                t05 = m["thresholds"]["0.5"]
                lines.append(f"- F1 @0.5: **{_fmt(t05.get('f1', 0))}** (P={_fmt(t05.get('precision', 0))}, R={_fmt(t05.get('recall', 0))})")
            elif "spearman_mean" in m:
                # rank
                lines.append(f"- Spearman ρ (mean): **{_fmt(m['spearman_mean'])}**")
                lines.append(f"- Kendall τ (mean): {_fmt(m.get('kendall_mean', 0))}")
                lines.append(f"- Pairwise accuracy: {_fmt(m.get('pairwise_accuracy', 0), '.1%')}")
            else:
                lines.append(f"_Metrics: {list(m.keys())}_")
            lines.append("")

    # Routing head
    lines.append("## Routing Head Experiment\n")
    routing = metrics.get("routing_head", {})
    if routing.get("insufficient_data"):
        lines.append("_Insufficient data for routing-head analysis._\n")
    else:
        lines.append(f"- **N cases:** {routing.get('n_cases', 0)} (correct: {routing.get('n_correct', 0)}, incorrect: {routing.get('n_incorrect', 0)})")
        lines.append(f"- **Escalate AUC:** {_fmt(routing.get('escalate_auc', 0))}")
        lines.append(f"- **Confidence AUC:** {_fmt(routing.get('confidence_auc', 0))}")
        lines.append(f"- Incorrect rate: {_fmt(routing.get('incorrect_rate', 0), '.1%')}")
        
        # Show raw statistics
        esc_stats = routing.get("escalate_stats", {})
        if esc_stats:
            lines.append(f"- Escalate stats: min={_fmt(esc_stats.get('min', 0))}, max={_fmt(esc_stats.get('max', 0))}, mean={_fmt(esc_stats.get('mean', 0))}, std={_fmt(esc_stats.get('std', 0))}, range={_fmt(esc_stats.get('range', 0))}, unique={esc_stats.get('unique_count', 0)}")
        
        conf_stats = routing.get("confidence_stats", {})
        if conf_stats:
            lines.append(f"- Confidence stats: min={_fmt(conf_stats.get('min', 0))}, max={_fmt(conf_stats.get('max', 0))}, range={_fmt(conf_stats.get('range', 0))}, unique={conf_stats.get('unique_count', 0)}")
        
        verdict = routing.get("verdict", "unknown")
        if verdict == "keep":
            lines.append(f"- **Verdict:** keep — escalate predicts failure better than confidence")
        elif verdict == "kill":
            lines.append(f"- **Verdict:** kill — escalate has no predictive value")
        else:
            lines.append(f"- **Verdict:** {verdict}")
    lines.append("")

    # Systems benchmark
    if systems:
        lines.append("## Systems Benchmark\n")
        warm = systems.get("warm_single", {})
        if warm:
            lines.append("### Warm single-call latency\n")
            for shape, stats in warm.items():
                lines.append(f"- {shape}: {stats.get('median', 0):.1f} ms (p95: {stats.get('p95', 0):.1f} ms)")
            lines.append("")

        cold = systems.get("cold_reload", {})
        if cold:
            lines.append(f"### Cold reload: {cold.get('median', 0):.1f} ms (range: {cold.get('min', 0):.1f}–{cold.get('max', 0):.1f} ms)\n")

        conc = systems.get("concurrency", {})
        if conc:
            lines.append("### Concurrency sweep\n")
            for level, stats in conc.items():
                lines.append(f"- {level}: {stats.get('median', 0):.1f} ms/req, {stats.get('throughput_rps', 0):.1f} req/s")
            lines.append("")

        cache = systems.get("cache", {})
        if cache:
            lines.append(f"### Cache: hit {cache.get('hit', {}).get('median', 0):.1f} ms, miss {cache.get('miss', {}).get('median', 0):.1f} ms\n")

        opts = systems.get("option_count", {})
        if opts:
            lines.append("### Latency vs option count\n")
            for key, stats in opts.items():
                tokens = stats.get("input_tokens", 0)
                lines.append(f"- {key}: {stats.get('median', 0):.1f} ms ({tokens} tokens)")
            lines.append("")

        vram_before = systems.get("vram_before", {})
        vram_after = systems.get("vram_after", {})
        if vram_before:
            lines.append(f"### VRAM: before {vram_before.get('vram_used_mb', 0):.0f} MB, after {vram_after.get('vram_used_mb', 0):.0f} MB\n")

    # Diff mode
    if baseline_path and baseline_path.exists():
        lines.append("## Regression Diff (vs baseline)\n")
        with open(baseline_path) as f:
            baseline = json.load(f)
        diff_lines = _compute_diff(metrics, baseline)
        if diff_lines:
            lines.extend(diff_lines)
        else:
            lines.append("_No significant changes detected._")
        lines.append("")

    # Envelope summary
    lines.append("## Envelope Summary (green/yellow/red per family)\n")
    lines.append("_Pass bars not yet defined — run the benchmark, observe the distribution, then freeze the bars on the dev split and evaluate on lock._\n")

    report = "\n".join(lines)

    # Write output
    if output_path is None:
        output_path = metrics_path.parent / (metrics_path.stem.replace("-metrics", "") + "-report.md")
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Wrote report to {output_path}")
    return report


def _compute_diff(current: dict, baseline: dict) -> list[str]:
    """Compute per-family metric deltas between current and baseline."""
    lines = []
    curr_families = current.get("families", {})
    base_families = baseline.get("families", {})

    for fam in sorted(set(curr_families) | set(base_families)):
        curr_fam = curr_families.get(fam, {})
        base_fam = base_families.get(fam, {})
        for split in ["dev", "lock"]:
            curr_m = curr_fam.get(split, {})
            base_m = base_fam.get(split, {})
            if not curr_m or not base_m:
                continue
            # Compare key metrics
            for key in ["top1_accuracy", "auc", "mae", "spearman"]:
                if key in curr_m and key in base_m:
                    delta = curr_m[key] - base_m[key]
                    if abs(delta) > 0.01:  # >1% change
                        direction = "↑" if delta > 0 else "↓"
                        lines.append(f"- {fam}/{split} {key}: {base_m[key]:.3f} → {curr_m[key]:.3f} ({direction}{abs(delta):.3f})")

    # Routing head diff
    curr_r = current.get("routing_head", {})
    base_r = baseline.get("routing_head", {})
    if curr_r and base_r:
        for key in ["escalate_auc", "confidence_auc"]:
            if key in curr_r and key in base_r:
                delta = curr_r[key] - base_r[key]
                if abs(delta) > 0.02:
                    direction = "↑" if delta > 0 else "↓"
                    lines.append(f"- routing/{key}: {base_r[key]:.3f} → {curr_r[key]:.3f} ({direction}{abs(delta):.3f})")

    return lines


def main():
    """CLI: python -m benchmarks.harness.report <metrics.json> [--systems systems.json] [--diff baseline.json]"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--systems", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--diff", type=Path, default=None, help="Baseline metrics.json for regression diff")
    args = parser.parse_args()

    generate_report(args.metrics, args.systems, args.output, args.diff)


if __name__ == "__main__":
    main()
