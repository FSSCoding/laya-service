# ⚠️ SMOKE / PIPELINE VALIDATION — NOT MODEL CHARACTERISATION

**This report is from a smoke corpus (33 cases). It validates the harness pipeline, not the model.**

**Do not draw conclusions about Laya's capabilities from this report.**

---

**Generated:** 2026-09-29 10:13 UTC

## Provenance

- **Corpus version:** v1
- **Total cases:** 33 (dev: 20, lock: 13)

## Model Quality (per family, per split)

### choice/2-option

**DEV split** (n=1)

- Top-1 accuracy: **100.0%**
- P(gold) mean: 0.905
- Confidence mean: 0.547
- ECE: 0.453
- Option-count sensitivity: 2=100.0%, 3-5=n/a, 6-10=n/a, 11-20=n/a, 25+=n/a

**LOCK split** (n=0)

_No cases in this split._

### choice/multiclass

**DEV split** (n=1)

- Top-1 accuracy: **100.0%**
- P(gold) mean: 0.574
- Confidence mean: 0.297
- ECE: 0.703
- Option-count sensitivity: 2=n/a, 3-5=100.0%, 6-10=n/a, 11-20=n/a, 25+=n/a

**LOCK split** (n=1)

- Top-1 accuracy: **100.0%**
- P(gold) mean: 0.736
- Confidence mean: 0.562
- ECE: 0.438
- Option-count sensitivity: 2=n/a, 3-5=n/a, 6-10=100.0%, 11-20=n/a, 25+=n/a

### choice/overlapping

**DEV split** (n=1)

- Top-1 accuracy: **100.0%**
- P(gold) mean: 0.968
- Confidence mean: 0.852
- ECE: 0.148
- Option-count sensitivity: 2=n/a, 3-5=100.0%, 6-10=n/a, 11-20=n/a, 25+=n/a

**LOCK split** (n=0)

_No cases in this split._

### choice/stress

**DEV split** (n=0)

_No cases in this split._

**LOCK split** (n=1)

- Top-1 accuracy: **100.0%**
- P(gold) mean: 0.947
- Confidence mean: 0.935
- ECE: 0.065
- Option-count sensitivity: 2=n/a, 3-5=n/a, 6-10=n/a, 11-20=n/a, 25+=100.0%

### consistency/irrelevant-text

**DEV split** (n=0)

_No cases in this split._

**LOCK split** (n=2)

- AUC: **N/A (AUC: only one class represented)**
- Brier score: 0.528
- ECE: N/A (ECE: insufficient N (2 < 10))
- Accuracy @0.5: 50.0%

### consistency/option-reorder

**DEV split** (n=2)

- Top-1 accuracy: **100.0%**
- P(gold) mean: 0.773
- Confidence mean: 0.228
- ECE: 0.772
- Option-count sensitivity: 2=100.0%, 3-5=n/a, 6-10=n/a, 11-20=n/a, 25+=n/a

**LOCK split** (n=0)

_No cases in this split._

### inversion/choice-vs-noul

**DEV split** (n=1)

- Top-1 accuracy: **100.0%**
- P(gold) mean: 0.917
- Confidence mean: 0.687
- ECE: 0.313
- Option-count sensitivity: 2=n/a, 3-5=100.0%, 6-10=n/a, 11-20=n/a, 25+=n/a

**LOCK split** (n=1)

- Top-1 accuracy: **100.0%**
- P(gold) mean: 0.811
- Confidence mean: 0.451
- ECE: 0.549
- Option-count sensitivity: 2=n/a, 3-5=100.0%, 6-10=n/a, 11-20=n/a, 25+=n/a

### multi_select/disjoint

**DEV split** (n=1)

- F1 @0.5: **1.000** (P=1.000, R=1.000)

**LOCK split** (n=0)

_No cases in this split._

### multi_select/none

**DEV split** (n=0)

_No cases in this split._

**LOCK split** (n=1)

- F1 @0.5: **0.000** (P=0.000, R=0.000)

### multi_select/overlapping

**DEV split** (n=1)

- F1 @0.5: **0.857** (P=0.750, R=1.000)

**LOCK split** (n=0)

_No cases in this split._

### noul/adversarial

**DEV split** (n=0)

_No cases in this split._

**LOCK split** (n=1)

- AUC: **N/A (AUC: only one class represented)**
- Brier score: 0.000
- ECE: N/A (ECE: insufficient N (1 < 10))
- Accuracy @0.5: 100.0%

### noul/ambiguous

**DEV split** (n=1)

- AUC: **N/A (AUC: only one class represented)**
- Brier score: 0.163
- ECE: N/A (ECE: insufficient N (1 < 10))
- Accuracy @0.5: 100.0%

**LOCK split** (n=0)

_No cases in this split._

### noul/factual

**DEV split** (n=1)

- AUC: **N/A (AUC: only one class represented)**
- Brier score: 0.059
- ECE: N/A (ECE: insufficient N (1 < 10))
- Accuracy @0.5: 100.0%

**LOCK split** (n=0)

_No cases in this split._

### noul/symmetry

**DEV split** (n=1)

- AUC: **N/A (AUC: only one class represented)**
- Brier score: 0.029
- ECE: N/A (ECE: insufficient N (1 < 10))
- Accuracy @0.5: 100.0%

**LOCK split** (n=1)

- AUC: **N/A (AUC: only one class represented)**
- Brier score: 0.000
- ECE: N/A (ECE: insufficient N (1 < 10))
- Accuracy @0.5: 100.0%

### ood/nonsense

**DEV split** (n=1)

- Top-1 accuracy: **0.0%**
- P(gold) mean: 0.163
- Confidence mean: 0.359
- ECE: 0.359
- Option-count sensitivity: 2=0.0%, 3-5=n/a, 6-10=n/a, 11-20=n/a, 25+=n/a

**LOCK split** (n=0)

_No cases in this split._

### ood/wrong-domain

**DEV split** (n=1)

- AUC: **N/A (AUC: only one class represented)**
- Brier score: 0.000
- ECE: N/A (ECE: insufficient N (1 < 10))
- Accuracy @0.5: 100.0%

**LOCK split** (n=0)

_No cases in this split._

### rank/3-item

**DEV split** (n=1)

- Spearman ρ (mean): **-0.500**
- Kendall τ (mean): -0.333
- Pairwise accuracy: 33.3%

**LOCK split** (n=0)

_No cases in this split._

### rank/5-item

**DEV split** (n=0)

_No cases in this split._

**LOCK split** (n=1)

- Spearman ρ (mean): **-1.000**
- Kendall τ (mean): -1.000
- Pairwise accuracy: 0.0%

### rank/near-tie

**DEV split** (n=1)

- Spearman ρ (mean): **0.500**
- Kendall τ (mean): 0.333
- Pairwise accuracy: 66.7%

**LOCK split** (n=0)

_No cases in this split._

### score/10-level

**DEV split** (n=0)

_No cases in this split._

**LOCK split** (n=1)

- MAE: **5.00**
- Exact hit: 0.0%
- Spearman ρ: N/A (Spearman: insufficient N (1 < 3))
- Kendall τ: N/A (Kendall: insufficient N (1 < 3))

### score/3-level

**DEV split** (n=1)

- MAE: **1.00**
- Exact hit: 0.0%
- Spearman ρ: N/A (Spearman: insufficient N (1 < 3))
- Kendall τ: N/A (Kendall: insufficient N (1 < 3))

**LOCK split** (n=0)

_No cases in this split._

### score/4-level

**DEV split** (n=1)

- MAE: **0.00**
- Exact hit: 100.0%
- Spearman ρ: N/A (Spearman: insufficient N (1 < 3))
- Kendall τ: N/A (Kendall: insufficient N (1 < 3))

**LOCK split** (n=0)

_No cases in this split._

### score/7-level

**DEV split** (n=0)

_No cases in this split._

**LOCK split** (n=1)

- MAE: **0.00**
- Exact hit: 100.0%
- Spearman ρ: N/A (Spearman: insufficient N (1 < 3))
- Kendall τ: N/A (Kendall: insufficient N (1 < 3))

### state/long

**DEV split** (n=1)

- Top-1 accuracy: **100.0%**
- P(gold) mean: 0.892
- Confidence mean: 0.624
- ECE: 0.376
- Option-count sensitivity: 2=n/a, 3-5=100.0%, 6-10=n/a, 11-20=n/a, 25+=n/a

**LOCK split** (n=0)

_No cases in this split._

### state/noisy

**DEV split** (n=1)

- Top-1 accuracy: **100.0%**
- P(gold) mean: 0.658
- Confidence mean: 0.307
- ECE: 0.693
- Option-count sensitivity: 2=n/a, 3-5=100.0%, 6-10=n/a, 11-20=n/a, 25+=n/a

**LOCK split** (n=0)

_No cases in this split._

### state/short

**DEV split** (n=0)

_No cases in this split._

**LOCK split** (n=1)

- AUC: **N/A (AUC: only one class represented)**
- Brier score: 0.531
- ECE: N/A (ECE: insufficient N (1 < 10))
- Accuracy @0.5: 0.0%

## Routing Head Experiment

- **N cases:** 25 (correct: 20, incorrect: 5)
- **Escalate AUC:** N/A (escalate has no variance (1 unique values); escalate range too small (0.00e+00 < 1e-6))
- **Confidence AUC:** N/A (escalate has no variance (1 unique values); escalate range too small (0.00e+00 < 1e-6))
- Incorrect rate: 20.0%
- Escalate stats: min=0.000, max=0.000, mean=0.000, std=0.000, range=0.000, unique=1
- Confidence stats: min=0.212, max=0.935, range=0.723, unique=16
- **Verdict:** insufficient_evidence

## Envelope Summary (green/yellow/red per family)

_Pass bars not yet defined — run the benchmark, observe the distribution, then freeze the bars on the dev split and evaluate on lock._
