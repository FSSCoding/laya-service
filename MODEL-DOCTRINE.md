# Laya Operating Doctrine

**Status:** working mental model (inference, not proven internals). Derived from hands-on
characterisation probes (2026-09-29, `campaign/` scripts). Not a spec — a usage guide.
If the model is retrained or replaced, this document is stale.

---

## The core insight

The three primitives are not three unrelated capabilities. They are three different ways
of using the same underlying signal: **semantic compatibility between a state and a
proposition (or set of propositions)**.

| Primitive | Question it asks | Output form |
|-----------|-----------------|-------------|
| `noul` | "How compatible is this one proposition with the state?" | P(true) — binary |
| `choice` | "Which of these competing options is most compatible, relative to the others?" | softmax distribution — competitive |
| `score` | "Where does the state project onto this ordered set of anchors?" | expected position — ordinal |
| `batch` | (many of the above, one pass) | execution, not a separate capability |

This means the model is less a "tiny reasoner" and more a **structured semantic matching
machine**: it learns a compact shared representation, and the primitive type determines
how that representation is read out (binary / competitive / ordinal).

The practical consequence: **the quality of the answer depends on how well the question
is formulated relative to what is actually present in the state**, not on how much
background knowledge the model "should" have.

---

## Safe-usage rules

### Noul

- **Safe:** membership, classification, actionability, and other compatibility questions
  where the evidence is explicitly present in (or semantically very close to) the state.
  - "Does this ticket mention a failed backup?" → safe
  - "Is this customer on the enterprise plan?" → safe (if stated in the ticket)
  - "Does this require immediate action?" → safe (the state carries the urgency cues)
- **Unsafe:** propositions that require multi-step world-knowledge chains not encoded
  in the state text.
  - "Is a basketball spherical?" → the state says "basketball" but the model must
    resolve: basketball → object category → shape attribute → spherical. It won't.
  - "Was this email written by a human?" → requires inference about authorship from
    surface features. It won't.
- **Calibration warning:** 0.5 is NOT the ordinary true/false boundary, and there is no
  universal threshold. The useful cutoff is **domain- and formulation-dependent**: in the
  probes, easy positive entailments sat at ~0.7–0.9 while world-knowledge failures sat
  at ~0.0–0.3, but those are probe-specific numbers, not a general rule. Validate a
  threshold against your own domain slice before routing on it. Do not use noul for
  binary truth claims.

### Choice

- **Safe:** triage, tagging, routing, classification — anywhere the decision space is
  mutually distinguishable and each option is well described.
  - Triage: technical_support vs billing vs compliance → works if the descriptions
    are precise and non-overlapping.
  - Relevance tagging: spam vs newsletter vs transactional → works.
- **Unsafe:** overlapping categories, near-synonym options, or cases where two options
  describe adjacent responsibilities.
  - "technical_support: faults, outages, configuration, storage and backup failures"
    vs "storage_admin: disk capacity, volume management, and backup scheduling" →
    margin collapses to ~0.16 (ambiguous). The model cannot distinguish them because
    the options compete on the same semantic features.
- **Taxonomy is part of the inference mechanism.** The options you supply define the
  decision boundary. Improving an option's description doesn't just make the UI clearer —
  it changes the inference outcome. A precise, non-overlapping taxonomy IS the model
  performing correctly.
- **Distractor count:** up to 10 irrelevant distractors were handled well in the probe
  (winner prob barely moved between 3 and 10 options). The danger is not option count
  but relevant-but-overlapping competitors. Whether this holds beyond 10 options is
  untested.

### Score

- **Safe:** ordered severity/urgency/impact scales where every level is a distinct,
  coherent semantic landmark on the same axis.
  - "no impact / minor inconvenience / degraded service / major outage / complete
    outage" → monotonic, usable, distribution sharpens with severity.
- **Unsafe:** near-synonym ladders, scales where the axis is ambiguous, or levels that
  don't form a coherent continuum.
  - "bad / okay / decent / pretty good / good / really good / excellent" → the model
    reads this as a *quality* axis (good = positive), not a *severity* axis (high = bad).
    Result: a data-loss event scores as "bad" (lowest) at 0.97 while a minor typo
    scores higher. The signal **inverts**, not just degrades.
- **The ladder defines the latent axis.** This is the single most important rule for
  score. If the labels are semantically clean, evenly spaced, and all on the same
  dimension, you get a useful projection. If they're mushy, overlapping, or on the
  wrong dimension, the output is meaningless or actively wrong.

---

## Where Laya is genuinely valuable

Anything where:
1. The **state itself contains the evidence** (no hidden knowledge required).
2. The **decision space is carefully designed** (mutually exclusive, well-described
   options or a coherent ordered scale).
3. The task is a **judgment, not a proof** (triage, routing, tagging, prioritisation,
   actionability, policy gates, relevance).

Concrete fits: ticket triage, email classification, urgency scoring, policy gates,
action-item detection, customer-segment routing, content tagging, escalation decisions.

## Where not to trust it

- Hidden factual inference ("is this product safe to eat?")
- Commonsense chains ("if the backup failed, the data might be lost")
- Nuanced causality ("did the upgrade *cause* the failure?")
- Long reasoning paths (multi-step derivations)
- Anything where the answer depends on knowledge **not actually present in the state**

---

## Probe evidence

Characterisation runs (2026-09-29, disposable scripts in `campaign/`):

| Script | What it tested | Key finding |
|--------|---------------|-------------|
| `noul-probe.py` | lexical → synonym → category → attribute → false-attr, 2 states | Lexical strongest (0.87–0.94); category works (0.44–0.74); attribute entailment failed in the probes (0.06–0.27); false controls stay low (0.00–0.05) |
| `choice-probe.py` | fixed winner, 4 distractor conditions | Winner prob swings 0.58→0.83 with distractor quality; near-synonym collapses margin to 0.16; many irrelevant options don't hurt |
| `score-probe.py` | 2 ladders × 5 severity levels | Clean ladder: monotonic, distribution sharpens. Mushy ladder: signal inverts (data loss → "bad" at 0.97) |

Single-run numbers. The **patterns** are the durable part, not the exact values.
Re-run the scripts if you need to check whether a specific formulation generalises.

---

## Question-formulation checklist

Before sending a question to Laya, verify:

- [ ] **Is the evidence in the state?** If the answer requires knowledge not present
      in the state text, Laya will guess, not know.
- [ ] **Is the decision space clean?** Options mutually distinguishable? Score levels
      on one coherent axis, semantically distinct from each other?
- [ ] **Am I asking for judgment or proof?** Laya judges. It doesn't prove. If you
      need a proof, you need a prover.
- [ ] **For noul: am I crossing the 0.5 boundary meaningfully?** If the expected answer
      is ~0.7 and the threshold is 0.5, you have margin. If it's ~0.52, you're in the
      noise.
- [ ] **For choice: do my options overlap?** If two options describe adjacent
      responsibilities, expect a low-margin near-tie. That's the model telling you
      the taxonomy is ambiguous, not that it's failing.
- [ ] **For score: does my ladder make sense as an axis?** Read it aloud. Does each
      level mean something distinct? Are they all measuring the same thing? If "bad"
      and "excellent" are your endpoints, are you measuring quality or severity? Be
      explicit.
