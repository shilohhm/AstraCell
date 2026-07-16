# AstraCell — a battery diagnostic that knows what it can't see

A from-first-principles study of a question most battery-diagnostic tools skip: *before*
asking "which fault is this?", ask "is that question answerable from the data I actually
have?" AstraCell answers the second question, refuses the ones it cannot, and — where a
refusal can be undone — computes what measurement would change its mind.

Everything below is measured and reproducible; every number links to the command that
regenerates it. The detailed account is in the [technical report](TECHNICAL_REPORT.md).

---

## The problem

A production battery-management system does not measure what you want to diagnose. It reads
one voltage per cell, but only a handful of temperature sensors for ~96 cells, and current
only at the pack terminals. So a per-cell *thermal* fault is, for most cells, simply not
present in the telemetry. "Highlight the faulty cell on a 3D pack map" is undecidable for
those faults — and most projects render the map anyway, reporting confidence they have not
earned.

## The observability insight

The Fisher Information Matrix of the pack under its real sensor topology yields a
Cramér–Rao lower bound on the variance of *every* unbiased estimator — not one algorithm.
If the bound says a 40% cooling fault sits at 1.2σ, no detector will find it, and the
honest output is grey. The grey cells fall out of the bound, not out of a distance rule.

The deeper insight is about being wrong: **more data can make a structurally wrong answer
more certain.** A model that is wrong in a way the data cannot expose grows *more* confident
as you feed it more samples. A diagnostic that does not check for this is, given enough
data, certain and wrong.

## What I built

An identifiability engine (Fisher/CRLB) with two ordered gates — separability (VIF) then
detectability (SNR) — that returns DIAGNOSE, WEAK, or one of three refusals, each with a
recommendation where one exists (instrument a cell, or excite it harder). Around it: a
model-bias gate that prices the observer's own structural error, an estimator, a Monte
Carlo calibration harness, an external PyBaMM plant, a positive control, and — in v0.6–v0.8 — a
real-cell run across all eight cells of the Oxford dataset — on every cell the first-order ECM is
directionally wrong and AstraCell refuses it, a refusal a second-order observer leaves unchanged
(v0.8). The core is numpy-only; 246 tests assert theorems, not observed numbers. Built in five
science passes (v0.0 → v0.4), since packaged and taken to eight measured cells.

## How I know it works

This is the part I care most about, because the failure mode I built against is confident
wrongness. The evidence is organised into three tiers of validation, and I do not let a
result at one tier license a claim at a higher one.

**Tier 1 — internal.** The physics identities hold to machine precision (energy balance to
1.8×10⁻¹⁵). The Cramér–Rao bound is *attained*, not just asserted: under a matched model
the estimator's interval coverage tracks nominal across 50–99% confidence — the first
check in the project that the bound is reachable. Adding a sensor never increases the CRLB
(a theorem, tested). And the central result: fitting a first-order equivalent-circuit model
to a plant with unmodelled diffusion **manufactures an 18.5% capacity loss from a 5% fault**,
and *more data makes it worse* — replicate 10 000× and the reported confidence climbs to
14 611σ while the credible confidence sits fixed at 6.49σ. The structural bias is invariant
to replication, bit-for-bit. Calibration turns this into a frequency: under mismatch the
variance-only interval covers the truth 0% of the time, and the model-bias gate drops the
harmful-overclaim rate on capacity from 100% to 0% — by refusing. Calibration made several
of my headline numbers *worse*, which is the only kind of diagnostic worth trusting.

**Tier 2 — an independently developed external simulator.** I then swapped my own plant for
PyBaMM, an electrochemical simulator I did not implement, whose mismatch I did not design.
On a perfectly healthy cell the observer reports a phantom −67.6% ± 0.145% capacity loss
(466σ from zero); a self-consistency control proves the collapse is PyBaMM's physics, not my
harness (coverage nominal to within 0.011). The gate refuses the phantom. Then the positive
control: with a healthy baseline, the same machinery recovers a real injected fault at
+20.0000% with nominal coverage, and still refuses a real degradation it cannot express —
by a one-percent margin I report rather than round away. Along the way it caught my own bug:
v0.3's external gate turned out to refuse *everything* (its statistic was pure noise), and
only a positive control could reveal that. Diagnosis is not detection — I score a true
positive only when the interval actually covers the truth.

## What it cannot do

Equally important, and stated plainly rather than buried.

**No physical battery validates any of this.** v0.6 first ran the observer against a real measured
cell and v0.7 widened it to all eight (Oxford Cell1–8) — and it refused every one: the first-order
ECM's capacity estimate came back wrong in *sign* on seven of eight (a +10.5% "gain" against Cell1's
measured −24.2% fade, ≈1150σ from truth), REFUSE_MODEL_BIAS on all 104 scored ages in both OCV modes
(208/208). v0.8 checked whether a *second-order* observer would rescue the estimate; it does not — a
fixed second RC branch changes 0/208 verdicts (largest change 10⁻¹⁴) — so the refusal is not a
model-order artefact, though *fitting* the extra dynamics (v0.9) is untested. That is **contact, not
validation** — eight cells but one chemistry, no fault detected, the ECM confirmed nowhere; the
refusal is the honest result, not a green light.
Every Tier 1/2 result above is still conditional on models that have otherwise not touched a cell —
the OCV curves are stand-ins, so every SNR is a statement about a model, not a battery. The Cramér–Rao bound is variance-only and blind to
model bias by construction; the screen I use to catch that bias externally is a *screen, not
a bound* — on the one case where I know the truth, it captures only 31% of the error it
warns about, and a bias three times smaller would have slipped through. My internal mismatch
plant is four hand-chosen terms, so it is a lower bound on how wrong the observer is, not a
measurement of it. The positive control's baseline is the identical simulation, which no
real workshop can supply, so its detection rates are upper bounds. There is no fault
classifier, no residual bank, no ramped fault onsets, no pack-scale electrochemistry; the
2σ/5σ thresholds and the 2D thermal geometry are conventions and caricatures. And to be
unambiguous: **there is no EV-level validation and no evidence of safety-critical deployment
readiness.** This is a research scaffold for the identifiability question.

## What I would do next

Only for the faults this machinery certifies as answerable and trustworthy. First and above
all, **more chemistries, and a better observer for them.** v0.6 ran one cell and v0.7 ran all eight
(Oxford Cell1–8); the first-order ECM came back directionally wrong on every one and AstraCell refused
every age — the prediction that a real cell would mismatch the ECM harder than PyBaMM did, now measured
eight times over. v0.8 ruled out the obvious escape — a *fixed* second-order observer changes no
verdict (0/208), so depth alone is not the fix — which points the honest next step at *fitting* the
added dynamics (v0.9), a 4→6-parameter identifiability problem, alongside a same-day baseline, other
chemistries and formats, and an observer that can express real OCV drift — to learn whether the
refusal ever becomes a trustworthy diagnosis. That is the only path
that moves anything to Tier 3, and no simulation stands in for it. Then DFN with degradation
submodels and injected capacity fade; ramped and concurrent faults; and, built last rather
than first, a classifier — on the faults the identifiability layer has already certified.
Building the classifier first would reproduce exactly the failure mode this project exists
to refuse.

---

**Verify any of this yourself:** [reproduction commands](REPRODUCIBILITY.md) ·
[claims → evidence](CLAIMS.md) · [full technical report](TECHNICAL_REPORT.md) ·
[what did not work](WHAT_DID_NOT_WORK.md).
