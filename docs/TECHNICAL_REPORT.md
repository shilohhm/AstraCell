# AstraCell — Technical Report

The canonical, detailed account of what AstraCell is, what it proves, and how each claim
was tested. The [README](../README.md) is the five-minute version; this is where the
evidence lives in full. [LIMITATIONS.md](../LIMITATIONS.md) is the caveat ledger,
[CLAIMS.md](CLAIMS.md) the claim-to-evidence map, and the specialised docs
([CALIBRATION](CALIBRATION.md), [EXTERNAL_PLANT](EXTERNAL_PLANT.md),
[POSITIVE_CONTROL](POSITIVE_CONTROL.md)) the deep dives. Every number below is regenerated
by a command in [REPRODUCIBILITY.md](REPRODUCIBILITY.md); none is quoted from memory.

---

## The three tiers of validation

Read this first. It is the distinction most battery-diagnostic work blurs, and the one
this report is organised around. A result at one tier does not license a claim at a higher
one.

| Tier | Meaning | Where AstraCell stands |
|---|---|---|
| **1 — internal self-consistency and synthetic experiments** | Demonstrated within AstraCell's own models, and by theorems about the estimator | Extensive: identifiability, calibration, model-bias accounting all measured |
| **2 — independently developed external simulator** | Tested against PyBaMM — an electrochemical simulator AstraCell did not implement, whose mismatch it did not design | The phantom-fault refusal and the positive control |
| **3 — physical battery validation** | A measured cell, a real fault, a real dataset | **Contact, not validation.** v0.6 ran one real cell (Oxford Cell1) and the ECM refused every age; no validation, one cell of eight. Stated, not hidden |

Everything in §4 is labelled by tier. Everything in §6 is why Tier 3 has no *validation* — one cell of contact (§7.1) does not make one.

---

## 1. The problem

A production battery-management system does not measure what you want to diagnose. The
sensor budget is radically asymmetric:

| Quantity | Real coverage | Consequence |
|---|---|---|
| Cell voltage | one per cell, ~1 mV | per-cell voltage faults **are** identifiable |
| Temperature | **4–12 sensors for ~96 cells**, ±0.5–1 K | per-cell thermal faults mostly **are not** |
| Current | **pack-level only**, one shunt | per-cell current is inferred, never measured |

So "highlight the faulty cell on a 3D pack map" is, for thermal faults, *undecidable* from
real telemetry. Most battery-diagnostic projects render the map anyway — reporting
confidence without first establishing whether the fault was ever observable. A classifier
built on that footing is confident exactly where it should be silent.

## 2. The insight

Two ideas, in order of importance.

**More data can make a structurally wrong answer more certain.** A wrong model does not
merely add error you can average away; it converges to the wrong answer and then grows
*more confident about it* the more data you feed it. Any diagnostic that reports a
shrinking confidence interval without asking whether its model is right will, given enough
data, be certain and wrong. AstraCell measures this directly (§4, Tier 1) and gates
against it.

**Identifiability is the logically prior question.** Before "which fault is it?" comes "is
that question answerable from this data?" AstraCell answers the second. The tool is the
**Fisher Information Matrix** of the pack's parameters under the actual sensor topology and
the **Cramér–Rao lower bound** it implies — a bound on the variance of *every* unbiased
estimator, not of one algorithm. When the bound says a 40% cooling fault sits at 1.2σ, no
detector anyone writes will find it, and the honest thing to render is grey. The grey cells
fall out of the bound, not out of a distance-to-sensor heuristic.

The two ideas are two kinds of error: **variance** (how finely a *correct* model could be
pinned down — the CRLB) and **bias** (how far a *wrong* model lands from the truth however
clean the data). Keeping them apart is the whole discipline.

## 3. What AstraCell does

Four things, and deliberately not a fifth (there is no fault classifier — that is the
point):

1. **Identifiability.** Compute `FIM = SᵀΣ⁻¹S` over every cell's parameters, then the CRLB,
   the VIF (separability), and the SNR (detectability). Two gates in order: **isolation**
   (`VIF > 10` ⇒ confounded) then **detection** (`≥5σ` diagnose, `2–5σ` weak, `<2σ`
   refuse). Isolation is first because two parameters can be jointly well-determined while
   individually unidentifiable.
2. **Model bias.** Against a higher-fidelity plant, compute the structural bias
   `b = FIM⁻¹SᵀΣ⁻¹r` the observer's wrong model incurs, and refuse
   (`REFUSE_MODEL_BIAS`) when it dwarfs the signal.
3. **Empirical coverage.** Wrap an estimator around the oracle and run Monte Carlo with a
   known injected truth, to check the intervals mean what they claim.
4. **Abstention.** Three refusal gates, each carrying a recommendation where one exists
   (instrument a cell, or excite it harder). Refusal is the product.

A sensor topology enters only as a *row mask* over `S`, so counterfactual sensor placement
costs a matrix slice, not a re-simulation. Full derivations are in the
[methods appendix](#methods-appendix).

## 4. How it was tested

The longest section, by design. Organised by validation tier.

### Tier 1 — internal self-consistency and synthetic experiments

Demo pack throughout: a 4×8 grid, 32 cells, 32 voltage + 4 temperature channels
(thermocouples on cells 4/12/20/28) + one current shunt, 1 mV / 0.5 K noise, 1200 s of
1.0C pulse excitation. Reproduce with `python examples/01_first_demo.py` unless noted.

**The physics is correct to machine precision.** The energy identity
`I·(OCV − V) ≡ I²R0 + I·v1` holds to 1.8×10⁻¹⁵; charge conservation, `V ≤ OCV` on
discharge, and the conduction matrix being a Laplacian are all asserted in
`tests/test_physics_invariants.py`. This is the simulator being right, not the model being
a battery.

**Identifiability falls out as claimed.** On the demo pack:

| Fault hypothesis | Thermocouple? | CRLB (1σ) | SNR | Verdict |
|---|---|---|---|---|
| `R0` +20% on cell 5 | no | ±0.13% | **149.2σ** | DIAGNOSE |
| capacity −5% on cell 17 | no | ±0.15% | **32.6σ** | DIAGNOSE |
| cooling −40% on cell 12 | **yes** | ±7.31% | **5.5σ** | DIAGNOSE |
| cooling −40% on cell 10 | no | ±32.4% | **1.2σ** | **REFUSE** |

Cooling faults are identifiable on **4 of 32** cells — exactly the instrumented ones. And
identifiability is **not monotone in grid distance**: the sensor's next-door neighbour
(cell 3, 1.33σ) is beaten by a far pack corner (cell 0, 1.55σ), because a corner has fewer
conduction paths and so warms more. A hop-count heuristic gets this backwards; the Fisher
information does not.

**Refusal comes with a fix, and excitation can substitute for a sensor.** For the refused
cell 10, a counterfactual thermocouple lifts the cooling SNR from 1.24σ to **5.71σ** (a
4.6× tighter bound) — or, with no new hardware, a **1.83C** pulse makes the cell's own
voltage work as a thermometer through `R0(T)` (SNR 0.06σ → 3.17σ → 15.37σ as excitation
goes 0.05C → 1.39C → 2.50C).

**Excitation buys isolation, not merely precision** (`examples/02`). Carrying the pack
current as a nuisance parameter, under pulsed excitation the cost is near nil (`R0` ×1.00,
capacity ×1.18). Under **constant current** the design collapses: `R0` cost ×6.50, worst
VIF 261.7 — a constant offset is indistinguishable from every cell being slightly more
resistive.

**Correlated noise reallocates information** (`examples/02`). Modelling AFE 1/f noise as
AR(1), at `rho = 0.99` relative to white noise: capacity ×10.2 (worse), `R0` ×0.39 and
`hA` ×0.38 (*better*). A thermocouple's share of cooling information collapses from 90.4%
(`rho=0`) to 0.8% (`rho=0.99`) — a thermal time constant is DC against a 1 Hz sampler, so
whitening annihilates it. One headline verdict flips (`hA` on the instrumented cell 12,
5.46σ → 1.93σ REFUSE at `rho=0.9`), and the instrumented-beats-uninstrumented ordering
inverts at `rho=0.99`. The old assumption that idealisations only flatter the bound was
false in both directions (see [WHAT_DID_NOT_WORK](WHAT_DID_NOT_WORK.md) §2).

**Planning: optimise the target's axis, not the ellipsoid volume** (`examples/03`).
D-optimality (`det FIM`) crowns a 900 s cooldown (5.39σ) because it sharpens parameters
nobody asked about; Ds-optimality crowns a pulse train (**7.96σ**). Running it takes the
blind cell from 1.40σ (REFUSE) to 7.96σ (DIAGNOSE) with no new hardware. When nothing in
the library clears the threshold — an 8% fault under 1/f noise — AstraCell says so.

**The CRLB is attained** (`examples/05`). Under a matched model the MLE's interval coverage
tracks nominal (54.4/79.6/91.2/95.6/99.2% at 50/80/90/95/99% nominal) — the first evidence
in the repository that the bound is *reached*, not merely asserted, and an end-to-end check
of the noise sampler, whitening, and interval arithmetic.

**The bound cannot see model bias — the central Tier-1 result** (`examples/04`). Run the
first-order ECM observer against a higher-fidelity internal plant (SOC-dependent `R0`, a
diffusion branch, a core/surface split, a laggy thermocouple). The fit converges not on the
truth but on the pseudo-true `θ₀ = θ* + b`:

| hypothesis (cell 10) | SNR (variance) | structural bias | SNR (credible) | verdict |
|---|---:|---:|---:|---|
| `R0` +20% | 146.11σ | −3.08% | **6.49σ** | DIAGNOSE |
| capacity −5% | 32.65σ | **−18.53%** | **0.27σ** | **REFUSE_MODEL_BIAS** |
| cooling −40% | 1.40σ | +1043% | 0.04σ | REFUSE (unobservable) |

Fitting a first-order ECM over a 20-minute window **manufactures an apparent 18.5%
capacity loss** — nearly four times the 5% fault it was asked to find — because a slow
polarisation droop is arithmetically indistinguishable from lost coulombs. And **more data
makes it worse, not better**: replicate 10 000×, and the reported SNR climbs from 146σ to
**14 611σ** while the credible SNR sits fixed at 6.49σ. The bias is invariant to
replication and to noise scale, bit-for-bit (`tests/test_mismatch.py`). This is the insight
of §2, now a measurement.

Two corollaries that indict other parts of the system, honestly:
- **Excitation routes bias, it does not remove it.** 0.25C → 2.5C drives `R0`'s bias
  through zero and capacity's from −2.2% to −19.3% — so the Ds-optimal planner of §3 will
  sharpen a capacity estimate while destroying its credibility (ceiling 2.26σ → 0.26σ). It
  optimises variance and is blind to bias.
- **A nuisance parameter is where model error hides.** Freeing the current bias collapses
  capacity's bias from −18.5% to +0.3% — but reports a shunt offset of +1.11 A that is
  model error in disguise, inside its prior and flagged by nothing.

**Calibration makes the numbers worse, correctly** (`examples/05`). Under mismatch the
variance-only interval covers the truth **0% of the time** at every nominal level; the
estimate cloud tightens by 100× onto the pseudo-true −35.5%, never the −5% truth. The
model-bias gate turns those confident errors into refusals, dropping the harmful-overclaim
rate on capacity from **100% to 0%**. The capacity fault v0.0 diagnosed at 32.6σ is now
refused — because 30 of those sigmas were the observer's own model error. Worse numbers,
and the only kind worth trusting.

### Tier 2 — independently developed external simulator (PyBaMM)

The mismatch above was four terms *we* chose. Tier 2 replaces the plant with **PyBaMM**, an
electrochemical simulator AstraCell did not implement, whose SPMe electrolyte diffusion the
first-order ECM cannot express and whose gap we did not design. The estimator, FIM,
coverage, and gate are unchanged; PyBaMM only fills the data slot. Reproduce with
`python examples/06_external_plant_gate.py` and `..._07_external_positive_control.py`
(both need PyBaMM).

**The phantom fault** (`examples/06`). On a **perfectly healthy** cell the observer reports
a capacity deviation of **−67.6% ± 0.145% — 466σ from the truth of zero** — mistaking the
slow diffusion droop for capacity loss. A self-consistency control (an ECM-generated trace
through the same pipeline) covers at nominal to within **0.011**, so the collapse is the
external simulator's mismatch, not our plumbing. Without the gate the observer diagnoses
this phantom in every trial at every C-rate; the bias gate refuses all of them (overclaim
100% → 0%). The one number *not* to quote is the −67.6% itself: it swings +114% → −68% with
C-rate. Its instability is the proof it is not a capacity loss ([WHAT_DID_NOT_WORK](WHAT_DID_NOT_WORK.md) §5).

**The positive control** (`examples/07`). A negative control — a healthy cell, correctly
not diagnosed — is passed by a system that never diagnoses anything, and v0.3 shipped one
of those: its bias gate was a pure-noise statistic (`E[SNR²]=1` for any input) that refused
20 mV of sine, a 1718% ramp, and a real doubled-resistance fault alike. v0.4 injects real
PyBaMM faults and scores AstraCell as a detector. With a healthy baseline and a lack-of-fit
gate:

| scenario | true fault | true positive | false positive | refusal |
|---|---|---|---|---|
| healthy cell | — | — | **0.00** | 0.95 |
| real fault (+5.00 mΩ contact R) | +20% `R0` | **0.94** | — | 0.00 |
| confounded (diffusivity ×0.3) | none | — | **0.00** | **1.00** |

The paired estimator recovers the injected fault at **+20.0000%** with zero cross-talk, and
refuses the confounder it cannot express — the capacity hypothesis at **1.98σ against a
2.00σ threshold**, by one percent (pinned as a regression test). **Diagnosis is not
detection**: without the baseline, a doubled series resistance is diagnosed in every trial
at +91.4% against a truth of +100% — an 8.6-point miss inside a 0.23%-wide interval (145σ). Scoring only overclaim rewards
silence; `detection_metrics` scores a true positive as *DIAGNOSE ∧ interval covers ∧ sign
right*, so the true-positive rate saturates at 0.94, not 1.00 (a 95% interval misses 5% by
construction).

## 5. What failed, and what changed

The development history is preserved in full in
[WHAT_DID_NOT_WORK.md](WHAT_DID_NOT_WORK.md): the AR(1) whitening bug that inverted a
conclusion; the D-optimal planner's wrong objective; the retracted "realism only worsens
the bound"; the notebook output trap; the PyBaMM phantom's non-robust magnitude; the
first-order ECM's inability to calibrate capacity; and v0.3's gate that refused everything.
None is hidden, and each is pinned by a test.

## 6. What it cannot claim

The second-longest section, deliberately. Tier 3 has no *validation* — v0.6's one cell of
contact does not make one — and here is the full account of why. The complete ledger is
[LIMITATIONS.md](../LIMITATIONS.md); the essentials:

- **No physical battery validates this.** v0.6 ran the observer against one real cell (Oxford
  Cell1) and it *refused* — the first-order ECM's capacity estimate was wrong in sign (+10.5%
  against a measured −24.2% fade, ≈1150σ from truth), REFUSE_MODEL_BIAS on all 13 scored ages
  (§7.1, [REAL_CELL.md](REAL_CELL.md)). That is contact, not validation: one cell of eight, no
  fault detected, the ECM confirmed nowhere. Every Tier 1/2 result remains conditional on models
  that have otherwise not touched a cell. This is still the largest gap in the project.
- **The OCV curves are stand-ins.** `NMC_LIKE` uses a Li-polymer fit; `LFP_LIKE` is
  hand-built. Every SNR and CRLB is a statement about *this model*. Replace `cell/ocv.py`
  with measured tables before quoting a figure elsewhere.
- **The CRLB is variance-only.** It is blind to model bias by construction (§4). The whole
  model-bias apparatus exists because of this, and it is a *screen, not a bound*: on the one
  case with known truth the lack-of-fit screen captures only **31%** of the bias it warns
  about. A structural error three times smaller would have slipped through.
- **The internal mismatch plant is four hand-chosen terms** — a *lower bound* on how wrong
  the observer is, not a measurement of it.
- **The positive control's baseline is the identical simulation**, which no workshop can
  supply, so its detection rates are **upper bounds** on what any real baseline could
  deliver.
- **No classifier, no residual bank, no ramped faults, no pack-scale electrochemistry.**
  Faults are step faults, single, present from `t=0`.
- **Thresholds (2σ/5σ) and the thermal geometry are conventions/caricatures.** The
  conduction constants are the least defensible parameters and have among the largest effect
  on the grey boundary.
- **No EV-level validation, no safety-critical deployment readiness.** AstraCell is a
  research scaffold for the identifiability question.

## 7. What comes next

Only for the faults this machinery says are worth chasing and can be trusted:

1. **More real cells, and a better observer for them.** v0.6 took the first measured-cell step:
   `plant/oxford.py` turns the Oxford Battery Degradation Dataset into a measured pseudo-OCV and
   a 1C-discharge pair, and `examples/08` scored the observer against the dataset's own measured
   fade on Oxford Cell1 — where the first-order ECM came back directionally wrong and AstraCell
   refused every age (REFUSE_MODEL_BIAS 13/13; [REAL_CELL.md](REAL_CELL.md),
   [LIMITATIONS §16](../LIMITATIONS.md)). That is contact, not validation. The next step is the
   other seven cells, a same-day baseline, and an observer that can express real OCV drift — to
   learn whether the refusal ever becomes a trustworthy diagnosis.
2. **DFN with degradation submodels**, and **injected capacity fade** — which requires
   giving up the shared-OCV control that currently isolates dynamic mismatch, so it is a
   larger change than it looks.
3. **Ramped fault onsets** and **concurrent multi-cell faults**, both currently idealised
   away.
4. **A classifier / residual bank** — built last, on the faults the identifiability layer
   has certified as answerable and trustworthy. Building it first would reproduce exactly
   the failure mode this project exists to refuse.

---

## Methods appendix

Symbols: `θ` parameters, `θ*` truth, `θ₀` pseudo-true, `S = ∂(measurement)/∂θ` the
sensitivity tensor (central differences, `eps = 1e-3` relative), `Σ` the measurement-noise
covariance, `r = plant(θ*) − observer(θ*)` a residual.

- **Fisher information.** `FIM = SᵀΣ⁻¹S`, built over all cells' parameters plus an optional
  pack-global current-bias nuisance with a Van Trees prior. Additive across independent
  experiments; Loewner-monotone in sensors. Code: `observability.fisher`.
- **CRLB.** `Var(θ̂_j) ≥ [FIM⁻¹]_jj`. Computed by eigendecomposition, discarding directions
  at floating-point noise and returning `inf` for unidentified parameters — **not** `pinv`,
  whose minimum-norm solution would report finite variance for parameters the data cannot
  constrain. Code: `observability.fisher.crlb`.
- **VIF.** `VIF_j = FIM_jj·[FIM⁻¹]_jj ≥ 1`; the isolation gate. Not `cond(FIM)`, which is
  dominated by the single worst-informed direction.
- **SNR.** `|magnitude|/√CRLB`, magnitude-independent under local linearisation.
- **AR(1) whitening.** Whitening is a scaled first difference
  `(x[t] − ρ·x[t−1])/√(1−ρ²)`; its effect on information is an exact reciprocal pair — a DC
  sensitivity keeps `(1−ρ)/(1+ρ)`, an alternating one gains `(1+ρ)/(1−ρ)`. Cross-checked
  against a densely-inverted correlation matrix in `tests/test_noise_correlation.py` (the
  test that caught the sign bug).
- **Structural bias.** `b = FIM⁻¹SᵀΣ⁻¹r`. The linearised one-step value; the iterated
  `pseudo_true_bias` is the fit's actual resting point (quote the iterated one where they
  differ). Exactly invariant to replication and noise scale.
- **Lack-of-fit screen.** Where the truth is unknown, decompose the whitened residual
  against the projector onto the sensitivity columns. The in-span part *is* the estimate;
  the out-of-span part `ρ̃ = (I−P)Δg` is independent of the fault and gives
  `b_lof_i = ‖ρ̃‖·σ_i`. A screen, not a bound: zero when the change is expressible. Code:
  `observability.bias.lack_of_fit_bias`.
- **Estimators.** `fit_linear` (weighted least squares, exact for linear-Gaussian,
  covariance `FIM⁻¹`) for magnitude sweeps; `fit_gauss_newton` (damped fixed-information,
  against the nonlinear observer) so "the bound is attained" is a measurement, not a
  tautology. Code: `observability.estimator`.
- **Metrics.** Coverage (two-sided Gaussian); harmful overclaim (a DIAGNOSE whose
  variance-only interval misses the truth at 95%); `detection_metrics`, which scores a true
  positive only as *DIAGNOSE ∧ interval covers ∧ sign correct*. Code: `calibration.metrics`.
