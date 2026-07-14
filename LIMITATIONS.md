# Limitations

Written before the code, updated as the code taught us things. Read this before
believing any number this repository produces.

The organising principle: AstraCell exists to refuse to overclaim. A limitations
document that is discovered late is a marketing document. This one is a design
input.

---

## 1. The models are stand-ins, not cells

Neither OCV curve is fitted to a real cell.

- `NMC_LIKE` uses the Chen & Rincon-Mora (2006) functional form with their
  published coefficients, which were fitted to a **lithium-polymer** cell. Its
  shape is broadly graphite/NMC-like. It is not an NMC cell.
- `LFP_LIKE` is hand-constructed. It reproduces the *qualitative* LFP plateau
  (`dOCV/dSOC` of order 0.5 mV per percent SOC). It is not quantitatively faithful
  to anything.
- The entropic coefficients `dOCV/dT` are plausible in sign and magnitude
  (~0.1 mV/K) and measured on nothing.
- `R0`, `R1`, `C1`, thermal mass, `hA`, and the conduction constants are
  order-of-magnitude representative of a large-format prismatic cell. None comes
  from a datasheet.

**Consequence.** Every SNR, every Cramér–Rao bound, and every "this fault is
detectable above 0.8C" statement is a statement about *this model*, not about a
battery. The machinery is what is being validated here; the numbers are
illustrative. Replace `cell/ocv.py` with PyBaMM-derived or measured tables before
quoting a single figure outside this repository.

---

## 2. The Fisher information is an optimistic upper bound — but not uniformly

The CRLB says: *no unbiased estimator can do better than this.*

**This section previously claimed that every modelling simplification inflates the
information, so that real performance is "worse than reported, never better," and
that this was therefore the safe direction for a system built to abstain. That
claim was an estimate, it was never measured, and it is false.** Two of the three
per-cell parameters get *better* when the white-noise idealisation is relaxed. The
error was not conservative; it was simply wrong about the sign, in both directions
at once. `examples/02_noise_robustness.py` measures it. What follows is the
measurement, not a footnote.

### 2a. Noise is not white, and correlated noise does not uniformly hurt

Real AFE noise has a 1/f component. `NoiseModel` now carries an AR(1) lag-1
correlation `rho` on the voltage and temperature channels, whitened exactly (see
`whiten_ar1`, and the dense-inverse cross-check in
`tests/test_noise_correlation.py`).

Whitening an AR(1) process is a scaled first difference. Its effect on information
is an exact reciprocal pair:

| sensitivity shape | information multiplier at `rho` | at `rho = 0.9` |
|---|---|---|
| constant (DC)     | `(1-rho)/(1+rho)` | ÷19 |
| alternating       | `(1+rho)/(1-rho)` | ×19 |

So the question is not *how much* information is lost. It is *which parameters*
lose it. Measured on the 4×8 demo pack, 1200 s of 1.0C pulse train, over the full
97-parameter spec set (so cross-cell confounding is carried), CRLB relative to
white noise:

| `rho` | `R0` | capacity | `hA` |
|---|---|---|---|
| 0.5  | ×1.54 | ×1.70 | ×1.52 |
| 0.9  | ×1.21 | ×4.00 | ×1.18 |
| 0.99 | **×0.39** | ×10.2 | **×0.38** |

Capacity degrades monotonically and tracks the DC prediction
`sqrt((1+rho)/(1-rho))`, because its signature is a slow SOC ramp. `R0` and `hA`
are **non-monotone**: they worsen out to `rho ≈ 0.7`, then recover, and at
`rho = 0.99` are 2.6× *tighter* than under white noise. Differencing destroys the
noise faster than it destroys a pulse edge. Pulsed excitation is lock-in
detection.

Consequences that break earlier claims in this repo:

- At `rho = 0.9`, one of the four README headline verdicts changes: cooling on an
  *instrumented* cell falls from `DIAGNOSE` (5.46σ) to `REFUSE` (1.93σ). Cooling
  faults are then identifiable **nowhere** on this pack, not "exactly where the
  thermocouples are".
- A thermocouple's share of the cooling information collapses from **90.4%** at
  `rho = 0` to **0.8%** at `rho = 0.99` — the thermal time constant (~200 s) is
  essentially DC against a 1 Hz sampler, so whitening annihilates exactly the
  channel that was supposed to see the fault. `hA` is then read through `R0(T)`'s
  leak into the voltage channel.
- Consequently the instrumented-beats-uninstrumented ordering **inverts** at
  `rho = 0.99` (cell 12 with a thermocouple: 3.14σ; cell 10 without: 3.25σ).
- Under `rho = 0.9`, ten minutes of hard pulsing beats fitting the best possible
  thermocouple (7.08σ vs 2.00σ, on the local spec set of
  `examples/03_next_best_test.py`). The correct recommendation is the *opposite*
  of the white-noise recommendation: excite, do not instrument.

**We do not know `rho` for any real pack.** We know `rho = 0` is the most
optimistic choice available and that this repo made it silently. The headline
numbers in `README.md` are quoted at `rho = 0` and are now labelled as such.

### 2b. The current is now carried as a nuisance parameter — cheaply, usually

`ParamKind.CURRENT_BIAS` is a pack-global unknown offset on the measured current,
given a Gaussian prior equal to the shunt's accuracy (2 A) via `prior_information`
(Van Trees). Physical per-cell parameters get **no** prior: we are not willing to
assume a cell is healthy in order to conclude that it is healthy.

Measured cost of admitting we do not know the current (full 97-parameter spec set,
white noise, cell 10):

| excitation | `R0` | capacity | `hA` | `I_bias` posterior | worst VIF |
|---|---|---|---|---|---|
| pulse train, 1.0C | ×1.00 | ×1.18 | ×1.00 | ±0.0055 A | 5.1 |
| constant current  | **×6.50** | ×1.29 | ×1.00 | ±0.1767 A | **261.7** |

Under pulsed excitation the cost is near nil — 32 voltage channels pin a
common-mode offset ~400× better than the shunt's own spec. `R0` and `hA` pay
nothing; capacity pays 18%.

Under **constant current** the design collapses. A constant current offset is
indistinguishable from every cell being slightly more resistive, and from a slow
capacity drift. Every VIF crosses the multicollinearity threshold of 10 by one to
two orders of magnitude, `R0` alone degrades 6.5×, and the decision layer must
return `REFUSE_CONFOUNDED`. **With no excitation there is no way to tell the
ammeter from the pack.** Constant-current parameter estimation is not identifiable
once you admit the ammeter is imperfect — and every real ammeter is.

This is opt-in: `grey_cell_map(..., include_current_bias=True)`. The default is
`False`, which is the optimistic choice, and is one of the reasons the bound
remains an upper bound.

### 2c. Still-unrelaxed idealisations

3. **The model is assumed correct.** This was, until §12 below, "the largest
   remaining gap, not quantified anywhere in this repo." It is now quantified, and
   it is worse than the noise idealisation was. Everything the CRLB says remains
   conditional on a model structure that is false, and the resulting bias is
   invisible to every number in §§1–11. See **§12**.
4. **Estimators are assumed unbiased.** The CRLB does not bound biased estimators.
   Regularised, thresholded, or ML estimators routinely beat it in MSE by trading
   bias for variance. "Below 2σ" therefore means "no *unbiased* estimator sees
   this", which is the right conservative reading but not the only one. Note the
   irony now that §12 exists: the estimator is not unbiased, and its bias is not a
   deliberate trade for variance. It is an accident of the model.
5. **Sensitivities are local.** Central finite differences at `eps = 1e-3`
   relative. A 40% capacity fault is not a small perturbation, and the linearised
   bound will misstate detectability for large faults — in an unknown direction.
6. **The noise correlation is a single AR(1) pole.** Real 1/f noise is not AR(1);
   it is a superposition of poles. AR(1) captures the qualitative DC-versus-pulse
   split exactly, but a measured PSD would not be reproduced by any single `rho`.

---

## 3. There is no fault classifier, and that is on purpose

This scaffold answers *"is this question answerable?"*, not *"which fault is it?"*.
There is no detector, no residual bank, no CUSUM, no posterior, no conformal
calibration. Nothing here has ever detected a fault. Nothing here has ever been
tested against real telemetry.

Building a classifier before the identifiability question is settled produces a
system that is confident exactly where it should be silent. That is the failure
mode this repository is organised against.

---

## 4. Faults are step faults, applied for the whole run

Real degradation ramps in over tens to hundreds of cycles. A step fault present
from `t = 0` is easier to identify than a real one, because every sample carries
information about it. Ramped onsets are the honest next step. They are deferred,
not faked.

Related: only single faults are considered. Real packs are not single-fault. Two
concurrent faults on neighbouring cells will confound in ways this scaffold does
not measure.

---

## 5. Cross-cell confounding is included in one place and not the other

- `observability.grey_cell_map` builds the Fisher information over **all** cells'
  parameters, so its CRLB marginalises over both within-cell confounding (R0 vs
  capacity vs hA) and cross-cell confounding (a hot neighbour looks like a hot
  cell). This is the honest computation. It costs `2 × n_cells × 3` simulations.
- `observability.detectability_heatmap` uses only the **target cell's** three
  parameters, for cost. It therefore *understates* the CRLB — it is optimistic.

Both are documented at their call sites. Do not mix them up when quoting a number.

---

## 6. The sensor model omits real failure modes

`NoiseModel` has Gaussian noise and uniform quantisation. It does not model:
dropouts, timestamp jitter, sensor latency, cross-channel coupling, ADC
nonlinearity, self-heating of the NTC, or the thermal lag between a cell's core
and the surface-mounted thermocouple that supposedly measures it.

That last one matters most. The simulator has one thermal node per cell. A real
thermocouple reads a *surface* temperature that lags the core by tens of seconds.
A two-state (core/surface) thermal model would reduce the information available
about `hA` and would move the grey boundary outward.

---

## 7. Thermal geometry is a 2D grid, not a pack

Cells conduct to their four grid neighbours. Real packs have coolant plates,
module walls, busbars (which conduct heat *along the electrical path*, not the
geometric one), and potting compound. The conduction anisotropy here
(`k_intra = 0.6`, `k_inter = 0.15` W/K) is a caricature with the right qualitative
shape: heat crosses a module boundary reluctantly.

Since the grey-cell boundary is set by how far a thermal disturbance travels
before it reaches a thermocouple, **the grey regions are directly sensitive to
these two numbers.** They are the least defensible parameters in the package and
they have among the largest effect on the headline result.

---

## 8. Busbar and contact resistance are indistinguishable from cell resistance

Given cell voltage and pack current alone, a resistance increase in the interconnect
between two cells is observationally identical to a resistance increase inside one
of them. Nothing in this repository can separate them, and nothing could. This is
a property of the sensor topology, not of the algorithm.

---

## 9. LFP is much harder than NMC, and the code knows it

On the `LFP_LIKE` curve, `dOCV/dSOC` in the plateau is roughly one sixth of the
NMC-like value. Capacity faults are correspondingly harder to see mid-SOC. This
is real, is reproduced by `notebooks/01_identifiability_study.ipynb`, and is a
qualitative statement only — see limitation 1.

---

## 10. What has never been done

- No real battery has been measured.
- No real fault has been detected.
- No public dataset has been ingested.
- No claim in this repository has been validated against anything but itself.

This section used to add a fifth line: *"No CRLB has been shown to be attainable… nobody has run
an estimator on noisy data and compared its scatter to the bound."* **That is now done** — see
§13 and `observability.estimator`. A Gauss–Newton fit's scatter matches `sqrt(CRLB)` and its
coverage tracks nominal under a matched model. But the estimator runs against the *synthetic*
plant, so it shows the bound is attainable *for this model*, not that this model is a battery.
The first four lines above still stand, and they are the ones that matter.

This section used to end: *"the correct next step is to inject faults with a **higher-fidelity
model** than the one the observer assumes, so that model mismatch is part of the experiment.
That is not done here."* **It is now done** — see §12 and `src/astracell/plant/` — but with a
hand-written intermediate plant, not PyBaMM.

That choice was deliberate and it has a cost. The benefit: a plant whose mismatch knobs all
default to zero collapses onto the observer bit-for-bit, so "the bias vanishes when the
mismatch does" is a *falsifiable* control rather than an article of faith. With PyBaMM you
can never switch the mismatch off, so you can never establish that a reported bias is not a
bug in your own harness. The cost: the four structural terms are hand-chosen, so §12 measures
**a lower bound on how wrong we are**, not how wrong we are. Replacing the plant with PyBaMM
(DFN, or SPMe with degradation submodels) is the next step, and now it has a harness and a
control to be checked against.

---

## 11. Thresholds are conventions

`2σ` for "weak" and `5σ` for "observable" are conventions borrowed from other
fields. They are exposed as arguments (`weak_sigma`, `strong_sigma`) rather than
buried, because there is no principled reason for them absent a stated cost of a
false positive versus a missed detection. The condition-number threshold for
declaring parameters confounded (`1e4`) is likewise a convention.

---

## 12. The observer's model is wrong, and now we know what that costs

Sections 1–11 all describe *variance*: how finely a correct model could be pinned
down by imperfect data. This section describes *bias*: how far a wrong model lands
from the truth however perfect the data. They are not the same kind of error and
the difference is the point.

`examples/04_model_mismatch.py` runs a higher-fidelity **plant** — SOC-dependent
`R0`, a slow diffusion branch the ECM lacks, a core/surface thermal split, and a
laggy thermocouple — against the same first-order ECM **observer** the rest of the
repository uses. Both are given *identical* parameters, so every difference is
structural. The fit does not converge on the true `θ*` but on the pseudo-true

```
θ₀ = argmin_θ ‖ g(θ*, u) − f(θ, u) ‖²_Σ⁻¹        b = θ₀ − θ*
```

### 12a. What it costs

Measured on the 4×8 demo pack, 1200 s of 1.0C pulse train, cell 10, ammeter believed,
on the three-parameter local spec set (so the CRLBs differ slightly from README §1, which
carries all 97 parameters and hence cross-cell confounding):

| hypothesis | CRLB (1σ) | SNR (variance) | structural bias | SNR (total) | verdict change |
|---|---:|---:|---:|---:|---|
| `R0` +20% | ±0.14% | 146.11σ | −3.08% | **6.49σ** | DIAGNOSE → DIAGNOSE |
| capacity −5% | ±0.15% | 32.65σ | **−18.53%** | **0.27σ** | DIAGNOSE → **REFUSE_MODEL_BIAS** |
| cooling −40% | ±28.6% | 1.40σ | +1043% | 0.04σ | REFUSE → REFUSE |

Because the mismatch is pack-global the capacity bias is **common-mode** (−18.45% / −18.53% /
−18.56% on cells 5 / 10 / 17). It does not single out a bad cell, so no cross-cell comparison
detects it. This is the same fact that lets a pack-global nuisance parameter absorb it (§12c).

The capacity bias is not a bug. It is arithmetic: a slow polarisation droop of a few
millivolts is indistinguishable from coulombs that never left the cell, so the observer
attributes it to a capacity 18.5% smaller than the truth — **almost four times the fault
it was asked to look for.** Short-window capacity estimation with a first-order ECM
manufactures the very fault it is hunting.

`hA`'s bias is dominated by the **electrical** blind spots, not the thermal ones, because
an uninstrumented cell's cooling coefficient is only ever seen through `R0(T)`. Model
error in the voltage channel arrives disguised as a thermal fault. That is a direct
consequence of the mechanism celebrated in README §3.

### 12b. Why more data does not help

`b` is *exactly* invariant to the two things that shrink the CRLB. Replicate the same
experiment `k` times and both the FIM and the score scale by `k`, so `b = FIM⁻¹Sᵀ Σ⁻¹ r`
is unchanged bit for bit. Scale every noise `σ` by `c` and the same cancellation occurs.
Meanwhile `sqrt(CRLB)` falls as `1/√k` and rises as `c`. Both invariances are asserted in
`tests/test_mismatch.py`.

So `SNR_total = m/√(CRLB + b²) → m/|b|`, a **ceiling**. On the table above, ten thousand
repetitions of that experiment move `R0`'s reported SNR from 146σ to 14 611σ and its
*actual* credibility not at all: it sits at 6.49σ throughout. **A CRLB-only system grows
unboundedly more confident and no less wrong the more data you feed it.**

### 12c. What this breaks elsewhere in the repo

- **Excitation does not remove structural error. It decides which parameter absorbs it.**
  Raising the pulse amplitude from 0.25C to 2.5C improves every CRLB, drives `R0`'s bias
  from −10.1% through zero to +1.1%, and drives capacity's from −2.2% to −19.3% — capacity's
  bias *ceiling* falling from 2.26σ to 0.26σ as it does so. The Ds-optimal test planner of
  `examples/03_next_best_test.py` will therefore recommend a hard pulse train to sharpen
  a capacity estimate and destroy its credibility in the process. **It optimises variance
  and cannot see bias. That is a real defect in §8 of the README, not a nuance.**
- **Nuisance parameters are where model error hides.** Freeing the pack-global current
  bias collapses capacity's bias from −18.5% to +0.3% — genuinely, because a nuisance
  regressor spanning the residual is the textbook cure for omitted-variable bias. The
  price is a reported shunt offset of **+1.11 A that is not a shunt offset at all**, sits
  comfortably inside its 2 A prior, and is flagged by nothing. §2b called that nuisance
  parameter "cheap". It is cheap in variance and expensive in interpretability.
- **`README.md` §1's headline table, and every SNR in §§1–8, are variance-only.** They are
  upper bounds on the performance of an estimator using a model we know to be false.

### 12d. What §12 itself is not

- **The plant is not a battery.** It is the ECM plus four hand-chosen structural terms at
  order-of-magnitude-plausible strengths. It is a *lower bound on how wrong we are*, not a
  measurement of it. The next step is PyBaMM (DFN or SPMe) as the plant. It is not done.
- **`b` is computed against a plant we guessed**, so it is an estimate of the *scale* of an
  error we cannot average away, never a correction to subtract. Subtracting it would
  assume the guess. The decision layer therefore widens uncertainty rather than shifting
  the estimate.
- **The ceiling belongs to an experiment, not to a pack.** A different duty cycle has a
  different `b`. Do not read the 1090σ ceiling that appears at a 2.00C pulse as an
  operating point: it is a zero crossing, and a ±10% change in the assumed diffusion
  resistance *flips the sign* of `R0`'s bias there. Tuning a duty cycle to null a bias
  inferred from your own guess of the plant calibrates against the guess.
- **`parameter_bias` is first order.** At full `REALISTIC_MISMATCH` it overstates
  capacity's bias by ~18% and understates `hA`'s by ~43% against the iterated
  `pseudo_true_bias`. Quote the iterated fit. The two converge as the mismatch shrinks
  (45% → 0.4% relative error over a 300× range), which is the only reason to believe
  either.
- **The mismatch is pack-global**, hence common-mode. A *cell-specific* structural error —
  one cell whose diffusion branch has degraded — would not be absorbed by any pack-global
  nuisance, and is not modelled.
- **`θ*` is a convention.** The plant has no parameter called "`R0`"; it has `R0(SOC, T)`.
  Every mismatch term is defined to vanish at the initial operating point so that the
  residual is exactly zero at `t = 0` and `θ*` is at least unambiguous *there*.

### 12e. The one thing §12 does not undermine

`REFUSE_UNOBSERVABLE` still fires first, and it is still right. A cooling fault on a cell
with no thermocouple was never diagnosable, and adding model bias to a hypothesis that was
already refused changes nothing. Where AstraCell was silent, it remains correctly silent.
Where it was *confident*, it was sometimes confidently wrong, and now it says so.

---

## 13. Calibration proves self-consistency, not truth

§12 priced the model error on one example. `calibration/` asks whether that was luck: across
thousands of repeated experiments with a *known injected fault*, do AstraCell's intervals and
verdicts mean what they claim? The answer is yes under a matched model and no under mismatch —
which is exactly what should happen, and the first time this repository has *measured* it rather
than argued it. `examples/05_calibrated_abstention.py` and `docs/CALIBRATION.md` carry the full
result. What follows is what it does **not** establish.

### 13a. There is now an estimator, and it is load-bearing

Everything before v0.2 was a property of the design; coverage needs a decision on realised data.
`observability.estimator` supplies the smallest honest one. Two consequences for how to read the
numbers:

- The **matched-model coverage** result (the MLE attains the CRLB) is genuine but uses the
  `fit_gauss_newton` estimator, which is a *fixed-information* M-estimator, not the exact Newton
  MLE. Its scatter matches the CRLB up to the model's curvature — measured at ~6% on the demo
  fault, not zero. At larger faults the linearisation degrades in a direction coverage would
  reveal but a point estimate would not.
- The **mismatch coverage** results use the exact linear-Gaussian `fit_linear`, whose own
  curvature bias (~0.1%) is three orders of magnitude below the structural bias under study
  (~30%), so it does not contaminate the conclusion. The two estimators are used where each is
  most defensible; mixing them is deliberate, not sloppy, and is stated in the example.

### 13b. "Calibrated" is conditional on the mismatch we wrote

The bias gate is exactly as good as the plant we guessed. The coverage curves show the
variance-only interval undercovering and the bias-aware gate refusing — but *the bias it gates
on is computed against `REALISTIC_MISMATCH`*. A structural error of a different shape (a degraded
cell, a chemistry the ECM mismodels differently, a cell-specific rather than pack-global fault —
see §12d) would produce a different bias, and might not be gated at all.

> **v0.3 update.** §14 relaxes this one step: the gate is now tested against a mismatch we did
> *not* write — a PyBaMM cell's diffusion — and it still refuses. That is a mismatch we did not
> *design*, but it is still a *synthetic* one; PyBaMM is a model, not a cell. Calibration under a
> mismatch you did not write is now tested; calibration against a *real* cell still cannot be,
> synthetically.

### 13c. The metrics encode choices

- **Harmful overclaim** is defined as a DIAGNOSE whose *variance-only* interval misses the truth
  at 95%. That flags R0 diagnoses whose magnitude is biased past a tight interval even though the
  fault is real — arguably harsh, since the existence claim is correct. The definition is
  deliberately strict (an interval that misses is an interval that lied), but a different cost
  model would count differently.
- **Coverage** is two-sided and Gaussian (`two_sided_z` via the normal quantile). The estimator's
  finite-sample distribution is not exactly Gaussian, so coverage at the extreme levels (99%)
  leans on a tail the Monte Carlo samples thinly.
- Everything is measured at seed 0 on a 2×2 pack over 600 s. The *shape* of every result is the
  claim; the specific percentages move with pack, window, and seed.

### 13d. What it genuinely establishes

Set against those caveats, three things are now measured facts about the code, not hopes:

1. The noise model, whitening, and interval arithmetic are mutually consistent — the sampled
   noise whitens to white, and matched-model coverage is nominal to sampling error. The AR(1)
   whitening bug of §2a could not survive this test.
2. Under mismatch the variance-only interval undercovers to the point of *never* covering, and
   more data does not fix it: the estimate cloud tightens onto the pseudo-true value at a fixed
   offset from the truth. The central claim of §12 is now a frequency, not an anecdote.
3. The model-bias gate reduces the harmful-overclaim rate to zero on the parameter whose model
   error is fatal (capacity: 100% → 0%), by refusing. Refusal is doing measurable work.

None of that is a statement about a battery. It is a statement that AstraCell is honest about the
model it assumes and measurably stops overclaiming when that model is wrong. Whether the model
resembles a cell remains the unproven question at the centre of every section above.

## 14. The external plant is still a model, not a cell

§13b named the sharpest limit of the calibration work: the bias gate was only ever tested against
`REALISTIC_MISMATCH`, *a plant we wrote*. v0.3 closes that specific gap and no other. The data now
comes from a **PyBaMM** SPMe single cell — electrolyte and particle diffusion the first-order ECM
cannot express, a gap we did not design. `examples/06_external_plant_gate.py`,
`tests/test_external_plant.py`, and `docs/EXTERNAL_PLANT.md` carry the result; what follows is what
it still does not establish.

### 14a. It is external mismatch, not external truth

PyBaMM is a sophisticated *model*, not a measured battery. v0.3 swaps a simple synthetic plant for
a complex one; it introduces no real cell. Everything §1 says about the models being stand-ins
still holds — SPMe is a better stand-in than the ECM, that is all. This is not EV validation, not
even bench validation: one cell, one parameter set (Chen2020), one duty cycle, isothermal, no
ageing, no pack. The claim is narrow on purpose: AstraCell's abstention behaves correctly when the
data-generating model is richer than the observer. Whether the observer is right about a *physical*
cell is untouched.

### 14b. The result is stronger and worse than v0.1's, which is the point

Against our own `REALISTIC_MISMATCH`, capacity carried a ~30% structural bias. Against PyBaMM the
same observer, on a **healthy** cell, reports a **−67.6% ± 0.145%** capacity deviation — 466σ from
the truth of zero — and diagnoses a fault that does not exist in 100% of ungated trials. The bias
gate refuses all of them (harmful overclaim 100% → 0%). The self-consistency control (an
ECM-generated trace through the same pipeline) covers at nominal to within 0.011, so the collapse
is the plant's mismatch and not a harness bug. That AstraCell looks *worse* against a more faithful
plant, and responds by refusing, is the honest outcome, not a regression.

### 14c. The phantom fault's magnitude is not robust — only the refusal is

The −67.6% is not a property of the cell. It swings from **+114% to −68%**, changing sign, as the
mean C-rate changes, and runs from **−94% to +5%** as the observer's fixed RC branch is retuned
toward the diffusion timescale (`docs/EXTERNAL_PLANT.md` §3d). Its instability *is* the evidence
that it is model mismatch rather than capacity loss. An earlier draft of `calibration/external.py`
asserted the bias was "robust to the R1/C1 choice"; measurement falsified that, and the claim was
retracted in the code and here. What survives every excitation and every RC tuning is the
*inequality* — bias exceeds the CRLB σ by more than 30× — so the variance-only interval overclaims
and the gate refuses in all cases. Quote the direction, never the number.

### 14d. Deliberately out of scope for v0.3

Capacity is the only target, because it is the only quantity the ECM and PyBaMM share a truth about
(a healthy cell's deviation is exactly zero); `R0` and `hA` have no such clean external ground truth
and are not assessed. No degradation is injected — the mismatch is entirely the model-order gap on a
sound cell, and PyBaMM-side ageing is deferred. PyBaMM is an optional dependency: the core repo
stays numpy-only and the whole external-plant suite skips cleanly without it, so nothing above is
part of the guarantee the base install makes. The next honest step is the one every version has
pointed at — a *measured* pseudo-OCV and pulse response — which no simulation, however faithful,
can stand in for.

---

## 15. The refusal now discriminates — but only because v0.4 went looking for the bug

v0.3 asked whether AstraCell refuses under an external plant. It does. v0.4 asked the question that
had to come next, and that v0.3 had no way to ask: **when the external cell really is broken, does
AstraCell notice?**

### 15a. v0.3's external bias gate was a constant, and nothing in v0.3 could have said so

`calibration.external.prepare_external` projected the very trace the estimator was about to fit, so
its `structural_bias` was algebraically the expected estimate, `b ≡ E[δ̂]`. Substituting into the
credibility statistic gives, exactly and for any residual,

```
SNR_total = |b + ε| / sqrt(σ² + b²)        E[SNR_total²] = (b² + σ²)/(σ² + b²) = 1
```

**A pure noise statistic, carrying no information about the data.** Whenever the bias exceeds the
noise — the only regime a bias gate exists for — it concentrates on 1σ and returns
`REFUSE_MODEL_BIAS` with certainty. Measured: 300/300 refusals on 20 mV of sine wave, on an absurd
1 V ramp implying a 1718% capacity deviation, and on a PyBaMM cell whose **series resistance has
doubled**, where it dutifully labels the 91.4% fault "bias".

v0.3's headline results all reproduce. Its `REFUSE_MODEL_BIAS` label was arithmetic, not evidence.
This is what a negative control cannot catch, and what a positive control catches immediately.

### 15b. The honest gate under-warns by 3.2×, and bounds nothing

The replacement (`observability.bias.lack_of_fit_bias`) reads only the part of the residual that
**no setting of the observer's parameters reproduces** — the out-of-span component, which is exactly
independent of any fault present and needs no counterfactual. It is invariant to the noise scale, as
a structural bias must be.

It is a **screen, not a bound**. The bias lives in the *in-span* part, which is unmeasurable without
the truth; the screen infers it from the orthogonal part on the assumption that a model wrong in one
subspace is wrong in the other. On the one case where the truth is known — v0.3's healthy cell, where
the entire −67.6% estimate *is* bias — it reports 20.9%, capturing **31%** of the error it warns
about. It caught the overclaim. **It would not have caught one three times smaller.**

And a structural error lying entirely *in* the observer's span leaves the screen at exactly zero. That
is a theorem about the experiment, not a defect: such a change is indistinguishable from a parameter
shift by any procedure looking only at this data.

### 15c. The positive control's fault is in the model's span by construction

PyBaMM's `Contact resistance [Ohm]` produces a differential of `−ΔR·I` to **3.6 × 10⁻¹⁶ V**, and the
ECM's `∂V/∂R₀ = −I` exactly. The paired estimator recovers `+20.0000%` from a 20% injection with zero
cross-talk onto capacity. That exactness is *what a positive control is*: it proves the pipeline can
recover a fault the observer can express, and it proves **nothing whatever** about faults it cannot.

The confounder is where the physics is. Cathode diffusivity ×0.3 — real degradation, true
`(R₀, capacity)` deviation `(0, 0)`, verified by the C/20 shift being linear in current (199 mV/C over
a 10× range) and hence a vanishing overpotential rather than a capacity loss. The observer reads it as
a 119% capacity fault at 579σ. The gate refuses. **On the capacity hypothesis it refuses at 1.98σ
against a 2.00σ threshold — by one percent.** Shorten the window from 600 s to 300 s and the same
hypothesis reaches 3.35σ and is merely weakened. The margin is real and it is thin, and it is pinned
as a regression test.

### 15d. The baseline is the assumption, and ours is impossible

Baseline subtraction and paired comparison are the same estimator — the projection is linear, and the
identity holds to 6 × 10⁻¹⁶. The correction is exact. But the healthy baseline used here is the
**identical simulation** with the fault parameter at its healthy value: same solver, same grid, same
initial SOC, same day. A real beginning-of-life fingerprint is none of those. **Every detection rate
in `docs/POSITIVE_CONTROL.md` is an upper bound on what any real baseline could deliver**, and the
paired estimator's nominal coverage is a statement about the arithmetic, not about a workshop. Where
no comparable baseline exists, the honest behaviour is v0.3's — refuse — which is a correct answer,
not a failure mode, provided nobody mistakes it for discrimination.

The cost of the correction is not zero: two traces means two noise draws, so `σ` widens by exactly
`√2`.

### 15e. Diagnosis is not detection, and the old metrics could not tell

A system can diagnose a real fault in **every** trial while its interval **never** covers the truth.
The raw path does exactly this at a doubled series resistance: `DIAGNOSE` at 10.7σ, reporting +91.4%
against +100%, an 8.6-point miss inside a 0.23%-wide interval — 145σ of misplaced confidence, and a
harmful overclaim in every trial. `harmful_overclaim_rate` alone rewards silence; `diagnosis_rate`
alone rewards noise. `detection_metrics` reports both, and scores a true positive only as
*DIAGNOSE ∧ the interval covers the truth ∧ the sign is right*.

Note the ceiling this imposes: the true-positive rate saturates at **0.94, not 1.00**, because a 95%
interval misses 5% of the time by construction. Harmful overclaim floors at `1 − coverage_level` for
any detector that fires at all. v0.3's 0% overclaim came from 0% diagnosis. It is not a number to
aspire to.

### 15f. Still out of scope for v0.4

Capacity fade is **still not injected**, and cannot be cleanly: `Nominal cell capacity [A.h]` only
normalises the C-rate and changes no electrode capacity, while real fade (loss of lithium inventory,
loss of active material) moves the stoichiometry window and hence the OCV–SOC map — which would break
the shared-pseudo-OCV control that isolates *dynamic* mismatch in the first place. Fixing that means
giving up the control or fitting the OCV too, and both are larger changes than v0.4 earns.

Also absent: SEI growth, thermal coupling, pack scale, DFN, more than one chemistry parameter set,
more than one duty cycle, and any measured cell whatsoever. PyBaMM remains a model, the fault remains
one we injected into it, and the next honest step remains the one every version has pointed at.

---

## 16. The real cell is contact, not validation

Every section above ends the same way: the next honest step is a *measured* cell, which no
simulation can stand in for. v0.6 takes that step. It runs the observer against the **Oxford Battery
Degradation Dataset** — eight real Kokam 740 mAh cells cycled to end of life — and scores the
capacity estimate against each age's **measured** fade. For the first time the ground truth is a
number nobody in this project chose, and the first run is done: on Cell1 the ECM's estimate is wrong
in *sign* and AstraCell refuses all 13 scored ages. `docs/REAL_CELL.md` carries the numbers, the
figure, and the how-to-run; what follows is what that result does *not* settle.

### 16a. The first real run has happened — and it refused

The run exists, on Oxford Cell1: a measured fade to **−24.2%** over 78 ages, the first-order ECM's
capacity estimate wrong in sign (a **+10.5%** "gain" at end of life), and `REFUSE_MODEL_BIAS` on all
13 scored ages (coverage **0/13**), in both OCV modes. Two things keep this honest. First, the
numbers are **not committed**: the ODC-ODbL data is a licensed ~266 MB download that never enters the
repo, so the offline test suite exercises the whole pipeline on *synthetic* Oxford-format data and
`examples/08` skips cleanly without the file — the repo stays green, and anyone reproducing the real
numbers fetches the data themselves. Second, and larger: a refusal on one cell is a *contact*, not a
*validation*. It shows AstraCell abstains where it should; it says nothing about whether the ECM is
right anywhere, because a system that refuses a real cell has still never confirmed one. What keeps
the refusal from being vacuous is §16d — the same estimator provably recovers a fault it *can*
express.

### 16b. What the real-cell result does, and does not, mean

The result exists now, and it is **contact, not validation**:

- **One cell of eight**, one chemistry, one duty cycle. A single cell cannot separate a property of
  *this* cell from a property of the observer.
- **The observer is still a first-order Thevenin ECM.** Everything §12 and §14 say about model-order
  bias applies with more force to a real cell, which has hysteresis, path dependence, and solid-phase
  diffusion the ECM cannot express. A real cell should mismatch the ECM at least as badly as PyBaMM
  did (§14b).
- **The fit is isothermal.** The observer ignores the 40 °C thermal history and the temperature
  dependence the dataset actually carries in its `T` channel; `dOCV/dT` is set to zero because a
  room-temperature pseudo-OCV sweep does not measure it.
- **The baseline is a real but imperfect fingerprint.** §15d named the identical-simulation baseline
  as impossible in a workshop; the dataset supplies the honest alternative — the earliest
  characterisation age — but that age is a *different day and temperature history* from every later
  one, so the paired differential now carries baseline drift the synthetic control never had. That
  is more realistic and strictly harder, not a free upgrade.
- **The ECM cannot represent the ends of the SOC range.** The observer re-integrates only within
  `SOC ∈ [~0.02, 0.98]`, so the paired window discards the top ~2 % of charge a full 1C discharge
  spans, and must stay above ~0.03 at the other end. A real full-discharge curve lives partly where
  the model structurally cannot follow it — itself a source of lack-of-fit, expected and not hidden.
- **No fault is injected or detected.** The dataset's ageing is real capacity fade scored against its
  own measurement, not a labelled fault-detection benchmark. AstraCell is being asked whether it
  *knows what it cannot see* on a real cell, not whether it can classify a named fault.

### 16c. The run corrected the Readme — the file, not the docstring, is the authority

"The scales are overridable should the file surprise us" was not idle: it did. The Readme states time
`t` is in **seconds**; the file stores a **MATLAB datenum in days** — Cell1's first sample is
735954.82, and day 735954 is 2015-01-08, exactly the Readme's own "Start date of tests". Read as
seconds, the derived 1C current came out near **64000 A** and the fit window collapsed to zero; the
run caught in one line what the Readme got wrong. The scale is now `_TIME_TO_S = 86400`, pinned in
`tests/test_oxford.py` against that embedded date, alongside the mAh→Ah and `dq/dt`→amps conversions.
The file also proved to be an old-format `.mat` that scipy reads directly (not the v7.3/HDF5 the
Readme implies), and its slow pseudo-OCV repeats SOC where the 1 Hz charge counter holds, so
`pseudo_ocv_curve` collapses exact duplicates before building the table. Every one of these was
invisible to the synthetic tests and visible on first contact with the file — which is the whole
argument for running against real data rather than trusting its documentation.

### 16d. Why the refusal means something rather than nothing

§15 is the cautionary tale: v0.3 refused everything and could not tell, because a refusal is only
evidence if the *same* machinery provably diagnoses a fault it can express. That positive control
exists and travels with this work. The identical paired estimator recovers an in-span capacity change
on ECM-generated data to within its curvature (`tests/test_oxford.py::test_adapter_pair_is_self_
consistent`), and recovered a real PyBaMM contact-resistance fault at `+20.0000%` (C14). So the
refusal on Cell1 is not the refuse-everything failure of v0.3: it happens because a real cell presents
a change this observer genuinely cannot express — lack-of-fit 91 → 815, not the ≈0 of an expressible
change. And the per-age-OCV control against the shared-OCV headline (both in `examples/08`) does its
job of catching a stuck "refuses always" gate: the two modes *disagree* in magnitude (+10.5% vs
+17.1% at end of life), so the machinery is discriminating, not frozen. What it discriminates toward
is still refusal — correctly, on a cell it cannot model — but it is refusal with its eyes open.
