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

3. **The model is assumed correct.** The FIM measures *parameter* uncertainty
   under a known model structure. It says nothing about model mismatch. A real
   observer runs a lower-fidelity model than the plant, and the resulting residual
   bias does not appear anywhere in a Cramér–Rao bound. This is the largest
   remaining gap and it is not quantified anywhere in this repo.
4. **Estimators are assumed unbiased.** The CRLB does not bound biased estimators.
   Regularised, thresholded, or ML estimators routinely beat it in MSE by trading
   bias for variance. "Below 2σ" therefore means "no *unbiased* estimator sees
   this", which is the right conservative reading but not the only one.
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
- No estimator has been implemented, so no CRLB has been shown to be attainable.
- No claim in this repository has been validated against anything but itself.

A simulation that validates a diagnostic method against the simulator that
generated the data is a tautology. The correct next step is to inject faults with
a **higher-fidelity model** than the one the observer assumes (PyBaMM DFN with
degradation submodels), so that model mismatch is part of the experiment. That is
not done here.

---

## 11. Thresholds are conventions

`2σ` for "weak" and `5σ` for "observable" are conventions borrowed from other
fields. They are exposed as arguments (`weak_sigma`, `strong_sigma`) rather than
buried, because there is no principled reason for them absent a stated cost of a
false positive versus a missed detection. The condition-number threshold for
declaring parameters confounded (`1e4`) is likewise a convention.
