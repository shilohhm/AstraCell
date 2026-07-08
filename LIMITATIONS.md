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

## 2. The Fisher information is an optimistic upper bound

The CRLB says: *no unbiased estimator can do better than this.* Every modelling
simplification below inflates the information, so the true achievable performance
is **worse** than reported, never better. That is the safe direction for a system
whose purpose is abstention, but it is not free — it means a cell rendered green
might still be undiagnosable in practice.

Specifically:

1. **Noise is assumed white.** Real AFE noise has a 1/f component; successive
   samples are correlated; the effective independent sample count is lower than
   `n_time`. The FIM scales linearly with sample count, so this is a direct
   inflation.
2. **The current is treated as known.** In reality the pack current is measured
   with ~0.5% error, and that error is *common-mode*: it perturbs every cell's IR
   drop identically. It should be carried as a nuisance parameter that steals
   information from every resistance estimate. It is not. `NoiseModel` models the
   current sensor but the observability layer does not marginalise over it.
3. **The model is assumed correct.** The FIM measures *parameter* uncertainty
   under a known model structure. It says nothing about model mismatch. A real
   observer runs a lower-fidelity model than the plant, and the resulting residual
   bias does not appear anywhere in a Cramér–Rao bound.
4. **Estimators are assumed unbiased.** The CRLB does not bound biased estimators.
   Regularised, thresholded, or ML estimators routinely beat it in MSE by trading
   bias for variance. "Below 2σ" therefore means "no *unbiased* estimator sees
   this", which is the right conservative reading but not the only one.
5. **Sensitivities are local.** Central finite differences at `eps = 1e-3`
   relative. A 40% capacity fault is not a small perturbation, and the linearised
   bound will misstate detectability for large faults — in an unknown direction.

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
