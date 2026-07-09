# The external-plant gate (v0.3)

> Does calibrated abstention survive a plant AstraCell did not write?

Every version before this one measured AstraCell against a battery **we built**. v0.0's plant was
the same ECM the observer fits. v0.1 added `plant.mismatch` — an intermediate model richer than
the ECM, but its extra terms were four things *we* chose, at magnitudes *we* set. So "the observer
is simpler than the plant" was true by construction, and the size of the gap was a dial on our
side. v0.3 is the first time the plant is something we did not design: a **PyBaMM** single cell.

Everything here is produced by `examples/06_external_plant_gate.py` and pinned by
`tests/test_external_plant.py`. The numbers below are measured, not asserted; re-run the example
to reproduce them.

---

## 0. What was built first: nothing new in the estimator

The design rule for v0.3 was *do not build a second estimator*. v0.2 already has one, and a whole
apparatus around it — `build_context`, `run_trials`, coverage, abstention — resting on one
contract: a scenario yields a `ScenarioContext` carrying a `plant_output` tensor and a
`structural_bias`. `calibration.external.prepare_external` supplies exactly that contract from an
**external voltage trace** instead of from `simulate_plant`. PyBaMM fills the `plant_output` slot;
the fit, the FIM, the coverage, and the bias gate are v0.2, unchanged. The only genuinely new code
is the PyBaMM adapter (`plant.pybamm_plant`) and the OCV-table builder (`cell.ocv.ocv_from_table`)
that the `ocv` module docstring had already promised.

---

## 1. PyBaMM feasibility: what was decided, and what was cut

PyBaMM installs cleanly on Python 3.13 from wheels (no source build) and solves one cell in
0.2 s (SPM) / 0.4 s (SPMe) / 1.4 s (DFN). Because the plant runs **once per scenario** — the noise
is drawn per trial on top of a cached trajectory — even DFN would be affordable. The decisions:

| Question | Decision | Why |
|---|---|---|
| Optional or required? | **Optional** (`pip install -e '.[pybamm]'`) | The core repo stays numpy-only; identifiability, mismatch, and calibration must not need an electrochemical solver. Tests and the notebook skip cleanly without it. |
| Which model? | **SPMe** | Adds the electrolyte diffusion a first-order ECM cannot express — the interesting mismatch — while staying fast and stable. SPM understates it; DFN is available but unnecessary for v0.3. |
| Inputs / outputs? | Current profile in, **terminal voltage** out | One cell, one instrumented channel. |
| Temperature? | **Cut** | PyBaMM's default cell is isothermal, so a temperature channel is a flat line. Voltage only. |
| Pack scale? | **Cut** | One cell. Pack electrochemistry is explicitly out of scope. |
| Fault injection? | **Cut** | Not needed. A *healthy* cell already exposes the mismatch (§3), and "healthy" is a truth both models share exactly (deviation = 0). Injecting electrode-level degradation cleanly is deferred. |

The target is **capacity**, because it is the one quantity the ECM and PyBaMM agree a truth about:
the cell is healthy, so the honest capacity deviation is **zero**. The observer **shares PyBaMM's
pseudo-OCV** (a slow C/20 discharge, turned into an `OcvCurve` by `ocv_from_table`), so the static
voltage–SOC relationship is not in dispute. Whatever residual remains under load is the *dynamics*
the ECM omits — not a curve-shape disagreement we could have removed.

---

## 2. The experiment, and the control that proves it is not a bug

- **Plant.** PyBaMM SPMe, Chen2020 parameters, one 5 Ah cell, isothermal, on a 600 s pulse train
  (0.5C mean + 1C pulses at 1 s resolution), started at 90% SOC.
- **Observer.** A one-cell first-order Thévenin ECM (fitted `R0` and `capacity`; a fixed `R1`–`C1`
  branch), using PyBaMM's pseudo-OCV, `ea_over_r_k = 0` so voltage is isothermal too. `hA` is not
  fitted: an isothermal voltage-only cell cannot see it.
- **Control.** Before PyBaMM is involved, the *same pipeline* is fed an **ECM-generated** trace.
  There the model mismatch is identically zero, so coverage must be nominal. It is:

  | nominal | 50% | 80% | 90% | 95% | 99% |
  |---|---|---|---|---|---|
  | control (ECM plant, 3000 trials) | 49.1% | 80.2% | 91.1% | 95.4% | 98.8% |

  Largest deviation from nominal: **0.011**, and the structural bias is `+0.0000%`. The
  external-plant harness adds no bias of its own. **So every collapse below is PyBaMM's model
  mismatch, not our plumbing.** This is the guard against mistaking a harness bug for physics, and
  it runs even when PyBaMM is not installed.

---

## 3. Measured results

### 3a. The residual, and a phantom fault on a healthy cell

On the pulse train, PyBaMM's terminal voltage departs from the best the first-order ECM can do by
**20.65 mV RMS** — twenty times the 1 mV measurement noise. Fit capacity to that data and, on a
cell that is **perfectly healthy**, the observer reports:

> estimated capacity deviation = **−67.6% ± 0.145%** (1σ) — **466σ** from the truth of zero.

The estimate is exquisitely *precise* and grossly *wrong*. The one RC branch cannot represent the
electrolyte diffusion, so the slow concentration droop under sustained current gets absorbed by the
only parameter that can mimic a slow droop — capacity — and the observer concludes the cell has
lost two-thirds of its capacity. It has lost none.

### 3b. Coverage collapses under the real plant; the control does not

| nominal | control (ECM) | PyBaMM, variance-only | PyBaMM, bias-aware |
|---|---|---|---|
| 50% | 49.1% | 0.0% | 0.0% |
| 80% | 80.2% | 0.0% | 100.0% |
| 90% | 91.1% | 0.0% | 100.0% |
| 95% | 95.4% | 0.0% | 100.0% |
| 99% | 98.8% | 0.0% | 100.0% |

The variance-only interval — pinned to ~0.15% around a centre 67% away — covers the truth **never**.
Admitting the structural bias (widening to `z·√(σ²+b²)`) restores coverage above 80%, but only by
making the interval about ±90% wide: honest, and diagnostically useless. That uselessness is the
correct report. It says *this excitation cannot resolve capacity through this model*.

### 3c. Overclaim across excitation, and the gate

Sweeping the mean discharge rate (1C pulses, 600 s, healthy cell throughout):

| mean rate | phantom fault | σ | overclaim (no gate) | overclaim (gated) |
|---|---|---|---|---|
| 0.2C | +113.9% | 0.675% | 1.00 | 0.00 |
| 0.3C | −19.5% | 0.344% | 1.00 | 0.00 |
| 0.5C | −67.6% | 0.145% | 1.00 | 0.00 |
| 0.8C | −59.2% | 0.083% | 1.00 | 0.00 |

Without the gate, the observer diagnoses a capacity fault on the healthy cell in **every trial at
every excitation**. The bias gate — using the *observable* residual to estimate its own bias —
converts all of them to `REFUSE_MODEL_BIAS`. Harmful overclaim goes **100% → 0%**.

### 3d. The honest caveat: the phantom fault's *size* is not robust

Note the phantom fault above swings from **+114% to −68%** and even changes sign as the C-rate
changes. It is not a stable property of the cell — its very instability is the proof that it is not
a capacity loss. It also depends strongly on the observer's fixed RC branch. Retuning `R1·C1`
toward the electrolyte diffusion timescale, on the same 0.5C data:

| RC time constant | 8 s | 30 s | 120 s | 300 s |
|---|---|---|---|---|
| capacity bias | −93.6% | −67.6% | −20.3% | **+4.7%** |

A near-perfectly tuned RC nearly erases the bias. So the headline **−67.6%** is *this* observer's
number, not a universal constant, and an earlier draft of the code claiming the bias was "robust to
the R1/C1 choice" was **wrong and has been retracted**. What *is* robust: at every C-rate and every
RC tuning measured, the bias exceeds the CRLB σ (≈0.15%) by **more than 30×**, so the variance-only
interval overclaims and the gate refuses in **every** case. The robust result is the overclaim and
the refusal — never the magnitude of the phantom fault.

---

## 4. Why "it refuses" is the whole point

A CRLB-only observer reports a −67.6% capacity fault, at 0.15% precision, on a healthy cell, and
would do so more confidently the more data you gave it (v0.2's money plot, now against a real plant).
The bias gate looks at the same 20 mV structured residual, infers that the capacity estimate is
unreliable, and declines. It cannot know the true capacity — but it *can* see, from its own fit
residual, that its confidence is unwarranted. That is exactly the behaviour v0.1 and v0.2 built,
now shown to hold against a plant we did not author. Under an honest plant, **AstraCell diagnoses
less**, and that is the correct direction.

---

## 5. What this does *not* prove

- **PyBaMM is a model, not a battery.** This validation is still entirely synthetic. It swaps a
  simple synthetic plant for a sophisticated one; it introduces no measured cell.
- **This is not real-EV validation.** One cell, one chemistry parameter set, a chosen duty cycle,
  isothermal, no ageing, no pack. Nothing here speaks to a vehicle.
- **Model choice matters, and we chose it.** SPMe, Chen2020, the C/20 pseudo-OCV, the RC branch, the
  SOC window — each is a decision, and §3d shows the headline number moves with them. A different
  model or parameter set would give different magnitudes. The *direction* (precise-but-biased →
  refuse) is what we claim survives; specific percentages are illustrative.
- **It tests external model mismatch, not physical truth.** The result is that AstraCell's abstention
  logic behaves correctly when the data-generating model is richer than the observer. Whether the
  observer is right about a *real* cell is a question no simulation can answer, and the next honest
  step — a measured pseudo-OCV and a measured pulse response — is still open.

See `LIMITATIONS.md` §14 for how this fits the ladder of what AstraCell has and has not established.
