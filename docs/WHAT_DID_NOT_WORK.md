# What did not work

A record of the claims this repository made and then had to withdraw, and the bugs
that changed a conclusion rather than merely a number. It is kept deliberately and
kept prominent. AstraCell exists to refuse to overclaim; a development history that
hides its own retractions would be the first overclaim.

Every item below is pinned by a test or an example, so none of it can quietly stop
being true. Where a number appears, it is reproduced by the command named next to it.

---

## 1. The AR(1) whitening bug - a sign that inverted a conclusion

**The bug.** Whitening an AR(1) noise process applies the Cholesky factor of `R⁻¹`.
The `sqrt(1 - rho²)` scaling sat on the **wrong row** of that factor, which multiplied
every information number by `(1 - rho²)` and, worse, inverted which parameters gained
and which lost under correlation.

**What caught it.** `tests/test_noise_correlation.py` cross-checks `whiten_ar1` against a
densely-inverted AR(1) correlation matrix - `xᵀ R⁻¹ x` computed both ways must agree.
It did not. The bug could not survive a test that recomputed the same quantity a
different way. A test that had merely pinned an observed SNR would have "passed" and
preserved the error.

**What it cost.** A retraction, below (§2), and a *reversed recommendation*: in one
configuration the honest engine says "run a pulse test," where the buggy one had said
"add a thermocouple." Reproduce: `python examples/02_noise_robustness.py`.

---

## 2. "Every idealisation makes the bound optimistic" - false in both directions

`LIMITATIONS.md` §2 used to argue that every modelling simplification inflates the
Fisher information, so real performance is *"worse than reported, never better,"* and
that this was the safe direction for a system built to abstain.

**It was an estimate, never measured, and it is false.** Relaxing the white-noise
idealisation to AR(1) at `rho = 0.99` moves the Cramér-Rao bound, relative to white
noise, to (`python examples/02_noise_robustness.py`):

| parameter | CRLB vs white noise at `rho = 0.99` |
|---|---|
| capacity | **×10.2** (worse - a slow SOC ramp is destroyed by differencing) |
| `R0` | **×0.39** (better) |
| `hA` | **×0.38** (better) |

The error was not conservative. It was wrong about the sign, in both directions at
once. Correlated noise does not uniformly hurt - **it reallocates**, and pulsed
excitation is lock-in detection that buys immunity to 1/f noise. The claim is now a
measurement, not a footnote.

---

## 3. D-optimality was the wrong planning objective

The first experiment planner maximised `det(FIM)` - **D-optimality**, the textbook
choice. It crowns the wrong test. For a cooling fault on the uninstrumented cell 10
(`python examples/03_next_best_test.py`):

| test | cost | `EIG_D` (all params) | `EIG` (target only) | SNR after |
|---|---:|---:|---:|---:|
| `pulse_2C_180s_cooldown` | 900 s | **2.90 nats** | 1.35 | 5.39σ |
| `pulse_train_2.5C` | 600 s | 2.63 | **1.74 nats** | **7.96σ** |

D-optimality crowns the cooldown because its 900 seconds sharpen `R0` and capacity -
parameters nobody asked about. **Ds-optimality** (the target's own marginal) crowns the
pulse train, and the realised SNR settles it, 7.96σ vs 5.39σ. *You are diagnosing one
thing; optimise that axis, not the ellipsoid volume.*

And a deeper defect, found only in v0.1 (§6 below): even the Ds planner **optimises
variance and cannot see bias.** It will recommend a hard pulse train to sharpen a
capacity estimate while driving that estimate's credibility ceiling from 2.26σ to 0.26σ
(`python examples/04_model_mismatch.py`, part 4). A planner that sharpens an answer into
a confident lie is a real defect, not a nuance.

---

## 4. The notebook output trap

The notebooks are generated from `scripts/build_notebook.py`, which writes **source-only**
cells and strips outputs. A committed notebook with stripped outputs renders as a blank
page on GitHub - figures gone, tables gone - while looking fine locally to whoever just
ran it. For a repository whose entire value is legible evidence, that is a silent
failure of the worst kind.

**The guard.** `make notebook` regenerates source (stripping outputs); `make notebook-run`
executes them in place and restores the outputs; and `tests/test_notebooks.py` enforces
**both** halves: a committed notebook must be in sync with its generator *and* carry
executed outputs whose last run raised no error. Edit the builder, never the `.ipynb`,
and always follow `make notebook` with `make notebook-run` before committing.

---

## 5. The PyBaMM phantom fault's magnitude is not robust - only the refusal is

Fitting the first-order ECM to a healthy PyBaMM cell yields a phantom capacity estimate
of **−67.6%** (`python examples/06_external_plant_gate.py`). An earlier draft of
`calibration/external.py` asserted this bias was *"robust to the R1/C1 choice."*

**Measurement falsified it, and the claim was retracted** - in the code and in
`docs/EXTERNAL_PLANT.md` §3d. The phantom swings **+114% → −68%** (changing sign) with
mean C-rate, and **−94% → +5%** as the observer's RC branch is retuned toward the
diffusion timescale. Its instability *is* the proof that it is model mismatch and not a
capacity loss. What is robust - at every C-rate and RC tuning measured - is the
*inequality*: the bias exceeds the CRLB σ by more than 30×, so the variance-only
interval overclaims and the gate refuses. **Quote the direction and the refusal, never
the −67.6%.**

---

## 6. A first-order ECM cannot calibrate capacity, and no amount of data fixes it

This is the single most important negative result in the project. Fit a first-order ECM
over a ~20-minute window to a plant with an unmodelled diffusion branch, and the slow
polarisation droop of a few millivolts is arithmetically **indistinguishable from
coulombs that never left the cell**. The observer manufactures the very fault it is
hunting:

| plant | true capacity fault | manufactured (biased) estimate | source |
|---|---|---|---|
| internal higher-fidelity plant | −5% | **−18.5%** (≈ 4× the fault) | `examples/04_model_mismatch.py` |
| PyBaMM SPMe, healthy cell | 0% | **−67.6%** at 466σ | `examples/06_external_plant_gate.py` |

Two escapes fail:
- **More data.** The bias `b = FIM⁻¹Sᵀ Σ⁻¹ r` is *exactly invariant* to replication -
  10 000 repeats move the reported SNR from 146σ to 14 611σ and the credible SNR not at
  all (it sits at 6.49σ throughout). Tested bit-for-bit in `tests/test_mismatch.py`.
- **Harder excitation.** It only **routes** the bias between parameters: 0.25C → 2.5C
  drives `R0`'s bias through zero and capacity's from −2.2% to −19.3%.

The honest response was to refuse: `REFUSE_MODEL_BIAS` is the only gate whose
recommendation is never "collect more data." Calibration then made the numbers *worse* -
the capacity fault v0.0 diagnosed at 32.6σ is now correctly refused, because 30 of those
sigmas were the observer's own model error (`python examples/05_calibrated_abstention.py`).

---

## 7. v0.3's external gate refused everything - and only a positive control could tell

v0.3 reported that calibrated abstention "survives" an external PyBaMM plant, because it
returned `REFUSE_MODEL_BIAS` on a healthy cell in 100% of trials. **The refusal was
unconditional.**

`calibration.external.prepare_external` projected the very trace the estimator was about
to fit, so its `structural_bias` was algebraically the expected estimate, `b ≡ E[δ̂]`.
Substituted into the credibility statistic, that gives a **pure noise statistic** with no
dependence on the battery at all:

```
SNR_total = |b + ε| / sqrt(σ² + b²)        E[SNR_total²] = (b² + σ²)/(σ² + b²) = 1
```

Measured (`python examples/07_external_positive_control.py`, act 1): 300/300 refusals on
20 mV of sine wave, on an absurd 1 V ramp implying a **1718%** capacity deviation, and on
a PyBaMM cell whose **series resistance had doubled** - where it cheerfully labelled the
91.4% real fault "bias". A **negative control** (a healthy cell, correctly not diagnosed)
is passed by a system that never diagnoses anything; v0.3 shipped one of those and could
not have known. Only a **positive control** - injecting a fault the observer *can*
express and checking it is recovered - reveals it. v0.4 built one. The trap is now pinned
by `test_the_v03_external_gates_credibility_statistic_is_pure_noise`.

The lesson generalises: **a negative control must never ship without its positive
counterpart.** Every headline number in v0.3 reproduced; the meaning of its refusal did
not.

---

## 8. Fitting the fast dynamics did not de-confound the capacity verdict - H1, pre-registered and falsified

v0.8 proved a *fixed* second RC branch is invisible to the `(R0, capacity)` fit (0/208
verdicts), and pinned the conclusion that model order can only matter if the added
dynamics are *fitted*. v0.9 pre-registered that fit as a testable hope - **H1**, in
`docs/V0.9_PLAN.md`, written before the run: turning the paired estimator into a
4-parameter fit `(R0, capacity, R1, C1)` would absorb the dynamic misfit polluting the
capacity fit and slide the phantom from **+10%** toward the true **−24%**, or drop the
lack-of-fit materially - the first directionally-sane real-cell estimate.

**Measurement falsified H1, and it is retracted.** Across all eight Oxford cells
(`python examples/08_real_cell.py`), fitting `R1,C1` changed the capacity verdict on
**1/208** scored ages - and that one change is `REFUSE_MODEL_BIAS → REFUSE_UNOBSERVABLE`
(Cell5, cyc2300, estimate `+0.64% → +0.03%`, below the noise floor), **not** a diagnosis.
No age became `DIAGNOSE` or `WEAK`. Cell1's phantom barely moved, `+10.50% → +9.52%`; the
largest capacity move anywhere was **3.57%**; the lack-of-fit ratio (4-param / 2-param)
had median **0.960**. The misfit is essentially untouched because the dominant residual is
the **moving OCV**, which fitting the fast branch does not address (the pre-registered
**H0**).

**The confounding H2 predicted is real, but quarantined to `R1`.** A 1C discharge cannot
identify the fast branch - **VIF(R1) median ≈ 287 ≫ 10** - exactly the information-poverty
H2 warned of. But it does not leak into capacity: capacity's VIF stays ~4 and its CRLB σ
inflates only ×**1.025**, so the capacity refusal never flips to `REFUSE_CONFOUNDED`. The
mechanism is confirmed; its hoped-for (H1) *and* feared (H2-on-capacity) effects on the
verdict are both absent.

**What made this interpretable - the positive control passed first** (the v0.4 lesson,
§7). On ECM data with a pulse train, the same 4-parameter gauss-newton recovers injected
`R1 0.15 → 0.1500` and `C1 0.30 → 0.3000` with zero cross-talk, and the same `R1` fault is
`DIAGNOSE`d under pulses yet `REFUSE_CONFOUNDED` under constant current
(`python -m pytest tests/test_dynamics_fit.py`). So the real-cell null is not a broken
fit: it is a true statement that a 1C discharge lacks the excitation to identify the
branch, and that the phantom is OCV drift regardless. v0.9 *sharpens* v0.8 - fixed branch
invisible, fitted branch inert - rather than overturning it.

---

## What survived every attempt to break it

For balance - these were attacked and did not break:

- **`REFUSE_UNOBSERVABLE` fires first and is always right.** Adding model bias to a
  hypothesis that was already refused for lack of signal changes nothing. Where AstraCell
  was silent, it stayed correctly silent.
- **Resistance faults are identifiable at every noise correlation tested** (`rho` from 0
  to 0.99).
- **The physics identities hold to machine precision** - the energy balance
  `I·(OCV − V) ≡ I²R0 + I·v1` to 1.8 × 10⁻¹⁵, asserted in
  `tests/test_physics_invariants.py`.
- **Adding a sensor never increases the CRLB** (Loewner monotonicity), a theorem, tested
  rather than observed.
