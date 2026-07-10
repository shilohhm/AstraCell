# The external positive control (v0.4)

> v0.3 proved AstraCell refuses. It could not prove AstraCell refuses *for a reason*.

A negative control — a healthy cell, correctly not diagnosed — is passed by a diagnostic that
never diagnoses anything. v0.3 shipped one of those and did not know it. This document is how that
was found, what replaced it, and how much of the replacement is real.

Everything here is produced by `examples/07_external_positive_control.py` and pinned by
`tests/test_positive_control.py`. Numbers are measured on the 600 s, 0.5C-mean pulse window; where
they depend on that choice, it says so.

---

## 1. What a positive control is for

v0.3's question was *does the observer invent a fault that is not there?* Yes: **−67.6% capacity,
466σ, on a perfectly sound PyBaMM cell.** And the bias gate refused it, driving harmful overclaim
from 100% to 0%.

v0.4's question is the complement, and it is the one that decides whether any of that meant
anything: **when the cell really is broken, does the observer find it?** Four scenarios, one plant,
one estimator:

| Scenario | Injected into PyBaMM | True `(δ_R0, δ_cap)` | Required |
|---|---|---|---|
| healthy | nothing | `(0, 0)` | find nothing |
| real fault | contact resistance `+ΔR` | `(ΔR/R₀, 0)` | **diagnose, correct magnitude** |
| weak fault | contact resistance, tiny `ΔR` | small | weak, or refuse |
| confounded | cathode diffusivity `×0.3` | `(0, 0)` | **refuse** |

---

## 2. The finding: v0.3's credibility statistic was pure noise

`calibration.external.prepare_external` computed the structural bias like this:

```
structural_bias = FIM⁻¹ Sᵀ Σ⁻¹ (plant_voltage − f(θ*))
```

and `estimator.fit_linear` computed the estimate like this:

```
δ̂           = FIM⁻¹ Sᵀ Σ⁻¹ (plant_voltage + ε − f(θ*))
```

These are the same projection of the same trace. So `b ≡ E[δ̂]`. Substitute that into the credibility
gate's statistic and it collapses into something with no dependence on the battery at all:

$$\mathrm{SNR}_{\text{total}} = \frac{|b + \varepsilon|}{\sqrt{\sigma^2 + b^2}},
\qquad \mathbb{E}\!\left[\mathrm{SNR}_{\text{total}}^2\right] = \frac{b^2 + \sigma^2}{\sigma^2 + b^2} = 1
\quad\textbf{exactly, for any } b$$

**It is a pure noise statistic.** Act 1 of example 07 measures it on three residuals:

| residual fed to the gate | implied bias | mean | RMS | max | refusals |
|---|---|---|---|---|---|
| 20 mV of sine wave | +1.18% | 0.9831 | 0.9913 | 1.36 | **300/300** |
| a 1 V linear ramp (absurd) | **+1718.42%** | 1.0000 | 1.0000 | 1.00 | **300/300** |
| a 1 µV whisper | −0.00% | 0.7685 | 0.9730 | 3.30 | 0/300 |

Not a battery, not a fault, not physics. Whenever `|b| ≫ σ` — the only regime in which a bias gate
has a job — the statistic concentrates on 1σ and `REFUSE_MODEL_BIAS` is *guaranteed*. The 1 µV row is
the degenerate other end: `b → 0`, so it collapses to `|ε|/σ` and merely re-tests the noise.
Uninformative in both directions.

And the sharpest case, from act 3: point v0.3's gate at a PyBaMM cell whose **series resistance has
doubled** (+25 mΩ, a catastrophically corroded tab):

```
bias == estimate   +91.44%
SNR_total          1.0000 sigma
verdicts           300/300 REFUSE_MODEL_BIAS   <- it calls the fault "bias"
```

This was *correct behaviour* on v0.3's terms. Its plant was always healthy, so the entire estimate
really was bias, and calling it bias was right. But the same computation on a degraded cell calls the
degradation "bias" and refuses to see it. **v0.3's external gate had no discriminative power, and no
experiment in v0.3 could have revealed that.** Only a positive control can.

The trap is now documented at `prepare_external`, and pinned by
`test_the_v03_external_gates_credibility_statistic_is_pure_noise`. The function keeps its behaviour —
it is correct for the healthy-plant case it was written for — and gains a `baseline_voltage=`
argument that fixes it.

### 2a. Which of v0.3's claims survive

| v0.3 claim | Status |
|---|---|
| A healthy PyBaMM cell makes the ECM report ≈ −67.6% capacity at 466σ | **Stands.** Reproduced. |
| The variance-only interval covers the truth essentially never | **Stands.** |
| Harmful overclaim goes 100% → 0% with the gate | **Stands.** |
| The gate returns `REFUSE_MODEL_BIAS` in 100% of trials | **Stands, but was unconditional.** |
| Therefore calibrated abstention "survives" an external plant | **Corrected.** It survived by refusing everything. |

Under v0.4's honest gate (§3), the same healthy cell scores `SNR_total = 3.24σ` on the capacity
hypothesis: `WEAK_EVIDENCE`, not `REFUSE_MODEL_BIAS`. Harmful overclaim is still 0% — `WEAK` is not
`DIAGNOSE` — so the headline result is **corrected, not retracted**. But "100% refusal" was an
artifact of a tautological bias, and the true margin is one notch thinner than v0.3 reported.

---

## 3. The fix, and why the two obvious fixes are one fix

A bias must be measured on a plant known to be healthy. Given a healthy baseline `g_h` and a
faulted trace `g_f` on the same excitation, the task's brief offered two corrections:

1. **Baseline residual subtraction** — fit `g_f`, subtract `b_h = FIM⁻¹ Sᵀ Σ⁻¹ (g_h − f(θ*))`.
2. **Paired healthy/faulty comparison** — fit `g_f − g_h` directly.

They are **the same estimator**. The projection is linear:

$$\mathrm{FIM}^{-1}S^\top\Sigma^{-1}(g_f - f) \;-\; \mathrm{FIM}^{-1}S^\top\Sigma^{-1}(g_h - f) \;\equiv\; \mathrm{FIM}^{-1}S^\top\Sigma^{-1}(g_f - g_h)$$

Measured residual of the identity: **6.4 × 10⁻¹⁶**. There was never a choice to make, and
`test_baseline_subtraction_and_paired_comparison_are_the_same_estimator` asserts it to 1e-12.

What the identity does buy is the **honest noise**. Two measured traces carry two independent
draws, so the differential has covariance `2Σ`. `paired_noise` doubles the variance exactly — not
by scaling `voltage_sigma_v` by √2, which would leave the quantisation term `lsb²/12` behind and
desynchronise the sampler from the FIM — and the CRLB widens to `√2·σ`. **Subtraction is not free.**

The third option in the brief, an *explicit mismatch prior*, is what v0.2's `parameter_bias` already
is (`b` computed for a plant we guessed). It is unavailable here by construction: the whole point of
an external plant is that we did not write it. The fourth, *conservative refusal when no baseline
exists*, is the correct behaviour and is what v0.3 does — usefully, once you know that is what it is
doing.

### 3a. What gates the corrected estimate

Not `parameter_bias` of the differential: that is the estimate again, and refuses everything again.
Not zero: that would assert the ECM is right about how a fault looks, which is the sin this
repository exists to prevent.

Decompose the whitened structural residual against the projector `P` onto the sensitivity columns:

- The **in-span** part *is* the estimate. Any bias built from it forces `SNR_total ≤ 1`.
- The **out-of-span** part `ρ̃ = (I−P)Δg` is exactly independent of the fault — `S·δ_true` lies in
  the span, so subtracting it leaves `ρ̃` untouched. It is measurable without knowing the truth, and
  it says: *this much of what I see, no setting of my parameters can produce.*

Convert it to parameter units with the tight Cauchy–Schwarz scale (`observability.bias.lack_of_fit_bias`):

$$b^{\text{lof}}_i = \lVert \tilde\rho \rVert \cdot \sigma_i, \qquad \sigma_i = \sqrt{[\mathrm{FIM}^{-1}]_{ii}}$$

Three properties, all tested:

- **Invariant to the noise scale.** Halve every sensor's σ: `‖ρ̃‖` doubles, `σ_i` halves, `b_lof` does
  not move. A structural error must not improve when you buy better sensors — the same invariance
  `parameter_bias` has, and the reason this may occupy the same slot in `decide`.
- **`bias_ceiling = SNR_var / ‖ρ̃‖`.** The model must fit the change at least as well as the fault is
  loud.
- **Exactly zero** when the change is expressible, so the gate stands aside rather than taxing a
  fault the observer genuinely understands.

---

## 4. Measured results

### 4a. The primary fault is exactly a series resistance

PyBaMM's `Contact resistance [Ohm]` is a corroded tab weld or a degraded interconnect: one scalar,
no extra solve cost. Measured against the healthy trace:

```
max | (g_f − g_h) + ΔR·I |  =  3.6 × 10⁻¹⁶ V
```

Machine precision. And the ECM's R0 sensitivity is `∂V/∂R₀ = −I` exactly, so the differential lies
*exactly in the observer's model span*. The paired estimator recovers it with zero cross-talk:

| ΔR | ΔR/R₀ | paired `δ̂[R0]` | paired `δ̂[cap]` | `‖ρ̃‖` |
|---|---|---|---|---|
| 0.5 mΩ | +2% | **+2.0000%** | +0.000000% | ~1e-14 |
| 5 mΩ | +20% | **+20.0000%** | +0.000000% | ~1e-13 |
| 25 mΩ | +100% | **+100.0000%** | +0.000000% | ~1e-12 |

**This exactness is the definition of a positive control, and it is the reason the control is
worth nothing on its own.** If the pipeline could not recover a fault the observer's own model can
express, no result about a fault it cannot express would mean anything. §5 is where the price is
paid.

![two real faults, one in the model's span and one outside it](../reports/figures/positive_control_traces.png)

The middle panel is the whole argument for the primary fault: the measured differential lies exactly
on `−ΔR·I(t)`, which is exactly the ECM's `R₀` sensitivity. The bottom panel is the whole argument
for the confounder: a slow, drifting shape that no first-order RC branch can produce at any
parameter setting. Note too that the diffusivity fault leaves the two cells identical at `t = 0` and
parts them only as transport lags — while a contact resistance bites on the first sample. One is
dynamics; the other is algebra.

### 4b. Raw vs baseline-corrected, on a real 20% fault

| | raw (v0.3 path, honest gate) | paired (baseline-corrected) |
|---|---|---|
| estimate | +11.44% | **+20.0000%** |
| σ | 0.059% | 0.084% (√2 wider) |
| diagnosis rate | 0.00 | 1.00 |
| true-positive rate | 0.00 | **0.94** |
| coverage at 95% | 0.00 | **0.94** |
| verdict | `REFUSE_MODEL_BIAS` | `DIAGNOSE` |

The raw estimate is the fault plus the R0 phantom (`−8.56%`), and `11.44% = 20% − 8.56%`. Its gate
refuses, correctly, because it cannot tell which part is which.

![the phantom displaces every raw estimate](../reports/figures/positive_control_estimates.png)

Every distribution here is narrower than a pixel, and that is the danger, not a drawing artefact:
these estimates are exquisitely precise, and two of the three are wrong.

### 4c. A loud fault makes the raw path confidently wrong

Double the series resistance (ΔR = 25 mΩ) and the raw path is loud enough to clear its own
phantom's gate: `SNR_total = 10.7σ`. It then **diagnoses in 100% of trials, reporting +91.4% against
a truth of +100%** — an 8.6-point miss inside a 0.23%-wide interval.

| | diagnosis | true positive | harmful overclaim |
|---|---|---|---|
| raw, 100% fault | 1.00 | **0.00** | **1.00** |
| paired, 100% fault | 1.00 | 0.94 | 0.06 |

Right about the fault, wrong about the fault. This is why `detection_metrics` scores a true positive
as *DIAGNOSE ∧ interval covers the truth ∧ correct sign*: naming a fault and mis-sizing it by 145σ
is not detection.

### 4d. The magnitude sweep, and where the transition sits

Paired path, 400 trials per point. `b_lof` is zero to floating-point dust throughout, so the
credibility gate is inert and the verdicts are pure Cramér–Rao.

| ΔR [mΩ] | ΔR/R₀ | SNR | diagnose | true pos | weak | refuse |
|---|---|---|---|---|---|---|
| 0.013 | 0.05% | 0.6 | 0.00 | 0.00 | 0.07 | 0.93 |
| 0.025 | 0.10% | 1.2 | 0.00 | 0.00 | 0.17 | 0.82 |
| 0.050 | 0.20% | 2.4 | 0.00 | 0.00 | 0.64 | 0.36 |
| 0.075 | 0.30% | 3.6 | 0.06 | 0.04 | 0.88 | 0.06 |
| 0.125 | 0.50% | 6.0 | 0.82 | 0.80 | 0.18 | 0.00 |
| 0.250 | 1.00% | 12.0 | 1.00 | 0.94 | 0.00 | 0.00 |
| 5.000 | 20.0% | 239.1 | 1.00 | 0.94 | 0.00 | 0.00 |
| 25.000 | 100% | 1195.4 | 1.00 | 0.94 | 0.00 | 0.00 |

The transition sits exactly where the CRLB puts it: `σ(ΔR) = 0.021 mΩ`, so 5σ is 0.105 mΩ, and
diagnosis begins there. Nothing about the 20 mV phantom enters — the differential cancelled it.

**The true-positive rate saturates at 0.94, not 1.00, and that is correct.** A 95% interval misses
its truth 5% of the time by construction, so 5% of confident, *correct* diagnoses are counted
harmful overclaims. Harmful overclaim floors at `1 − coverage_level` for any detector that
diagnoses at all. v0.3's 0% overclaim came from 0% diagnosis; it is not a number to aspire to.

### 4e. The confounder, refused — by one percent

Cathode particle diffusivity `×0.3`: particle cracking, surface reconstruction. **No lithium and no
active material is lost, so the true `(R0, capacity)` deviation is `(0, 0)`.** That is not asserted
from the parameter we set — it is measured. The faulted pseudo-OCV shifts by 9.6 mV RMS at C/20,
which a first draft of this document wrongly called "unchanged". But the shift is **linear in
current** (199 mV per C, over a 10× rate range), hence an overpotential, hence extrapolating to zero:
the equilibrium OCV–SOC relation, and with it the coulombic capacity, is untouched. A thermodynamic
capacity loss would shift the curve by a fixed amount at every rate, including zero.

Under a 0.5C pulsed load the same fault moves the voltage by 49.9 mV RMS, and the observer reads it as:

| hypothesis | paired `δ̂` | σ | SNR (variance) | `b_lof` | SNR (total) | verdict |
|---|---|---|---|---|---|---|
| R0 | +13.19% | 0.084% | **157.7σ** | 24.48% | **0.54σ** | `REFUSE_MODEL_BIAS` |
| capacity | −118.77% | 0.205% | **579.2σ** | 60.01% | **1.98σ** | `REFUSE_MODEL_BIAS` |

Variance alone would diagnose a 13% resistance fault at 158σ and a 119% capacity loss at 579σ, on a
cell whose resistance and capacity are exactly nominal. Without the gate, the false-positive rate is
**1.00**. With it, **0.00**.

**And note the margin: 1.98σ against a 2.00σ threshold.** The capacity hypothesis is refused by one
percent. Shorten the window to 300 s and the same hypothesis reaches **3.35σ** and is merely
`WEAK_EVIDENCE`. The R0 hypothesis refuses comfortably at both. This is pinned as a regression test,
because it is the kind of result that would otherwise quietly stop being true.

### 4f. The scenarios, side by side

All baseline-corrected, 400 trials each. A dash marks a rate that is undefined for that scenario:
a null has no true positives, and a faulted cell has no false ones.

| scenario | TPR | FPR | refusal | weak | overclaim |
|---|---|---|---|---|---|
| healthy (null) | — | **0.00** | 0.95 | 0.04 | 0.00 |
| weak fault (+0.05 mΩ) | 0.00 | — | 0.36 | 0.64 | 0.00 |
| real fault (+5.00 mΩ) | **0.94** | — | 0.00 | 0.00 | 0.06 |
| confounded (D ×0.3), R0 hypothesis | — | **0.00** | **1.00** | 0.00 | 0.00 |
| confounded (D ×0.3), capacity hypothesis | — | **0.00** | **1.00** | 0.00 | 0.00 |

![finds the fault it can express, refuses the one it cannot](../reports/figures/positive_control_rates.png)

The gate does exactly one thing, and does it only when needed. Against a fault the observer can
express, the bias-aware SNR *equals* the variance-only SNR at every magnitude — the lack of fit is
zero, so there is nothing to object to. Against the confounder, the same statistic cuts 579σ of
confidence down to 1.98σ:

![the gate stands aside, then bites](../reports/figures/positive_control_snr.png)

---

## 5. What this does *not* establish

This is the section to read before quoting anything above.

- **The lack-of-fit bias is a screen, not a bound.** It measures the model's error in the directions
  the estimate does *not* live in. On the one case where the truth is known — the healthy cell,
  where the entire −67.6% estimate *is* bias — it reports 20.9%: it captures **31% of the structural
  error it is warning about**. It caught the overclaim anyway. It would not have caught one three
  times smaller.

- **A structural error lying entirely in the observer's span is invisible to it**, and to everything
  else. `‖ρ̃‖ = 0` means the model reproduces the change exactly, so attributing it to a parameter is
  the *only* thing the data supports — whether or not that is what physically happened. This is a
  theorem about the experiment, not a defect of the implementation. Only a different excitation or a
  richer model separates such a change from a real parameter shift.

- **The primary fault is in the model's span by construction.** That is what makes it a control, and
  what makes it silent about faults that are not. A contact resistance *is* what an ECM's `R0` is.
  Recovering it to 4 decimal places demonstrates the plumbing, not the physics. The confounder is
  where the physics is, and there the margin is 1%.

- **The baseline here is unrealistically perfect.** `g_h` is the *identical simulation* with the
  fault parameter at its healthy value: same solver, same grid, same initial SOC, same day. A real
  beginning-of-life fingerprint is none of those. Every number in §4 is therefore an **upper bound
  on what any real baseline could deliver**, and the paired estimator's nominal coverage is a
  statement about the arithmetic, not a promise about a workshop.

- **No baseline, no correction.** Where a comparable healthy trace does not exist, the honest
  behaviour is v0.3's: refuse. That is not a failure mode; it is the correct answer to a question
  the data cannot support. It is only a problem when it is mistaken for discrimination.

- **PyBaMM is still a model, not a cell.** Everything in v0.3's `docs/EXTERNAL_PLANT.md` §5 applies
  unchanged. One cell, one chemistry parameter set, one chosen duty cycle, isothermal, no ageing, no
  pack. This tests external *model* mismatch with an injected *model* fault.

- **Capacity fade is still not injected.** `Nominal cell capacity [A.h]` only normalises the C-rate;
  it changes no electrode capacity. Real fade is loss of lithium inventory or active material, and
  both move the stoichiometry window and hence the OCV–SOC map — which would break the
  shared-pseudo-OCV control that isolates dynamic mismatch in the first place. `LIMITATIONS.md` §15.

---

## 6. What it does establish

Two things, both narrow.

**The refusal now discriminates.** The same gate, on the same plant, with the same estimator:
refuses a phantom, passes a real fault at its correct magnitude with nominal coverage, weakens a
marginal one, and refuses a real physical change it cannot name. Before v0.4 it did the first of
those and could not have done the others, and nothing in the repository would have said so.

**Diagnosis is not detection.** A system can diagnose a real fault in every trial while its interval
never covers the truth — §4c does exactly that, at 10.7σ of misplaced confidence. Scoring only
overclaim rewards silence; scoring only diagnosis rewards noise. `detection_metrics` reports both,
and the price of the honest gate is legible: **0.94 true positives, not 1.00, and a √2 wider σ.**

See `LIMITATIONS.md` §15 for where this sits on the ladder of what AstraCell has and has not
established.
