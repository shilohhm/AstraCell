# Glossary

Eight quantities carry AstraCell's entire argument. Definitions are exact and match
the code. Symbols: `θ` parameters, `θ*` the truth, `θ₀` the pseudo-true value a
mismatched fit lands on, `S` the sensitivity tensor `∂(measurement)/∂θ`, `Σ` the
measurement-noise covariance, `r` a model residual.

The one distinction to keep straight: **FIM, CRLB, VIF, and SNR are all about
*variance*** — how finely a *correct* model could be pinned down by noisy data.
**Model bias is about *accuracy*** — how far a *wrong* model lands from the truth
however clean the data. Everything AstraCell got wrong early, and everything the
model-bias gate exists to catch, lives in that gap. See also the three validation
tiers in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) and [CLAIMS.md](CLAIMS.md).

---

### FIM — Fisher Information Matrix

`FIM = Sᵀ Σ⁻¹ S`. How much the measurements constrain the parameters, given the
sensor topology and the noise. Two properties AstraCell leans on: it is **additive**
across independent experiments (`FIM_after = FIM_before + FIM(u)`, the basis of
experiment planning), and it is **Loewner-monotone** in sensors — adding a channel
never removes information (a theorem, tested). A sensor topology enters only as a
*row mask* over `S`, which is why counterfactual sensor placement costs a matrix
slice, not a re-simulation. Code: `observability.fisher`. → CRLB, VIF.

### CRLB — Cramér–Rao Lower Bound

`Var(θ̂_j) ≥ [FIM⁻¹]_jj` for **any unbiased estimator**. AstraCell reports
`sqrt(CRLB)` as the 1σ floor. It bounds *every* estimator, not one algorithm — which
is what makes a refusal principled rather than a limitation of a particular detector.
Computed without `pinv`: eigendecompose, discard directions at the level of
floating-point noise, and return `inf` for anything unidentified, so the system never
claims finite variance for a parameter it cannot see. **Blind to model bias** — its
central limitation. Code: `observability.fisher.crlb`. → SNR, model bias, abstention.

### VIF — Variance Inflation Factor

`VIF_j = FIM_jj · [FIM⁻¹]_jj ≥ 1`, equal to 1 exactly for an orthogonal design. How
much parameter j's variance is inflated by collinearity with the *other* parameters —
i.e. whether it is separable from the things it could be confused with. This is the
**isolation gate** (`VIF > 10` ⇒ confounded), checked *before* detection: two
parameters can be jointly well-determined while individually unidentifiable, and
reporting one of them then would be a confident lie. Not `cond(FIM)`, which is
dominated by the single worst-informed direction. → FIM, abstention.

### SNR — signal-to-noise ratio (detectability)

`SNR = |magnitude| / sqrt(CRLB)`. Under a local linearisation the CRLB does not depend
on the fault magnitude, so one simulation prices every fault size at once. Thresholds:
`≥ 5σ` diagnose, `2–5σ` weak, `< 2σ` refuse — conventions borrowed from other fields,
exposed as arguments, not physics. Variance-only by default; once model bias is
admitted the honest statistic is `SNR_total = |m| / sqrt(CRLB + b²)`, which *ceilings*
rather than growing without bound. Code: `observability.mask`. → CRLB, model bias.

### Model bias (structural bias)

`b = FIM⁻¹ Sᵀ Σ⁻¹ r`, with `r = plant(θ*) − observer(θ*)` the residual between a richer
data-generating process and the model being fitted. A wrong model does not converge on
`θ*` but on the pseudo-true `θ₀ = θ* + b`, and `b` is **bias**, not variance. It is
*exactly invariant* to the two things that shrink the CRLB — replicating the experiment
and rescaling the noise — so **more data does not reduce it** (both invariances are
tested bit-for-bit). This is the error the CRLB cannot see. Where the truth is unknown
(an external plant), `b` cannot be measured, only **screened** by the observer's
lack-of-fit to the data (`observability.bias.lack_of_fit_bias`) — a screen, not a
bound. Code: `observability.bias`. → CRLB, coverage, abstention.

### Calibration

Whether stated uncertainties match observed frequencies, measured by Monte Carlo over
many noise draws with a *known injected truth*. AstraCell's calibration establishes
**self-consistency** — that its intervals mean what they claim under the model it
assumes, and break in the measurable way when that model is wrong. It does **not**
establish that the model resembles a real cell (that would be Tier 3 validation, which
has not been done). Code: `calibration`. → coverage.

### Coverage

The fraction of trials whose claimed interval actually contains the truth; a 90%
interval should cover 90% of the time. The empirical face of the CRLB: under a
*matched* model coverage tracks nominal (the first evidence the bound is **attained**,
not merely asserted); under *mismatch* the variance-only interval covers essentially
never, because it is centred a whole bias away from the truth. This turns "the bound
holds" from an assertion into a frequency. → calibration, CRLB, model bias.

### Abstention (refusal)

Declining to answer when the question is unanswerable — AstraCell's actual product, not
a fallback. Three gates, in order: **REFUSE_UNOBSERVABLE** (SNR below threshold),
**REFUSE_CONFOUNDED** (VIF too high), **REFUSE_MODEL_BIAS** (bias / lack-of-fit too
large relative to the signal). Each refusal carries a recommendation where one exists
(instrument a cell, or excite it harder). A diagnostic that cannot say *"I cannot see
this"* is confident exactly where it should be silent. Code: `observability.decision`.
→ SNR, VIF, model bias.
