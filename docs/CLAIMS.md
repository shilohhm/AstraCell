# Claims and evidence

Every claim AstraCell makes, what tier of validation stands behind it, the evidence in
the repository, the command that reproduces it, and what it does **not** cover. If a claim
is not in this table, AstraCell does not make it.

**The three validation tiers** — the single most important distinction in this project,
and the one most battery-diagnostic work blurs:

- **Tier 1 — internal self-consistency and synthetic experiments.** Demonstrated within
  AstraCell's own models and by theorems about the estimator. True *of the model*.
- **Tier 2 — independently developed external simulator.** Tested against PyBaMM, an
  electrochemical simulator that AstraCell did not implement, whose mismatch it did not
  design. Stronger than Tier 1; still synthetic.
- **Tier 3 — physical battery validation.** A measured cell. **No validation** — v0.6 made first
  *contact*, v0.7 widened it to all eight cells (every one refused; C19), v0.8 showed the refusal
  survives a second-order observer (C20), and v0.9 that *fitting* the fast RC branch is inert on it
  too (C21). Still not validation. See the Tier 3 table, which states both the contact and the absence
  plainly.

Schema: **Claim ID · Claim · Validation tier · Evidence · Reproduction command · Limitations**.
`$PY` is the venv Python (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

---

## Tier 1 — internal self-consistency and synthetic experiments

| ID | Claim | Tier | Evidence | Reproduction | Limitations |
|---|---|---|---|---|---|
| C1 | On the demo pack, `R0` and capacity faults are identifiable on all 32 cells; a 40% cooling fault only on the 4 instrumented cells (149.2σ / 32.6σ / 5.5σ vs 1.2σ REFUSE) | 1 | `examples/01` §3–4 · `packmap_cooling.png` · `test_observability.py` | `$PY examples/01_first_demo.py` | Stand-in OCV curves, not a real cell; assumes white noise (ρ=0) and known current |
| C2 | Identifiability is **not** monotone in grid distance to a sensor — a far corner (1.55σ) beats the sensor's neighbour (1.33σ) | 1 | `examples/01` §4 | `$PY examples/01_first_demo.py` | A property of the 2D-grid thermal caricature ([LIMITATIONS](../LIMITATIONS.md) §7) |
| C3 | Adding a sensor never increases the CRLB (Loewner monotonicity) | 1 | `test_observability.py` | `$PY -m pytest tests/test_observability.py` | A theorem about the FIM, independent of model fidelity |
| C4 | `crlb()` returns `inf` for unidentified parameters — no `pinv`, so the system never claims finite variance it cannot support | 1 | `test_observability.py` · `fisher.crlb` | `$PY -m pytest tests/test_observability.py` | An implementation guarantee, not a physical claim |
| C5 | Under constant current, `R0`/capacity/`hA` become unidentifiable once the ammeter is imperfect (`R0` cost ×6.50, worst VIF 261.7) | 1 | `examples/02` §1 | `$PY examples/02_noise_robustness.py` | Synthetic; assumes a single pack-global current-bias nuisance |
| C6 | Correlated AR(1) noise **reallocates** information rather than uniformly hurting (capacity ×10.2 worse; `R0`/`hA` ×0.39/0.38 *better* at ρ=0.99) | 1 | `examples/02` §2 · `test_noise_correlation.py` | `$PY examples/02_noise_robustness.py` | Single-pole AR(1) is not real 1/f; ρ is unknown for any real pack |
| C7 | Ds-optimal experiment planning beats D-optimal for a single target (7.96σ vs 5.39σ) | 1 | `examples/03` · `test_experiment.py` | `$PY examples/03_next_best_test.py` | Neither is the decision rule (cheapest crossing is); the planner optimises variance, blind to bias (see C10) |
| C8 | The MLE attains the CRLB under a matched model — interval coverage tracks nominal | 1 | `examples/05` §1 · `test_calibration.py` | `$PY examples/05_calibrated_abstention.py` | Fixed-information M-estimator; scatter matches CRLB up to model curvature (~6%) |
| C9 | Under mismatch the variance-only interval covers the truth ~never; the model-bias gate drives harmful overclaim on capacity 100% → 0% | 1 | `examples/05` · `calibration_coverage.png` · `test_calibration.py` | `$PY examples/05_calibrated_abstention.py` | The gated bias is computed against the mismatch we wrote; a differently-shaped error might not be gated |
| C10 | A first-order ECM over a 20-min window manufactures an apparent 18.5% capacity loss; the structural bias is exactly invariant to replication and noise scale, so **more data cannot fix it** | 1 | `examples/04` §3 · `model_mismatch.png` · `test_mismatch.py` | `$PY examples/04_model_mismatch.py` | The internal plant is four hand-chosen terms — a *lower bound* on how wrong we are, not a measurement of it |
| C11 | The physics identities hold to machine precision (energy balance to 1.8×10⁻¹⁵; charge conservation; conduction matrix is a Laplacian) | 1 | `test_physics_invariants.py` | `$PY -m pytest tests/test_physics_invariants.py` | Correctness of the simulator, not fidelity to a physical cell |
| C12 | Monte Carlo is bit-for-bit deterministic under a seed | 1 | `test_calibration.py` | `$PY -m pytest tests/test_calibration.py` | A reproducibility guarantee, not a physical claim |

## Tier 2 — independently developed external simulator (PyBaMM)

| ID | Claim | Tier | Evidence | Reproduction | Limitations |
|---|---|---|---|---|---|
| C13 | Fitted to a **healthy** PyBaMM SPMe cell, the variance-only observer reports a phantom −67.6% ± 0.145% capacity deviation (466σ); an ECM self-consistency control covers at nominal (dev ≤ 0.011), proving the collapse is the external simulator's mismatch, not the harness | 2 | `examples/06` · `external_estimate_distribution.png` · `external_coverage.png` · `test_external_plant.py` | `$PY examples/06_external_plant_gate.py` *(needs pybamm)* | The −67.6% **magnitude is not robust** (swings +114%→−68%); only the refusal is. PyBaMM is a model, not a cell |
| C14 | With a healthy baseline + a lack-of-fit gate, the paired estimator recovers an injected contact-resistance fault at **+20.0000%** (TPR 0.94, nominal coverage) and refuses a confounder it cannot express (cathode diffusivity ×0.3; capacity hypothesis 1.98σ vs a 2.00σ line) | 2 | `examples/07` · `positive_control_rates.png` · `test_positive_control.py` | `$PY examples/07_external_positive_control.py` *(needs pybamm)* | Baseline is the identical simulation (impossible in a workshop) → rates are **upper bounds**; the screen captures only 31% of known bias; the primary fault is in the model's span by construction |
| C15 | v0.3's external bias gate was a **pure noise statistic** (E[SNR²]=1 for any input) that refused everything; only a positive control could reveal it | 2 | `examples/07` act 1 · `test_positive_control.py::test_the_v03_external_gates_credibility_statistic_is_pure_noise` | `$PY examples/07_external_positive_control.py` *(needs pybamm)* | A corrected result; see [WHAT_DID_NOT_WORK](WHAT_DID_NOT_WORK.md) §7 |

## Tier 3 — physical battery validation

These rows state an absence plainly: AstraCell makes **no** Tier 3 *validation* claim. C19 records
the real-cell *contact* — on all eight measured cells the ECM is directionally wrong and AstraCell
refuses — which is not validation, and does not become it by adding cells within one chemistry. C20
adds that this refusal is not a first-order artefact — a *fixed* second-order observer changes no
verdict — and C21 that *fitting* the fast RC branch is inert on the capacity verdict too, falsifying
the pre-registered hope (H1) that it would de-confound. All three are properties of the estimator
meeting real data, not validations of the ECM.

| ID | Claim | Tier | Evidence | Reproduction | Limitations |
|---|---|---|---|---|---|
| C16 | **No physical battery validates AstraCell, and no real fault has ever been detected.** v0.6 made first *contact* and v0.7 scored all eight real cells (Oxford Cell1–8; C19), on which the ECM refused every age — but that is contact, not validation: one chemistry, no fault detected, no diagnosis confirmed. No result is validated against anything but code, an electrochemical simulator, and eight real cells the observer declined to trust | 3 — none | [LIMITATIONS](../LIMITATIONS.md) §1, §10, §14, §15, §16 | see C19 for the real run; otherwise nothing — the *absence of validation* is the claim | This is the project's largest gap. Every Tier 1/2 result is conditional on models that have otherwise not touched a cell |
| C17 | The OCV curves are **stand-ins** (`NMC_LIKE` from a Li-polymer fit; `LFP_LIKE` hand-built), so every SNR and CRLB is a statement about *this model*, not a battery | 3 — none | [LIMITATIONS](../LIMITATIONS.md) §1 | — | Replace `cell/ocv.py` with measured tables before quoting any figure outside this repository |
| C18 | AstraCell claims **no** EV-level validation and **no** safety-critical deployment readiness | 3 — none | [LIMITATIONS](../LIMITATIONS.md) §14 · [POSITIVE_CONTROL](POSITIVE_CONTROL.md) §5 | — | It is a research scaffold for the identifiability question, nothing more |
| C19 | On **all eight Oxford cells** (real measured cells), the first-order ECM's capacity estimate is wrong in **sign** — every cell fades **−20% to −38%** while the deployable shared-OCV estimate reads **−0.2% to +14.3%** (a phantom *gain* on 7 of 8; the ±0.03% (1σ) interval puts Cell1's +10.5% ≈1150σ from its −24.2% truth) — and AstraCell refuses **all 104 scored ages (REFUSE_MODEL_BIAS, coverage 0/104), in both OCV modes (208/208 evaluations)**. Abstention holds on every measured cell | 3 — contact (eight cells, not validation) | `examples/08` · `plant/oxford.py` · `test_oxford.py` · `real_cell_capacity.png` · [REAL_CELL](REAL_CELL.md) | `$PY scripts/fetch_oxford.py` then `$PY examples/08_real_cell.py` *(needs the ODbL download + `.[oxford]` extra)* | Eight cells but one chemistry, first-order ECM, isothermal, shared baseline a different day; **no fault detected and the ECM is not validated — the refusal *is* the result**. Numbers reproduce from the fetched data, which is never committed. [REAL_CELL](REAL_CELL.md), [LIMITATIONS](../LIMITATIONS.md) §16 |
| C20 | The refusal is **not a first-order artefact.** Rerunning the identical eight-cell loop with a *second-order* observer — a second RC branch (fixed τ₂ ≈ 240 s), still fitting only `R0` and capacity — changes **0 of 208** verdicts; `REFUSE_MODEL_BIAS` stays 104/104 in both OCV modes and the largest first-vs-second-order gap over all 208 evaluations is **1.7×10⁻¹⁴** (lack-of-fit 2.8×10⁻¹¹), i.e. round-off. A *fixed* richer model is invisible to a fit over `(R0, capacity)`: the RC overpotentials cancel in `∂V/∂R0` and `∂V/∂Q` | 1 — internal invariance (measured on the Tier-3 run) | `examples/08` depth table · `test_second_order.py` · [REAL_CELL](REAL_CELL.md) · [LIMITATIONS §16e](../LIMITATIONS.md) | `$PY examples/08_real_cell.py` *(dataset)*; mechanism: `$PY -m pytest tests/test_second_order.py` | **Retraction:** the pre-run guess that depth would shrink the misfit is falsified. Only a *fixed* branch is tested; model order enters the verdict only by *fitting* the dynamics (`→ +R2,C2`), a 4→6-parameter problem trading bias for confounding — v0.9, not asserted here |
| C21 | **Fitting the fast RC branch is inert on the capacity verdict — H1 falsified.** Rerunning the eight-cell loop with a **4-parameter** `(R0, capacity, R1, C1)` fit — the branch now *estimated*, not fixed — changes **1 of 208** verdicts (a lone marginal age, not a DIAGNOSE), moves the capacity estimate by ≤**3.57%** (Cell1 +10.50% → +9.52%, phantom *gain* persists on 7/8), and cuts the lack-of-fit by a median ~4%. `R1` is unidentifiable from a 1C discharge (**VIF ≈ 287 ≫ 10**, H2's mechanism), but the confounding is *quarantined to `R1`*: capacity's VIF stays ~4, its CRLB inflates ~2.5%, so it stays `REFUSE_MODEL_BIAS`, never `REFUSE_CONFOUNDED`. The phantom is OCV drift, unreachable by RC fitting on a 1C discharge | 1 — internal invariance (measured on the Tier-3 run) + positive control | `examples/08` fit-dynamics table · `test_dynamics_fit.py` · [REAL_CELL §Fit-dynamics](REAL_CELL.md) · [LIMITATIONS §16f](../LIMITATIONS.md) | `$PY examples/08_real_cell.py` *(dataset)*; positive control: `$PY -m pytest tests/test_dynamics_fit.py` | **Retraction:** the pre-registered H1 (de-confounding) loses; the outcome is H0-dominant with H2 confirmed only on `R1`. The positive control shows the *same* fit recovers injected `R1,C1` **exactly** under a pulse train (VIF < 10) — the limit is the excitation, not the code |

---

## Withdrawn claims

Claims this repository made and then retracted are not listed above as active claims; they
are documented, with what falsified them, in
[WHAT_DID_NOT_WORK.md](WHAT_DID_NOT_WORK.md) — including "correlated noise is always more
conservative" (§1–2), "the phantom bias is robust to the RC choice" (§5), and "calibrated
abstention survives an external plant" in its original, unconditional form (§7).
