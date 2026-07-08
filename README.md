# AstraCell

**A battery diagnostic that knows what it can't see.**

AstraCell determines, from first principles, which battery faults are *identifiable*
given the battery model, the sensor topology, the measurement noise, and the excitation
actually present in the data. It refuses to diagnose the rest — and computes what
additional measurement would change its mind.

> ⚠️ **Read [`LIMITATIONS.md`](LIMITATIONS.md) before believing any number here.**
> The OCV curves are stand-ins, not fitted cells. Nothing in this repository has ever
> detected a real fault, or touched a real battery. Every bound below is a statement
> about *this model*, and every modelling simplification makes it **optimistic**.

---

## The thesis

A production BMS does not measure what you want to diagnose.

| Quantity | Real coverage | Consequence |
|---|---|---|
| Cell voltage | one per cell, ~1 mV | per-cell voltage faults **are** identifiable |
| Temperature | **4–12 sensors for ~96 cells**, ±0.5–1 K | per-cell thermal faults mostly **are not** |
| Current | **pack-level only**, one shunt | per-cell current is inferred, never measured |

So "highlight the faulty cell on a 3D pack map" is, for thermal faults, undecidable from
real telemetry. Most battery-diagnostic projects do it anyway.

AstraCell instead computes the **Fisher information matrix** of the pack's parameters
under the actual sensor topology, and the **Cramér–Rao lower bound** it implies. The CRLB
bounds the variance of *every unbiased estimator* — not of one algorithm. If it says a
40% cooling fault sits at 1.2σ, no detector you write will find it, and the honest thing
to render is grey.

**The grey cells are not painted by a distance-to-sensor rule. They fall out of the
Cramér–Rao bound.**

There is no fault classifier here, deliberately. A classifier answers *"which fault?"*.
This answers the logically prior question: *"is that question answerable?"* Building the
classifier first produces a system that is confident exactly where it should be silent.

---

## What the code found

All numbers below are produced by `python examples/01_first_demo.py` on a 4×8 pack
(32 cells, 32 voltage channels, 4 thermocouples, 1 current shunt), 1200 s of 1.0C pulse
excitation at 1 Hz, 1 mV voltage noise and 0.5 K temperature noise.

### 1. Resistance and capacity faults are identifiable everywhere. Cooling faults are not.

| Fault hypothesis | Thermocouple on that cell? | CRLB (1σ) | SNR | Verdict |
|---|---|---|---|---|
| `R0` +20% on cell 5 | no | ±0.13% | 149.2σ | **DIAGNOSE** |
| capacity −5% on cell 17 | no | ±0.15% | 32.6σ | **DIAGNOSE** |
| cooling −40% on cell 12 | **yes** | ±7.31% | 5.5σ | **DIAGNOSE** |
| cooling −40% on cell 10 | no | ±32.4% | **1.2σ** | **REFUSE** |

Cooling faults are identifiable on **4 of 32 cells** — exactly the four carrying a
thermocouple. The fault on cell 10 is *really there*; AstraCell declines to diagnose it.

![pack map](reports/figures/packmap_cooling.png)

### 2. A thermocouple informs about the cell it sits on, and almost nothing else

Conduction carries too little information to a neighbour. The cell *adjacent* to a
thermocouple is no better determined than a cell four hops away — it is **worse**,
because the far cell is a pack corner, and a corner has fewer conduction paths, so it
warms more for the same `hA` change and reads out more strongly through its own voltage.

| cell | grid distance to nearest thermocouple | SNR (40% cooling fault) |
|---:|---:|---:|
| 4 | 0 | **5.88σ** |
| 3 | 1 | 1.33σ |
| 9 | 3 | 1.21σ |
| 0 | 4 | **1.55σ** ← the corner beats the neighbour |

Identifiability is **not monotone in grid distance**. A hop-count heuristic gets this
exactly backwards. The Fisher information knows about thermal mass, conduction
anisotropy, boundary effects, excitation, and noise. A hop count knows about none of them.

### 3. Cell voltage is a thermometer — a terrible one

Why is a cooling fault visible *at all* on an uninstrumented cell? Because `R0` is
Arrhenius in temperature. A cooling fault warms the cell, lowering its resistance, moving
its voltage. Freeze that coupling (`ea_over_r_k = 0`) and only the far weaker entropic
`dOCV/dT` pathway survives:

| pathway | voltage-only `hA` bound (1σ) |
|---|---|
| `R0(T)` active | ±14.4% |
| `R0(T)` frozen (entropic only) | ±477% |

This is why the CRLB for `hA` comes back finite-but-enormous rather than infinite, and
why the *SNR threshold* — not a numerical rank cut — is what declares a cell unobservable.

### 4. Excitation is information, and it can substitute for a sensor

`SNR = magnitude / sqrt(CRLB)`, and the CRLB doesn't depend on the magnitude under a
local linearisation — so one simulation sweep gives every fault size at once.

Smallest fault visible at 5σ on cell 10 (no thermocouple):

| pulse amplitude | current σ | `R0` fault | cooling fault |
|---:|---:|---:|---:|
| 0.05C | 1.3 A | 1.66% | 3303% |
| 0.94C | 24.4 A | 0.72% | 165% |
| 2.50C | 65.0 A | 0.31% | **13.0%** |

Heat generation scales as `I²`. A **1.83C pulse makes cell 10's cooling fault observable
with no new sensor at all.** So "what should I measure next?" has two answers — instrument
it, or excite it harder — and both fall out of the same Fisher matrix:

```
Counterfactual sensor placement (no re-simulation, just a row mask):
  + thermocouple on cell 10  ->  SNR   5.71 sigma      <- the recommendation
  + thermocouple on cell  9  ->  SNR   1.26 sigma
  + thermocouple on cell 11  ->  SNR   1.26 sigma

With one extra thermocouple on cell 10:
  verdict      : DIAGNOSE          (was REFUSE_UNOBSERVABLE)
  detection SNR: 5.71 sigma        (was 1.24)
  CRLB (1s)    : +/- 7.01%         (was +/- 32.37%)   -- a 4.6x improvement
```

### 5. Constant current confounds resistance with capacity

A constant IR offset and a slowly-drifting OCV offset alias. Current *variation* breaks
the tie. Measured by the variance inflation factor, not the condition number:

| duty cycle | current σ | VIF(`R0`) |
|---|---:|---:|
| constant current | 0 A | 4.12 |
| pulse train | 8 A | 2.04 |
| pulse train | 39 A | 1.06 |

### 6. LFP is ~6× harder than NMC, and the code knows it

A capacity fault is only visible through `dOCV/dSOC`:

| SOC | NMC-like | LFP-like | ratio |
|---:|---:|---:|---:|
| 0.3 | 2.32 mV/% | 0.55 mV/% | 4.2× |
| 0.5 | 3.38 mV/% | 0.56 mV/% | 6.1× |
| 0.7 | 5.21 mV/% | 0.94 mV/% | 5.5× |

Same fault. Same sensors. Six times less signal.

---

## Install and run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,notebook]"    # Windows
# .venv/bin/python -m pip install -e ".[dev,notebook]"      # Linux / macOS
```

or, with [`uv`](https://docs.astral.sh/uv/): `uv venv && uv pip install -e ".[dev,notebook]"`

| Command | What it does |
|---|---|
| `python examples/01_first_demo.py` | the whole thesis in one script; writes `reports/figures/` |
| `python -m pytest` | 92 tests, ~35 s |
| `python -m pytest -m "not regression"` | skip the findings-pinning tests |
| `python -m ruff check src tests examples` | lint |
| `python -m mypy` | type-check `src/` |
| `python scripts/build_notebook.py` | regenerate the notebook |
| `jupyter lab notebooks/01_identifiability_study.ipynb` | the full study |

`make help` lists the same targets (`make check` = lint + typecheck + test).

---

## How it works

```
    duty cycle ──┐
    pack params ─┼──▶ simulate() ──▶ central differences ──▶ S[t, cell, {V,T}, param]
    fault inject ┘                                                    │
                                                                      │  ← every cell's
                                                                      │    V *and* T,
    sensor topology ──────────────────────────────────────────────────┤    instrumented
    noise model ──────────────────────────────────────────────────────┤    or not
                                                                      ▼
                                            row mask   ──▶  FIM = Sᵀ Σ⁻¹ S
                                                                      │
                            ┌─────────────────────────────────────────┤
                            ▼                                         ▼
                    CRLB = diag(FIM⁻¹)                  VIF = FIM_jj · [FIM⁻¹]_jj
                            │                                         │
                    SNR = magnitude/√CRLB                    isolation gate
                            │                                         │
                            └──────────────┬──────────────────────────┘
                                           ▼
                                    decide() ──▶ DIAGNOSE
                                              ├─▶ WEAK_EVIDENCE
                                              ├─▶ REFUSE_UNOBSERVABLE  + recommendation
                                              └─▶ REFUSE_CONFOUNDED    + recommendation
```

Because sensitivities are computed for **every** cell's voltage and temperature — not
just the instrumented ones — a sensor topology is only a *row mask* over the sensitivity
tensor. Counterfactual sensor placement therefore costs a matrix slice, not a
re-simulation. That is what makes `recommend_temp_sensor()` cheap.

### Two gates, in this order

1. **Isolation.** Is the parameter separable from the ones it could be confused with?
   Measured by the **variance inflation factor** `VIF_j = FIM_jj · [FIM⁻¹]_jj ≥ 1`, with
   the conventional `VIF > 10` threshold from regression diagnostics. *Not* the condition
   number: `cond(FIM)` is a property of the matrix, dominated by whichever direction is
   worst-informed, so a pack with invisible `hA` has a huge `cond(FIM)` even when its `R0`
   is perfectly isolated — gating on it would refuse every diagnosis.
2. **Detection.** Given separability, is the fault big enough to clear the CRLB floor?
   `≥5σ` observable, `2–5σ` weak, `<2σ` refuse.

Isolation is checked first because two parameters can be jointly well-determined (high
SNR on their sum) while being individually unidentifiable. Reporting "resistance fault,
8σ" when the data cannot separate resistance from capacity would be a confident lie.

### One implementation detail worth stating

`crlb()` does **not** use `np.linalg.pinv`. The pseudo-inverse of a rank-deficient FIM
returns the minimum-norm solution, which yields *finite* variances for parameters that are
completely unidentified — it would make the system claim it can see things it cannot.
Instead we eigendecompose, discard directions at the level of floating-point noise, and
return `inf` for any parameter with support on a discarded direction.

---

## Repository layout

```
src/astracell/
  cell/         ocv.py · ecm.py · thermal.py        first-order Thevenin + lumped thermal
  pack/         topology.py · params.py · simulate.py   4x8 grid, exact SOC/RC, Euler thermal
  duty/         profiles.py                          constant · pulse · random walk · rest+pulse
  sensors/      topology.py · noise.py               <- the observability bottleneck
  faults/       library.py · injector.py             physical faults vs sensor faults
  observability/
    sensitivity.py    central differences over relative perturbations
    fisher.py         FIM, CRLB (no pinv), VIF, condition number, D-optimal info gain
    mask.py           SNR -> {observable, weak, unobservable} -> GreyCellMap
    detectability.py  the (excitation x magnitude) heatmap
    decision.py       <- the refusal. Two gates, and a recommendation.
  viz/          packmap.py · heatmap.py

examples/01_first_demo.py                   the whole thesis, six acts
notebooks/01_identifiability_study.ipynb    generated by scripts/build_notebook.py
tests/        physics invariants (hypothesis) · faults · sensors · observability
LIMITATIONS.md                              written before the code. Read it.
```

## Physics implemented

```
V_i  = OCV(z_i, T_i) − I·R0_i(T_i) − v1_i        R0(T) = R0_ref · exp(Ea/R · (1/T − 1/T_ref))
ż_i  = −η·I / (Q_i · 3600)                        exact discrete update
v̇1_i = −v1_i/(R1·C1) + I/C1                       exact discrete update

C_th·Ṫ_i = I²R0_i + I·v1_i − I·T_i·∂OCV/∂T − Σ_j k_ij(T_i − T_j) − hA_i(T_i − T_cool)
           └── irreversible ──┘ └─ entropic ─┘  └── conduction ──┘  └── convection ──┘
```

The entropic term is kept. Most implementations drop it; it is 10–30% of heat generation
at low C-rate and its **sign flips between charge and discharge**, which is a
distinguishable telemetry signature. The identity `I·(OCV − V) ≡ I²R0 + I·v1` holds to
machine precision (1.8e-15) and is asserted in `tests/test_physics_invariants.py`.

Integration: SOC and the RC branch use their **exact** discrete solutions for
piecewise-constant current; temperature uses forward Euler at `dt = 1 s` against a ~220 s
thermal time constant. This lets the simulation step equal the sensor sampling period,
which matters — the Fisher information scales with the number of *independent samples*, so
an integrator forcing `dt << dt_sensor` would tempt you into counting information you
never measured.

## Tests

92 tests. They assert **theorems**, not observed numbers — a test that hard-codes "the SNR
is 149" is a test of the OCV curve, not of the code, and would break the moment the
stand-in curves are replaced with real ones, which is the plan.

- `dV/d(relative R0) == −I·R0` exactly at `t=0`, to machine precision
- charge conservation, energy balance, `V ≤ OCV` on discharge, monotone cooling at rest
- the conduction matrix is a Laplacian (symmetric, PSD, null space = constants)
- **adding a sensor never increases the CRLB** (Loewner monotonicity — a theorem)
- `crlb()` returns `inf` for perfectly collinear parameters (where `pinv` would lie)
- `VIF ≥ 1` always; `VIF == 1` exactly for an orthogonal design
- fault injection never mutates the ground truth
- one test marked `regression` pins a *finding* on one configuration, and says so

## Status

This is the **first technical scaffold**. It answers *"what is identifiable?"* and nothing
else. There is no detector, no estimator, no residual bank, no classifier, no uncertainty
calibration, no real data, no dashboard, and no LLM. Those come after — and only for the
faults this layer says are worth chasing.

The next honest step is to inject faults with a **higher-fidelity model** than the observer
assumes (PyBaMM DFN with degradation submodels), so that model mismatch becomes part of the
experiment rather than an unmeasured assumption. See [`LIMITATIONS.md`](LIMITATIONS.md) §10.

## License

Apache-2.0.
