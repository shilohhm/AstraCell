# AstraCell

**A battery diagnostic that knows what it can't see.**

AstraCell determines, from first principles, which battery faults are *identifiable* given
the battery model, the sensor topology, the measurement noise, and the excitation actually
present in the data. It refuses to diagnose the rest — and computes what additional
measurement would change its mind.

> ⚠️ **Read [`LIMITATIONS.md`](LIMITATIONS.md) before believing any number here.** The OCV
> curves are stand-ins, not fitted cells. **Nothing in this repository has ever detected a
> real fault, or touched a real battery.** Every bound below is a statement about *this
> model*, validated against itself and against an external simulator — never against a cell.

**Where to go from here:** the [technical report](docs/TECHNICAL_REPORT.md) is the full
argument; the [portfolio page](docs/PORTFOLIO.md) is the five-minute version;
[CLAIMS.md](docs/CLAIMS.md) maps every claim to the evidence that backs it;
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) regenerates every number from a fresh clone.

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

AstraCell instead computes the **Fisher information matrix** of the pack's parameters under
the actual sensor topology, and the **Cramér–Rao lower bound** it implies. The CRLB bounds
the variance of *every* unbiased estimator — not of one algorithm. If it says a 40% cooling
fault sits at 1.2σ, no detector you write will find it, and the honest thing to render is
grey.

There is no fault classifier here, deliberately. A classifier answers *"which fault?"*.
AstraCell answers the logically prior question: *"is that question answerable?"* Building the
classifier first produces a system that is confident exactly where it should be silent.

A second, sharper claim runs through everything: **more data can make a structurally wrong
answer more certain.** A model wrong in a way the data cannot expose grows *more* confident
as you feed it more samples — so AstraCell prices its own model bias and refuses when that
bias dwarfs the signal.

**The grey cells are not painted by a distance-to-sensor rule. They fall out of the
Cramér–Rao bound.**

## Three tiers of validation

The distinction this project is organised around. A result at one tier never licenses a
claim at a higher one.

| Tier | Meaning | Where AstraCell stands |
|---|---|---|
| **1 — internal self-consistency and synthetic experiments** | Within AstraCell's own models, plus theorems about the estimator | Extensive |
| **2 — independently developed external simulator** | Against PyBaMM, a simulator AstraCell did not implement | The phantom-fault refusal and the positive control |
| **3 — physical battery validation** | A measured cell, a real fault | **Contact, not validation.** v0.6 ran it: on a real cell the ECM is directionally wrong and AstraCell refuses ([REAL_CELL](docs/REAL_CELL.md)) — one cell, not validation |

## What the evidence shows

Produced by `python examples/01_first_demo.py` on a 4×8 pack (32 cells, 32 voltage
channels, 4 thermocouples, 1 current shunt), 1200 s of 1.0C pulse excitation, 1 mV / 0.5 K
noise. These numbers assume white noise and a known current — both are relaxed in the
[report](docs/TECHNICAL_REPORT.md), and one headline verdict flips when they are.

| Fault hypothesis | Thermocouple? | CRLB (1σ) | SNR | Verdict |
|---|---|---|---|---|
| `R0` +20% on cell 5 | no | ±0.13% | **149.2σ** | DIAGNOSE |
| capacity −5% on cell 17 | no | ±0.15% | **32.6σ** | DIAGNOSE |
| cooling −40% on cell 12 | **yes** | ±7.31% | **5.5σ** | DIAGNOSE |
| cooling −40% on cell 10 | no | ±32.4% | **1.2σ** | **REFUSE** |

Cooling faults are identifiable on **4 of 32 cells** — exactly the four carrying a
thermocouple. The fault on cell 10 is *really there*; AstraCell declines to diagnose it.

![pack map](reports/figures/packmap_cooling.png)

The rest of the argument, each finding linked to its example, test, and figure in
[CLAIMS.md](docs/CLAIMS.md) and derived in full in the [report](docs/TECHNICAL_REPORT.md):

- **Refusal comes with a fix.** A counterfactual thermocouple lifts cell 10's cooling SNR to
  5.71σ (4.6× tighter) — or a 1.83C pulse does it with no new hardware. Excitation
  substitutes for a sensor. *(`examples/03`)*
- **Correlated noise reallocates information**, it doesn't uniformly hurt: at `rho=0.99`,
  capacity gets 10× worse while `R0` and `hA` get *better*. This corrected a claim the repo
  once made the other way. *(`examples/02`,
  [WHAT_DID_NOT_WORK](docs/WHAT_DID_NOT_WORK.md) §2)*
- **The bound cannot see model bias.** A first-order ECM fitted over 20 minutes manufactures
  an 18.5% capacity loss from a 5% fault — and *more data makes it worse*: reported
  confidence climbs to 14 611σ while credible confidence sits fixed at 6.49σ. *(`examples/04`)*

  ![model mismatch](reports/figures/model_mismatch.png)

- **The verdicts are calibrated — and calibration makes them worse.** Under mismatch the
  variance-only interval covers the truth 0% of the time; the model-bias gate drops harmful
  overclaim on capacity from 100% to 0%, by refusing. *(`examples/05`)*
- **An independently developed external simulator produces a phantom fault.** Fitted to a
  healthy PyBaMM cell, the observer reports −67.6% ± 0.145% capacity loss (466σ from zero); a
  self-consistency control proves it is PyBaMM's physics, not our harness. The gate refuses.
  *(`examples/06`)*

  ![external phantom](reports/figures/external_estimate_distribution.png)

- **The refusal discriminates.** With a healthy baseline, the same machinery recovers a real
  injected fault at +20.0000% with nominal coverage, and still refuses a real degradation it
  cannot express — by a one-percent margin, reported not rounded away. *(`examples/07`)*

  ![positive control](reports/figures/positive_control_rates.png)

- **On a real cell, it refuses — and it should.** Run against a measured Oxford cell fading to
  −24.2%, the first-order ECM's capacity estimate comes back wrong in *sign* (a +10.5% "gain"),
  ~1150σ from the truth; AstraCell returns `REFUSE_MODEL_BIAS` on all 13 scored ages (coverage
  0/13). First contact with a battery, and the honest outcome. *(`examples/08`,
  [REAL_CELL](docs/REAL_CELL.md))*

## Install and run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,notebook]"    # Windows
# .venv/bin/python -m pip install -e ".[dev,notebook]"      # Linux / macOS
```

or, with [`uv`](https://docs.astral.sh/uv/): `uv venv && uv pip install -e ".[dev,notebook]"`.

| Command | What it does |
|---|---|
| `python examples/01_first_demo.py` | the whole thesis in one script; writes `reports/figures/` |
| `python examples/02_noise_robustness.py` | relaxes the white-noise and known-current assumptions |
| `python examples/03_next_best_test.py` | ranks what to measure next; refuses when nothing suffices |
| `python examples/04_model_mismatch.py` | prices the observer's own model error |
| `python examples/05_calibrated_abstention.py` | are the verdicts calibrated across repeated trials? |
| `pip install -e ".[pybamm]"` | add the optional external simulator (PyBaMM) for examples 06–07 |
| `python examples/06_external_plant_gate.py` | does abstention survive an external simulator? |
| `python examples/07_external_positive_control.py` | and when that cell is really broken, does it notice? |
| `pip install -e ".[oxford]"` · `python scripts/fetch_oxford.py` | add the Oxford real-cell dataset (ODC-ODbL, ~254 MB licensed download) |
| `python examples/08_real_cell.py` | Tier 3: score the observer against a real cell's measured fade (skips without the data) |
| `python -m pytest` | 236 tests, ~100 s (235 pass + 1 skip with PyBaMM; the real-cell test needs the Oxford dataset) |
| `python -m ruff check src tests examples scripts` · `python -m mypy` | lint · type-check |

`make check` runs lint + typecheck + test; `make help` lists every target. The full
pre-release sequence, the fresh-clone instructions, and the pinned-numbers guard are in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## How it works

```
    duty cycle ──┐
    pack params ─┼──▶ simulate() ──▶ central differences ──▶ S[t, cell, {V,T}, param]
    fault inject ┘                                                    │
    sensor topology ──────────────────────────────────────────────── │  ← row mask
    noise model ──────────────────────────────────────────────────── ▼
                                            FIM = Sᵀ Σ⁻¹ S ──▶ CRLB = diag(FIM⁻¹)
                                                       │              │
                                       VIF (isolation) ┤              ├ SNR = magnitude/√CRLB
                                                       ▼              ▼
                                          decide() ──▶ DIAGNOSE / WEAK_EVIDENCE
                                                    ├─▶ REFUSE_UNOBSERVABLE  + recommendation
                                                    ├─▶ REFUSE_CONFOUNDED    + recommendation
                                                    └─▶ REFUSE_MODEL_BIAS    (bias ≫ signal)
```

Because sensitivities are computed for **every** cell's voltage and temperature, a sensor
topology is only a *row mask* over the sensitivity tensor — so counterfactual sensor
placement costs a matrix slice, not a re-simulation. Two gates run in order: **isolation**
(is the parameter separable? `VIF > 10` ⇒ confounded) then **detection** (is the fault big
enough to clear the CRLB floor? `≥5σ` / `2–5σ` / `<2σ`). One detail worth stating: `crlb()`
does **not** use `pinv` — the pseudo-inverse returns finite variances for unidentified
parameters, which would make the system claim it can see things it cannot. Definitions of
every term are in [GLOSSARY.md](docs/GLOSSARY.md); the derivations in the
[report's methods appendix](docs/TECHNICAL_REPORT.md#methods-appendix).

## Repository layout

```
src/astracell/
  cell/ pack/ duty/ sensors/     first-order Thevenin + lumped thermal, 4x8 grid, duty cycles, AR(1) noise
  faults/                        physical faults vs sensor faults
  observability/                 sensitivity · fisher (FIM/CRLB/VIF) · mask · experiment · bias · estimator · decision
  plant/ calibration/            richer plants (incl. PyBaMM) · Monte Carlo coverage, external, positive control
  viz/                           pack maps and heatmaps
examples/01..07                  the thesis, then each honesty pass in turn
notebooks/01..04                 generated by scripts/build_notebook.py, committed executed
docs/                            TECHNICAL_REPORT · PORTFOLIO · CLAIMS · REPRODUCIBILITY · GLOSSARY ·
                                 FIGURES · WHAT_DID_NOT_WORK · CALIBRATION · EXTERNAL_PLANT ·
                                 POSITIVE_CONTROL · REAL_CELL
LIMITATIONS.md                   written before the code; now also a record of claims this repo got wrong
```

## Status

Built in five passes. **v0.0** answered *"what is identifiable?"* — the Fisher/CRLB
machinery, the grey-cell map, the refusal. **v0.1** added the **model-mismatch gate**: a
higher-fidelity plant, the structural bias `b = FIM⁻¹Sᵀ Σ⁻¹ r`, and `REFUSE_MODEL_BIAS`.
**v0.2** added **calibrated abstention**: an estimator, a Monte Carlo harness, and the
coverage/overclaim numbers showing the variance-only verdicts do not hold up under mismatch.
**v0.3** added the **external-simulator gate** (PyBaMM). **v0.4** added the **positive
control**, and found that v0.3's gate had refused *everything* — its statistic was pure
noise. The same machinery now recovers a real injected fault at its correct magnitude and
still refuses a real degradation it cannot express. Worse numbers, and the first ones that
mean anything.

There is still no residual bank, no classifier, no dashboard, and no LLM — and no *validated*
real-cell result. Those come after, and only for the faults this machinery says are worth chasing
and can be trusted. **v0.6 takes the measured-cell step** every prior version pointed at: run against
the Oxford Battery Degradation Dataset, the first-order ECM's capacity estimate on a real cell comes
back wrong in sign — a +10.5% "gain" against a measured −24.2% loss — and AstraCell refuses all 13
scored ages. Worse than any synthetic tier, and the honest outcome: the refusal is the result. It is
contact, not validation — one cell of eight. See [REAL_CELL.md](docs/REAL_CELL.md), the
[technical report](docs/TECHNICAL_REPORT.md) §7, and [LIMITATIONS.md](LIMITATIONS.md) §16.

## License

Apache-2.0.
