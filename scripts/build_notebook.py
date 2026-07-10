"""Generate the notebooks under ``notebooks/``.

The notebooks are build artifacts, not hand-maintained source files. Cells live here as
plain strings, which means they are diffable and greppable rather than buried in JSON.

    01_identifiability_study.ipynb      what could any estimator resolve? (v0.0-v0.1)
    02_calibrated_abstention.ipynb      are the verdicts calibrated across repeated trials? (v0.2)
    03_external_plant_gate.ipynb        does abstention survive a plant we did not write? (v0.3)
    04_external_positive_control.ipynb  and when the plant is really broken, does it notice? (v0.4)

This script writes *source-only* notebooks: no outputs, no execution counts, no cell ids. The
notebooks committed to the repository are the *executed* ones, several hundred kilobytes each,
because a notebook without its figures renders as a blank page on GitHub. Building and executing
are therefore two steps, and the second one is not optional:

    make notebook       # regenerate the source (this script) -- DISCARDS committed outputs
    make notebook-run   # execute in place, restoring the outputs

Running ``make notebook`` alone leaves every notebook stripped, which looks like a 900-line
deletion in ``git diff``. That is the build working as designed, not a corrupted file -- but do not
commit the result without running ``make notebook-run`` after it.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOK_01 = NOTEBOOKS / "01_identifiability_study.ipynb"
NOTEBOOK_02 = NOTEBOOKS / "02_calibrated_abstention.ipynb"
NOTEBOOK_03 = NOTEBOOKS / "03_external_plant_gate.ipynb"
NOTEBOOK_04 = NOTEBOOKS / "04_external_positive_control.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS_01 = [
    md("""
# AstraCell — Identifiability Study

**A battery diagnostic that knows what it can't see.**

This notebook answers one question, and deliberately not the next one:

> Given a battery model, a sensor topology, a noise model, and the excitation actually
> present in the data — *which faults could any estimator possibly resolve?*

Nothing here detects a fault. Detection comes later, and only for the faults this
notebook says are detectable. Building the detector first would produce a system that
is confident exactly where it should be silent.

The tool is the **Fisher information matrix** and its corollary the **Cramér–Rao lower
bound**, which bounds the variance of *every unbiased estimator* — not of one algorithm.
If the CRLB says a 40% cooling fault sits at 1.2σ, no amount of cleverness will find it.

> ⚠️ **Read [`LIMITATIONS.md`](../LIMITATIONS.md) first.** The OCV curves here are
> stand-ins, not fitted cells. Every number below is a statement about *this model*,
> not about a battery. The machinery is what is being demonstrated.
"""),
    code("""
import matplotlib.pyplot as plt
import numpy as np

from astracell.cell.ocv import LFP_LIKE, NMC_LIKE
from astracell.duty import constant_current, pulse_train
from astracell.observability import (
    ParamKind,
    all_specs,
    crlb,
    detectability_heatmap,
    fisher_information,
    grey_cell_map,
    local_specs,
    recommend_temp_sensor,
    sensitivities,
    variance_inflation,
)
from astracell.observability.decision import assess
from astracell.observability.sensitivity import ParameterSpec
from astracell.pack import PackTopology, nominal_pack, simulate
from astracell.sensors import NoiseModel
from astracell.sensors.topology import realistic_topology
from astracell.viz.heatmap import plot_detectability_heatmap, plot_min_detectable
from astracell.viz.packmap import plot_pack_map

plt.rcParams["figure.dpi"] = 110
SEED = 0
"""),
    md("""
---
## 1. Why chemistry decides the difficulty

A capacity fault shifts a cell's SOC relative to its neighbours. You see it in the
voltage only through `dOCV/dSOC`. On a graphite/NMC-like curve that slope is a few mV
per percent SOC. On LFP's plateau it is a fraction of a mV.

Same fault. Same sensors. Six times less signal.
"""),
    code("""
soc = np.linspace(0.05, 0.95, 400)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for curve, colour in ((NMC_LIKE, "#1565c0"), (LFP_LIKE, "#c62828")):
    axes[0].plot(soc, curve.ocv(soc), color=colour, label=curve.chemistry)
    axes[1].plot(soc, 10 * curve.docv_dsoc(soc), color=colour, label=curve.chemistry)

axes[0].set(xlabel="SOC", ylabel="OCV (V)", title="Open-circuit voltage")
axes[1].set(xlabel="SOC", ylabel="dOCV/dSOC (mV per % SOC)", title="Sensitivity to a capacity fault")
axes[1].set_yscale("log")
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.tight_layout()

for z in (0.3, 0.5, 0.7):
    ratio = NMC_LIKE.docv_dsoc(z) / LFP_LIKE.docv_dsoc(z)
    print(f"SOC {z:.1f}:  NMC {10 * NMC_LIKE.docv_dsoc(z):5.2f} mV/%   "
          f"LFP {10 * LFP_LIKE.docv_dsoc(z):5.2f} mV/%   ratio {ratio:.1f}x")
"""),
    md("""
---
## 2. The sensor topology, which is the actual constraint

A production BMS measures:

| Quantity | Coverage | Noise |
|---|---|---|
| Cell voltage | **one per cell** | ~1 mV (AFE) |
| Temperature | **4–12 for ~96 cells** | ~0.5–1 K |
| Current | **pack-level only** | ~0.5% FS |

Thirty-two voltage channels. Four temperature channels. One current channel. That
asymmetry, and not the algorithm, determines what is diagnosable.
"""),
    code("""
pack = PackTopology(n_modules=4, cells_per_module=8)
params = nominal_pack(pack, seed=SEED)
topology = realistic_topology(pack, n_temp_sensors=4)
noise = NoiseModel()

print(topology.summary())
print("thermocouples on cells:", topology.temp_cells)
print(f"voltage 1-sigma {1e3 * noise.voltage_sigma_v:.1f} mV, "
      f"temperature 1-sigma {noise.temp_sigma_k:.2f} K")

duty = pulse_train(1200.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0)
sim = simulate(params, duty.current_a, duty.dt_s)

fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
axes[0].plot(duty.time_s, duty.current_a, lw=0.8, color="#37474f")
axes[0].set_ylabel("current (A)")
axes[1].plot(sim.time_s, sim.voltage_v[:, ::8], lw=0.7)
axes[1].set_ylabel("cell V")
axes[2].plot(sim.time_s, sim.temp_k[:, topology.temp_index] - 273.15, lw=0.9)
axes[2].set(ylabel="sensed T (degC)", xlabel="time (s)")
for ax in axes:
    ax.grid(alpha=0.3)
axes[0].set_title(f"{duty.name}: mean {duty.mean_current_a:.0f} A, std {duty.current_std_a:.0f} A")
fig.tight_layout()
"""),
    md("""
---
## 3. Fisher information and the Cramér–Rao bound

For `y = h(θ) + e`, `e ~ N(0, Σ)`:

$$\\mathrm{FIM} = S^\\top \\Sigma^{-1} S, \\qquad S_{ij} = \\frac{\\partial y_i}{\\partial \\theta_j}, \\qquad \\mathrm{Var}(\\hat\\theta_j) \\ge [\\mathrm{FIM}^{-1}]_{jj}$$

We differentiate with respect to **relative** perturbations, so `sqrt(CRLB)` reads
directly as "the best 1σ uncertainty on this parameter, as a fraction".

A fault of size `m` is then detectable at `SNR = m / sqrt(CRLB)` σ, and that is an
upper bound on *every* detector's performance.
"""),
    code("""
cell = 10  # no thermocouple on this one
specs = local_specs(cell)
sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
fim = fisher_information(sens, topology, noise)
std = np.sqrt(crlb(fim))
vif = variance_inflation(fim)

print(f"cell {cell}   (thermocouple: {'yes' if cell in topology.temp_cells else 'no'})\\n")
print(f"{'parameter':10s} {'1-sigma floor':>15s} {'5-sigma fault':>15s} {'VIF':>8s}")
for spec, s, v in zip(specs, std, vif, strict=True):
    print(f"{spec.label():10s} {100 * s:14.3f}% {500 * s:14.2f}% {v:8.2f}")

print("\\nR0 and capacity are pinned to a fraction of a percent.")
print("hA -- the cooling coefficient -- is not pinned at all.")
"""),
    md("""
### Why is `hA` visible *at all* without a thermocouple?

Because `R0` is Arrhenius in temperature. A cooling fault warms the cell, which lowers
its resistance, which moves its voltage. **The cell's own voltage is a thermometer.**

It is a *terrible* thermometer — ~30% 1σ on `hA` — but it is not nothing, and that is
why the CRLB comes back finite rather than infinite. Freeze the Arrhenius coupling and
only the (much weaker) entropic `dOCV/dT` pathway remains.
"""),
    code("""
blind = topology.without_temp_sensors()
ha_specs = local_specs(cell, (ParamKind.HA,))

for label, p in (("R0(T) active   ", params), ("R0(T) frozen   ", params.evolve(ea_over_r_k=0.0))):
    s = sensitivities(p, duty.current_a, duty.dt_s, ha_specs)
    floor = np.sqrt(crlb(fisher_information(s, blind, noise))[0])
    print(f"{label} voltage-only hA 1-sigma: {100 * floor:10.1f}%")
print("\\nNo thermocouples anywhere. The voltage channel alone carries this information.")
"""),
    md("""
---
## 4. The grey-cell map

Per-cell identifiability, computed over the **full** parameter set (all cells × R0,
capacity, hA), so the bound marginalises over both within-cell confounding (R0 vs
capacity) and cross-cell confounding (a hot neighbour looks like a hot cell).

Grey means: *no estimator can resolve this fault here. AstraCell abstains.*
"""),
    code("""
maps = {}
for kind, magnitude in ((ParamKind.R0, 0.20), (ParamKind.CAPACITY, 0.05), (ParamKind.HA, 0.40)):
    maps[kind] = grey_cell_map(
        params, duty.current_a, duty.dt_s, topology, noise, kind=kind, magnitude=magnitude
    )
    print(f"{kind.value:12s} at {100 * magnitude:2.0f}%  ->  {maps[kind].counts()}")
"""),
    code("""
for kind in maps:
    plot_pack_map(maps[kind], pack, fault_cell=10 if kind is ParamKind.HA else None)
    plt.show()
"""),
    md("""
### The finding that justifies the whole approach

Cooling faults are identifiable on exactly the four cells carrying a thermocouple.

Now look at *how* identifiability decays away from a sensor. It doesn't decay. It falls
off a cliff at distance 0 and then goes **flat** — and what variation remains is
**not monotone in grid distance**.

The reason is worth sitting with. A thermocouple tells you about the cell it is *bolted
to*. At this noise level, conduction carries almost no `hA` information to it from a
neighbour: the cell adjacent to a thermocouple is no better determined than a cell four
hops away. Every uninstrumented cell's `hA` is instead read out through **its own
voltage**, via `R0(T)`. And what governs *that* is the cell's own total thermal
conductance — a corner cell, with fewer neighbours to conduct heat away, warms more for
the same `hA` change and is therefore *better* determined than an interior cell sitting
right next to a sensor.

So any hop-count heuristic ("grey out cells more than 2 positions from a sensor") gets
this exactly backwards. The Fisher information gets it right, because it knows about
thermal mass, conduction anisotropy, boundary effects, excitation, and noise. A hop
count knows about none of them.
"""),
    code("""
ha = maps[ParamKind.HA]
rows = []
for c in range(pack.n_cells):
    d = min(pack.grid_distance(c, s) for s in topology.temp_cells)
    rows.append((d, c, ha.snr[c]))

print(f"{'dist':>4s} {'cell':>5s} {'SNR':>8s}")
for d, c, s in sorted(rows)[:12]:
    print(f"{d:4d} {c:5d} {s:8.3f}")

by_distance = {}
for d, _, s in rows:
    by_distance.setdefault(d, []).append(s)

print("\\nmean SNR by grid distance to the nearest thermocouple:")
for d in sorted(by_distance):
    xs = by_distance[d]
    print(f"  distance {d}: mean {np.mean(xs):6.3f}   range [{min(xs):.3f}, {max(xs):.3f}]")

tc = topology.temp_cells[0]
neighbour = tc - 1
corner = 0
print(f"\\nthermocouple on cell {tc}          SNR {ha.snr[tc]:.3f}")
print(f"its immediate neighbour, cell {neighbour}  SNR {ha.snr[neighbour]:.3f}  (distance 1)")
print(f"the far corner, cell {corner}           SNR {ha.snr[corner]:.3f}  (distance "
      f"{min(pack.grid_distance(corner, s) for s in topology.temp_cells)})")
print("\\nThe corner beats the neighbour. Distance is not a sufficient statistic.")
"""),
    md("""
---
## 5. Detectability: excitation is information

`SNR = magnitude / sqrt(CRLB)`, and under a local linearisation `CRLB` does not depend
on the magnitude. So one simulation sweep over the *excitation* axis yields every fault
magnitude at once, and the heatmap is a statement about the Fisher information rather
than about a particular fault size.

For a **resistance** fault: the IR drop scales as `I`, so information scales as `I²`.

For a **cooling** fault: heat generation scales as `I²`, so the temperature deviation
does too — and information scales faster still.
"""),
    code("""
heat_r0 = detectability_heatmap(params, topology, noise, cell=10, kind=ParamKind.R0)
heat_ha = detectability_heatmap(params, topology, noise, cell=10, kind=ParamKind.HA)

for result in (heat_r0, heat_ha):
    plot_detectability_heatmap(result)
    plt.show()
    plot_min_detectable(result)
    plt.show()
"""),
    code("""
print("Smallest fault visible at 5 sigma on cell 10, versus excitation:\\n")
print(f"{'C-rate':>8s} {'I_std (A)':>10s} {'R0 fault':>12s} {'cooling fault':>15s}")
for k, c_rate in enumerate(heat_r0.excitation_c_rate):
    print(f"{c_rate:8.2f} {heat_r0.current_std_a[k]:10.1f} "
          f"{100 * heat_r0.min_detectable_magnitude()[k]:11.2f}% "
          f"{100 * heat_ha.min_detectable_magnitude()[k]:14.1f}%")
"""),
    md("""
### The condition number is not the confounding metric

`cond(FIM)` is a property of the *matrix*, dominated by whichever direction is
worst-informed. A pack whose `hA` is nearly invisible has a huge `cond(FIM)` even when
its `R0` is perfectly isolated. Gating on it would refuse every diagnosis.

The right per-parameter scalar is the **variance inflation factor**,
`VIF_j = FIM_jj · [FIM⁻¹]_jj ≥ 1`: how much confounding with the *other* parameters has
inflated this parameter's variance. `VIF > 10` is the conventional
regression-diagnostics threshold for serious multicollinearity. We inherit it.
"""),
    code("""
specs2 = local_specs(10, (ParamKind.R0, ParamKind.CAPACITY))
print(f"{'duty cycle':>18s} {'VIF(R0)':>10s} {'VIF(Q)':>10s}")
for profile in (constant_current(600.0, 1.0, 0.2),
                pulse_train(600.0, 1.0, mean_c_rate=0.2, pulse_c_rate=0.3),
                pulse_train(600.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.5)):
    s = sensitivities(params, profile.current_a, profile.dt_s, specs2)
    v = variance_inflation(fisher_information(s, topology, noise))
    tag = f"{profile.name} ({profile.current_std_a:.0f}A)"
    print(f"{tag:>18s} {v[0]:10.3f} {v[1]:10.3f}")

print("\\nConstant current confounds R0 with capacity: a constant IR offset and a")
print("slowly-drifting OCV offset alias. Current variation breaks the tie.")
"""),
    md("""
---
## 6. The refusal, and what would fix it

A 40% cooling fault is really present on cell 10. AstraCell declines to diagnose it —
and then tells you the two things that would change its mind.
"""),
    code("""
full_specs = all_specs(pack.n_cells)
full_sens = sensitivities(params, duty.current_a, duty.dt_s, full_specs)
target = ParameterSpec(10, ParamKind.HA)

before = assess(full_sens, topology, noise, full_specs, target, 0.40)
print(before.render())
"""),
    md("""
### Option A — instrument it

`sensitivities()` differentiates *every* cell's temperature, whether or not that cell is
instrumented. A sensor topology is therefore only a row mask over the sensitivity
tensor, and counterfactual sensor placement costs a matrix slice, not a re-simulation.
"""),
    code("""
ranked = recommend_temp_sensor(full_sens, topology, noise, full_specs, target, 0.40)
print("candidate thermocouple placements, best first:")
for c, snr in ranked[:5]:
    print(f"  cell {c:2d}  ->  SNR {snr:6.2f} sigma")

best_cell = ranked[0][0]
after = assess(full_sens, topology.with_temp_sensor_at(best_cell), noise, full_specs, target, 0.40)
print()
print(after.render())
print(f"\\nCRLB floor improved {before.crlb_std / after.crlb_std:.1f}x.")
print(f"Verdict: {before.kind.value.upper()} -> {after.kind.value.upper()}")
"""),
    md("""
### Option B — excite it harder

Heat generation scales as `I²`. A harder current pulse makes the cell's own voltage work
as a thermometer, via `R0(T)`. No new hardware; a different test.

Both options come out of the same Fisher information. The code does not have to choose —
it reports the trade.

*(This uses the per-cell heatmap, which ignores cross-cell confounding and is therefore
optimistic. See `LIMITATIONS.md` §5.)*
"""),
    code("""
snr_by_excitation = 0.40 / heat_ha.crlb_std
clears = np.flatnonzero(snr_by_excitation >= 5.0)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(heat_ha.excitation_c_rate, snr_by_excitation, marker="o", color="#1565c0")
ax.axhline(5.0, ls="--", color="#2e7d32", label="5 sigma (observable)")
ax.axhline(2.0, ls="--", color="#f9a825", label="2 sigma (abstain below)")
ax.axhline(before.snr, ls=":", color="#616161", label=f"demo duty cycle ({before.snr:.2f} sigma)")
ax.set(xlabel="pulse amplitude (C-rate)", ylabel="SNR for a 40% cooling fault (sigma)",
       yscale="log", title="Cell 10 has no thermocouple. Excitation can substitute for one.")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()

if clears.size:
    print(f"A {heat_ha.excitation_c_rate[clears[0]]:.2f}C pulse makes cell 10's cooling fault")
    print("observable with no additional sensor.")
"""),
    md("""
---
## What this notebook did *not* do

- It did not detect a fault. There is no detector.
- It did not validate anything against a real battery, or a real dataset.
- It assumed unbiased estimators. The CRLB does not bound biased ones.
- It assumed **white noise** (`rho = 0`) and a **perfectly known pack current**.
- It assumed **the model is correct**, which is the expensive one. The CRLB bounds
  *parameter* uncertainty under a known model structure and says nothing about model
  mismatch. `examples/04_model_mismatch.py` prices it: fit this first-order ECM to a plant
  with a slow diffusion branch and the estimator manufactures an apparent **18.5% capacity
  loss** — nearly four times the 5% fault it was asked to find. The 32.6σ capacity diagnosis
  above is worth **0.27σ**. Worse, that error is a *bias*: replicate this experiment 10 000
  times and the reported SNR climbs to 14 611σ while the credible SNR does not move at all.

An earlier version of this notebook claimed that each of these "makes the reported
identifiability *better* than reality, never worse — the safe direction for a system
whose purpose is abstention." **That was an estimate, and it is false.**
`examples/02_noise_robustness.py` measures it:

- Under AR(1) noise with `rho = 0.99`, the resistance and cooling bounds are **2.6×
  tighter** than under white noise, because whitening is a first difference and their
  signatures ride the current pulses. Capacity, whose signature is a near-DC SOC ramp,
  degrades 10×. Correlated noise does not uniformly hurt — it *reallocates*.
- At `rho = 0.9`, one of the four headline verdicts flips: cooling on an *instrumented*
  cell falls from `DIAGNOSE` (5.46σ) to `REFUSE` (1.93σ). A thermocouple's share of the
  cooling information collapses from 90.4% to 0.8%, and at `rho = 0.99` the
  instrumented-beats-uninstrumented ordering **inverts**.

So `rho = 0` is the *optimistic* default, not a conservative one, and the direction of
the error is not knowable without measuring it. A green cell is a hypothesis, not a
promise — and so, it turns out, is a grey one.

There are two ways to be wrong about a battery. This notebook computes one of them.
Only that one yields to more data.

Full accounting: [`LIMITATIONS.md`](../LIMITATIONS.md).
"""),
]


CELLS_02 = [
    md("""
# AstraCell — Calibrated Abstention

**Are the verdicts calibrated, or just convincing?**

v0.1 showed AstraCell can compute a structural bias and refuse when it dominates — on *one*
example. This notebook asks the harder question across thousands of repeated experiments with
a **known injected truth**:

> When AstraCell says DIAGNOSE, WEAK, or REFUSE, and when it draws a confidence interval, do
> those claims hold up as *frequencies*? Does a 90% interval cover the truth 90% of the time?

To ask that at all, v0.2 adds something the repository never had: **an estimator**. Everything
before was a property of the design. Here a Gauss-Newton (matched, to test attainment) or an
exact linear-Gaussian fit (mismatched, for speed) turns each noisy realisation into an
estimate and a verdict, and `astracell.calibration` counts how often they hold up.

> ⚠️ This proves **self-consistency**, not external truth. There are no real batteries here.
> It shows AstraCell is honest about its own noise and its own model error, on its own terms.
> The line between what that validates and what it cannot is drawn in
> [`docs/CALIBRATION.md`](../docs/CALIBRATION.md).
"""),
    code("""
import matplotlib.pyplot as plt
import numpy as np

from astracell.calibration import (
    NOMINAL_LEVELS,
    abstention_metrics,
    build_scenario,
    coverage_curve,
    run_trials,
    sample_count_curve,
    verdict_distribution,
)
from astracell.duty import pulse_train
from astracell.observability.decision import VerdictKind
from astracell.observability.sensitivity import ParamKind
from astracell.plant import REALISTIC_MISMATCH

plt.rcParams["figure.dpi"] = 110
SEED = 0
BLIND_CELL = 1          # voltage only, on the 2x2 demo pack
WINDOW = pulse_train(600.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0).current_a
"""),
    md("""
---
## 1. Coverage vs nominal: does a 90% interval cover 90% of the time?

A capacity fault, estimated two ways. Under a **matched** model the maximum-likelihood
interval should cover at its nominal rate — which would show the Cramér–Rao bound is
*attained*, not merely asserted. Under **mismatch** the variance-only interval is centred on
the pseudo-true value, an entire structural bias away from the truth, so it covers essentially
never. Widening it to admit the bias restores coverage only by becoming uselessly wide.
"""),
    code("""
matched = build_scenario(name="cap/matched", fault_kind=ParamKind.CAPACITY,
                         target_cell=BLIND_CELL, current_a=WINDOW)
mismatch = build_scenario(name="cap/mismatch", fault_kind=ParamKind.CAPACITY,
                          target_cell=BLIND_CELL, current_a=WINDOW, mismatch=REALISTIC_MISMATCH)

res_matched = run_trials(matched, 150, seed=SEED, estimator="gauss_newton",
                         estimator_options={"max_iter": 10, "step_tol": 1e-5})
res_mismatch = run_trials(mismatch, 250, seed=SEED, estimator="linear")

cov_matched = coverage_curve(res_matched)
cov_var = coverage_curve(res_mismatch, bias_aware=False)
cov_bias = coverage_curve(res_mismatch, bias_aware=True)

print(f"matched sigma = {res_matched.crlb_std:.3%}   mismatch bias = {res_mismatch.bias:+.2%}\\n")
print(f"{'nominal':>8s} {'matched (MLE)':>15s} {'mismatch var':>14s} {'mismatch bias-aware':>21s}")
for lvl, cm, cv, cb in zip(NOMINAL_LEVELS, cov_matched, cov_var, cov_bias, strict=True):
    print(f"{lvl:8.0%} {cm:15.2%} {cv:14.2%} {cb:21.2%}")
"""),
    code("""
fig, ax = plt.subplots(figsize=(6.0, 5.2))
grid = np.array(NOMINAL_LEVELS)
ax.plot([0, 1], [0, 1], ls="--", color="#455a64", lw=1.0, label="perfect calibration")
ax.plot(grid, cov_matched, "o-", color="#2e7d32", label="matched (MLE)")
ax.plot(grid, cov_var, "s-", color="#c62828", label="mismatched, variance-only")
ax.plot(grid, cov_bias, "^-", color="#1565c0", label="mismatched, bias-aware")
ax.set(xlabel="nominal confidence", ylabel="empirical coverage",
       xlim=(0.45, 1.0), ylim=(-0.03, 1.03),
       title="A 90% interval should cover 90% of the time.")
ax.legend(fontsize=8, loc="center left")
ax.grid(alpha=0.3)
plt.show()
"""),
    md("""
The matched curve hugs the diagonal: the estimator reaches the bound. The variance-only curve
under mismatch sits flat on the floor — confident and wrong at every level. This is the first
time the repository has shown the CRLB is attainable, and the first time it has *measured* its
own overconfidence rather than argued about it.
"""),
    md("""
---
## 2. More data buys precision, not accuracy

The central claim of the whole project, now empirical. Repeat the mismatched experiment `k`
times. The variance-only SNR grows as `sqrt(k)`, without bound. The bias-aware SNR saturates
at a ceiling `|m|/|b|` and stops dead — because the estimate cloud tightens onto the
pseudo-true value, which was never the truth.
"""),
    code("""
counts = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 1000.0, 10000.0])
curve = sample_count_curve(res_mismatch, counts)

print(f"true fault {res_mismatch.delta_true:+.0%}   bias {res_mismatch.bias:+.2%}   "
      f"estimate settles at {curve.center:+.2%}   ceiling {curve.ceiling:.3f} sigma\\n")
print(f"{'repeats':>9s} {'SNR (variance)':>16s} {'SNR (bias-aware)':>18s}")
for k, sv, sb in zip(counts, curve.snr_var, curve.snr_bias, strict=True):
    print(f"{int(k):9d} {sv:16.1f} {sb:18.3f}")

fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4))
left.loglog(counts, curve.snr_var, "o-", color="#c62828", label="variance only")
left.loglog(counts, curve.snr_bias, "s-", color="#1565c0", label="bias-aware")
left.axhline(5.0, ls=":", color="#2e7d32", label="5 sigma")
left.set(xlabel="independent repetitions", ylabel="detection SNR",
         title="Confidence climbs; credibility saturates.")
left.legend(fontsize=8); left.grid(alpha=0.3, which="both")

right.fill_between(counts, 100 * curve.band_lo, 100 * curve.band_hi,
                   color="#c62828", alpha=0.25, label="95% of estimates")
right.axhline(100 * curve.center, color="#c62828", lw=1.2, label="estimate centre")
right.axhline(100 * res_mismatch.delta_true, ls="--", color="#2e7d32", lw=1.3, label="the truth")
right.set(xscale="log", xlabel="independent repetitions", ylabel="capacity estimate (%)",
          title="The cloud tightens onto the wrong answer.")
right.legend(fontsize=8); right.grid(alpha=0.3, which="both")
plt.show()
"""),
    md("""
---
## 3. Where AstraCell diagnoses, weakens, and refuses

Sweep the true fault magnitude and watch the verdict distribution move. Under mismatch the
capacity diagnosis is *capped*: no matter how large the true fault, the structural bias keeps
the credible SNR below threshold, so `REFUSE_MODEL_BIAS` dominates and the diagnosis rate never
rises. Resistance, whose structural bias is small, diagnoses freely once the fault clears the
noise. Same gate, opposite outcomes — because one parameter's model error is fatal and the
other's is not.
"""),
    code("""
STYLE = {
    VerdictKind.DIAGNOSE: ("#2e7d32", "diagnose"),
    VerdictKind.WEAK_EVIDENCE: ("#f9a825", "weak"),
    VerdictKind.REFUSE_UNOBSERVABLE: ("#9e9e9e", "refuse: unobservable"),
    VerdictKind.REFUSE_CONFOUNDED: ("#6a1b9a", "refuse: confounded"),
    VerdictKind.REFUSE_MODEL_BIAS: ("#c62828", "refuse: model bias"),
}
sweeps = [
    ("R0", ParamKind.R0, np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])),
    ("capacity", ParamKind.CAPACITY, np.array([0.0, -0.01, -0.02, -0.05, -0.1, -0.2, -0.3])),
]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
for ax, (label, kind, mags) in zip(axes, sweeps, strict=True):
    fractions = {v: np.zeros(len(mags)) for v in VerdictKind}
    for i, mag in enumerate(mags):
        sc = build_scenario(name="s", fault_kind=kind, target_cell=BLIND_CELL,
                            fault_magnitude=float(mag), current_a=WINDOW, mismatch=REALISTIC_MISMATCH)
        for v, f in verdict_distribution(run_trials(sc, 120, seed=SEED)).items():
            fractions[v][i] = f
    x = 100 * np.abs(mags)
    bottom = np.zeros(len(mags))
    for v in VerdictKind:
        colour, name = STYLE[v]
        ax.fill_between(x, bottom, bottom + fractions[v], color=colour, alpha=0.85,
                        label=name, step="mid")
        bottom = bottom + fractions[v]
    ax.set(xlabel=f"|{label} fault| (%)", ylim=(0, 1), title=f"{label} under mismatch")
    ax.margins(x=0)
axes[0].set_ylabel("fraction of trials")
axes[1].legend(fontsize=7, loc="center left", framealpha=0.9)
plt.show()
"""),
    md("""
---
## 4. The number that says calibration worked

A **harmful overclaim** is a DIAGNOSE whose confident interval misses the truth. Under mismatch
the variance-only observer commits them constantly — it diagnoses a capacity fault that is
almost entirely its own model error. Turning on the bias gate converts exactly those into
refusals, driving the harmful-overclaim rate toward zero. That is the whole point of the gate,
and it costs diagnoses AstraCell should never have made.
"""),
    code("""
print(f"{'fault':>9s} {'gate OFF overclaim':>19s} {'gate ON overclaim':>18s}")
labels = [("R0", ParamKind.R0), ("capacity", ParamKind.CAPACITY)]
before, after = [], []
for label, kind in labels:
    off = abstention_metrics(run_trials(build_scenario(
        name="off", fault_kind=kind, target_cell=BLIND_CELL, current_a=WINDOW,
        mismatch=REALISTIC_MISMATCH, use_bias_gate=False), 250, seed=SEED))
    on = abstention_metrics(run_trials(build_scenario(
        name="on", fault_kind=kind, target_cell=BLIND_CELL, current_a=WINDOW,
        mismatch=REALISTIC_MISMATCH, use_bias_gate=True), 250, seed=SEED))
    before.append(off.harmful_overclaim_rate)
    after.append(on.harmful_overclaim_rate)
    print(f"{label:>9s} {off.harmful_overclaim_rate:19.2%} {on.harmful_overclaim_rate:18.2%}")

fig, ax = plt.subplots(figsize=(5.6, 4.0))
x = np.arange(len(labels))
ax.bar(x - 0.2, before, 0.4, color="#c62828", label="variance-only (gate off)")
ax.bar(x + 0.2, after, 0.4, color="#1565c0", label="bias-aware (gate on)")
ax.set(xticks=x, xticklabels=[label for label, _ in labels], ylabel="harmful overclaim rate",
       ylim=(0, 1.05), title="The model-bias gate turns overclaims into refusals.")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
plt.show()
"""),
    md("""
---
## What this notebook validated, and what it did not

**Validated (synthetically, on this model):**

- The maximum-likelihood estimator *attains* the Cramér–Rao bound under a matched model —
  coverage tracks nominal. The bound is not just a bound; it is reached.
- Under model mismatch the variance-only interval is overconfident to the point of never
  covering, and the model-bias gate converts the resulting overclaims into refusals.
- More data shrinks the variance and leaves the bias untouched, exactly. Confidence and
  credibility come apart, and only confidence responds to sample size.

**Not validated, and not claimable:**

- Anything about a **real battery**. The plant is a hand-built intermediate model, so this is a
  *lower bound* on how wrong the observer is, not a measurement of it.
- Calibration under a mismatch we did not write. The bias gate is only as good as the plant we
  guessed; a different structural error would need its own accounting.
- That the thresholds (2σ, 5σ) or the AR(1) noise or the known-current assumption are right.
  Those remain conventions and idealisations, as `LIMITATIONS.md` records.

Calibration made several of AstraCell's numbers *worse* — the capacity diagnosis it once made
confidently is now refused, and correctly. That is the result. A system that abstains when it
should is not a weaker diagnostic; it is the only kind worth trusting.

Full accounting: [`docs/CALIBRATION.md`](../docs/CALIBRATION.md) and
[`LIMITATIONS.md`](../LIMITATIONS.md).
"""),
]


CELLS_03 = [
    md("""
# AstraCell — The External-Plant Gate

**Does calibrated abstention survive a plant AstraCell did not write?**

Every version before this measured AstraCell against a battery *we* built. v0.1's mismatch was
four terms we chose, at sizes we set, so "the observer is simpler than the plant" was true by
construction. v0.3 is the first external-validity test: the data comes from a **PyBaMM** SPMe
single cell — electrolyte and particle diffusion a first-order ECM cannot express, a gap we did
not design. Same estimator, same gates; PyBaMM only supplies the voltage.

> ⚠️ **Still synthetic.** PyBaMM is a sophisticated *model*, not a measured cell. This tests
> external *model* mismatch, not physical truth. The four disclaimers are in
> [`docs/EXTERNAL_PLANT.md`](../docs/EXTERNAL_PLANT.md) §5. PyBaMM is an optional dependency
> (`pip install -e '.[pybamm]'`); the PyBaMM cells below skip cleanly without it, and the
> self-consistency control in §1 runs either way.
"""),
    code("""
import matplotlib.pyplot as plt
import numpy as np

from astracell.calibration import (
    NOMINAL_LEVELS,
    abstention_metrics,
    build_external_observer,
    coverage_curve,
    external_scenario,
    observer_voltage,
    prepare_external,
    pulse_profile,
    run_trials,
    two_sided_z,
)
from astracell.cell.ocv import NMC_LIKE, ocv_from_table
from astracell.plant import PYBAMM_AVAILABLE

plt.rcParams["figure.dpi"] = 110
SEED = 0
SOC0 = 0.9
N_TIME = 600
CAPACITY_AH = 5.0
CURRENT = pulse_profile(CAPACITY_AH, mean_c_rate=0.5, pulse_c_rate=1.0, n_time=N_TIME)

# PyBaMM-derived objects; populated in section 2 when the dependency is present.
observer = None
money_result = None

print("PyBaMM available:", PYBAMM_AVAILABLE)
"""),
    md("""
---
## 1. A control that cannot be a harness bug

Before PyBaMM is involved, feed the *same* external-plant pipeline an **ECM-generated** trace.
There the model mismatch is exactly zero, so coverage must track nominal. If it does, any
collapse under PyBaMM later is the plant's mismatch and not a bug in the plumbing. This runs
with no PyBaMM — the observer here uses the built-in `NMC_LIKE` curve.
"""),
    code("""
control_observer = build_external_observer(NMC_LIKE, CAPACITY_AH)
control_scn = external_scenario(name="control", observer=control_observer, current_a=CURRENT, soc0=SOC0)
control_prepared = prepare_external(control_scn, observer_voltage(control_scn))
control_result = run_trials(control_scn, 3000, seed=SEED, estimator="linear", prepared=control_prepared)

print(f"structural bias = {control_result.bias:+.4%}   (an ECM cannot mismatch itself)")
print(f"{'nominal':>10s} {'empirical coverage':>20s}")
for level, got in zip(NOMINAL_LEVELS, coverage_curve(control_result), strict=True):
    print(f"{level:10.0%} {got:20.1%}")
worst = max(abs(g - l) for g, l in zip(coverage_curve(control_result), NOMINAL_LEVELS, strict=True))
print(f"\\nlargest deviation from nominal: {worst:.3f}  ->  the harness adds no bias of its own.")
"""),
    md("""
---
## 2. The residual, and a phantom fault on a healthy cell

Now the real plant. The observer borrows PyBaMM's own pseudo-OCV (a slow C/20 discharge), so the
static voltage–SOC relationship is shared and the residual under load is purely the **dynamics**
the ECM omits. The cell is **healthy** — the true capacity deviation is zero — yet fitting the
ECM to PyBaMM's voltage manufactures a large, precise capacity fault out of the diffusion droop.
"""),
    code("""
if PYBAMM_AVAILABLE:
    from astracell.plant import pybamm_pseudo_ocv, simulate_pybamm_cell

    soc, ocv = pybamm_pseudo_ocv(model="SPMe", n_points=200)
    curve = ocv_from_table(soc, ocv, name="pybamm_spme", chemistry="SPMe/Chen2020")
    observer = build_external_observer(curve, CAPACITY_AH)

    plant = simulate_pybamm_cell(CURRENT, 1.0, model="SPMe", soc0=SOC0)
    money_scn = external_scenario(name="pybamm", observer=observer, current_a=CURRENT, soc0=SOC0)
    obs_v = observer_voltage(money_scn)
    money_prepared = prepare_external(money_scn, plant.voltage_v)
    money_result = run_trials(money_scn, 500, seed=SEED, estimator="linear", prepared=money_prepared)

    residual_mv = 1e3 * (plant.voltage_v - obs_v)
    rms = float(np.sqrt(np.mean(residual_mv**2)))
    print(f"plant - observer residual: {rms:.2f} mV RMS  (measurement noise is 1 mV)")
    print(f"true capacity deviation:      0.00%  (healthy cell)")
    print(f"estimated capacity deviation: {money_result.delta_hat.mean():+.2%} "
          f"+/- {money_result.crlb_std:.3%} (1 sigma) -- "
          f"{abs(money_result.delta_hat.mean()) / money_result.crlb_std:.0f} sigma from the truth")

    t = np.arange(N_TIME)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, height_ratios=[2, 1])
    top.plot(t, plant.voltage_v, color="#1565c0", lw=1.0, label="PyBaMM SPMe (plant)")
    top.plot(t, obs_v, color="#c62828", lw=1.0, ls="--", label="first-order ECM (observer)")
    top.set(ylabel="terminal voltage (V)", title="A gap small in volts, decisive in capacity")
    top.legend(fontsize=8); top.grid(alpha=0.3)
    bottom.plot(t, residual_mv, color="#455a64", lw=0.8); bottom.axhline(0, color="#c62828", ls=":")
    bottom.set(xlabel="time (s)", ylabel="residual (mV)"); bottom.grid(alpha=0.3)
    plt.show()
else:
    print("PyBaMM not installed -- skipping. Install with: pip install -e '.[pybamm]'")
"""),
    md("""
---
## 3. Coverage before and after admitting the bias

The variance-only interval — pinned to ~0.15% around a centre 60-odd percent from zero — covers
the healthy truth essentially never. The control (ECM plant) sits on the diagonal, proving the
collapse is the plant. Widening the interval for the structural bias restores coverage above
80%, but only by becoming wide enough to be diagnostically useless.
"""),
    code("""
if PYBAMM_AVAILABLE and money_result is not None:
    cov_control = coverage_curve(control_result)
    cov_var = coverage_curve(money_result, bias_aware=False)
    cov_bias = coverage_curve(money_result, bias_aware=True)

    print(f"{'nominal':>8s} {'control (ECM)':>15s} {'PyBaMM var':>12s} {'PyBaMM bias-aware':>18s}")
    for lvl, cc, cv, cb in zip(NOMINAL_LEVELS, cov_control, cov_var, cov_bias, strict=True):
        print(f"{lvl:8.0%} {cc:15.1%} {cv:12.1%} {cb:18.1%}")

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    grid = np.array(NOMINAL_LEVELS)
    ax.plot([0, 1], [0, 1], ls="--", color="#455a64", lw=1.0, label="perfect calibration")
    ax.plot(grid, cov_control, "o-", color="#2e7d32", label="control: ECM plant (matched)")
    ax.plot(grid, cov_var, "s-", color="#c62828", label="PyBaMM, variance-only")
    ax.plot(grid, cov_bias, "^-", color="#1565c0", label="PyBaMM, bias-aware")
    ax.set(xlabel="nominal confidence", ylabel="empirical coverage", xlim=(0.45, 1.0),
           ylim=(-0.03, 1.03), title="A plant we did not write breaks variance-only coverage.")
    ax.legend(fontsize=8, loc="center left"); ax.grid(alpha=0.3)
    plt.show()
else:
    print("PyBaMM not installed -- skipping the coverage comparison.")
"""),
    md("""
---
## 4. Harmful overclaim across excitation, with and without the gate

Without the gate the observer diagnoses the phantom fault in **every trial at every C-rate**. The
bias gate — reading its own fit residual to estimate its bias — refuses instead, driving harmful
overclaim to zero. Note the phantom fault's *size* swings with C-rate and even flips sign: its
instability is the proof that it is model mismatch, not a real capacity loss. What is robust is
the overclaim and the refusal.
"""),
    code("""
if PYBAMM_AVAILABLE:
    from astracell.plant import simulate_pybamm_cell

    c_rates = (0.2, 0.3, 0.5, 0.8)
    gated_rates, ungated_rates = [], []
    print(f"{'mean C':>7s} {'phantom fault':>14s} {'sigma':>8s} {'no gate':>9s} {'gated':>7s}")
    for mean_c in c_rates:
        cur = pulse_profile(CAPACITY_AH, mean_c_rate=mean_c, pulse_c_rate=1.0, n_time=N_TIME)
        pl = simulate_pybamm_cell(cur, 1.0, model="SPMe", soc0=SOC0)
        g_scn = external_scenario(name="g", observer=observer, current_a=cur, soc0=SOC0)
        g_prep = prepare_external(g_scn, pl.voltage_v)
        u_scn = external_scenario(name="u", observer=observer, current_a=cur, soc0=SOC0,
                                  use_bias_gate=False)
        u_prep = prepare_external(u_scn, pl.voltage_v)
        g = abstention_metrics(run_trials(g_scn, 300, seed=SEED, prepared=g_prep))
        u = abstention_metrics(run_trials(u_scn, 300, seed=SEED, prepared=u_prep))
        gated_rates.append(g.harmful_overclaim_rate)
        ungated_rates.append(u.harmful_overclaim_rate)
        print(f"{mean_c:6.1f}C {g_prep.target_bias:+13.1%} {g_prep.target_sigma:8.3%} "
              f"{u.harmful_overclaim_rate:9.2f} {g.harmful_overclaim_rate:7.2f}")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(c_rates))
    ax.bar(x - 0.2, ungated_rates, 0.4, color="#c62828", label="no gate (variance only)")
    ax.bar(x + 0.2, gated_rates, 0.4, color="#2e7d32", label="bias gate on")
    ax.set(xticks=x, xticklabels=[f"{c:.1f}C" for c in c_rates], ylabel="harmful overclaim rate",
           ylim=(0, 1.05), title="A healthy cell, diagnosed as faulty -- until the gate refuses.")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.show()
else:
    print("PyBaMM not installed -- skipping the excitation sweep.")
"""),
    md("""
---
## What this notebook validated, and what it did not

**Validated (synthetically, against a plant we did not design):**

- The external-plant harness is unbiased: fed an ECM trace, it covers at nominal (§1).
- Fed a **PyBaMM** cell, the same first-order ECM reports a large, precise capacity fault on a
  *healthy* cell, and the variance-only interval covers the truth essentially never (§2–3).
- The bias gate converts those overclaims into refusals at every excitation — harmful overclaim
  100% → 0% (§4). Under an honest plant, AstraCell diagnoses *less*.

**Not validated, and not claimable:**

- Anything about a **real battery**. PyBaMM is a model; this is external *model* mismatch, not
  physical truth.
- The *magnitude* of the phantom fault. It swings with C-rate and with the observer's RC tuning
  ([`docs/EXTERNAL_PLANT.md`](../docs/EXTERNAL_PLANT.md) §3d); only the direction — precise, biased,
  therefore refused — is robust. An earlier draft calling it "robust to R1/C1" was wrong and was
  retracted.
- A recovered *injected* fault. The mismatch here is the model-order gap on a sound cell; PyBaMM-side
  degradation (and the DFN) is deferred.

Full accounting: [`docs/EXTERNAL_PLANT.md`](../docs/EXTERNAL_PLANT.md) and
[`LIMITATIONS.md`](../LIMITATIONS.md) §14.
"""),
]


CELLS_04 = [
    md("""
# AstraCell — The External Positive Control

**v0.3 proved AstraCell refuses. It could not prove AstraCell refuses for a *reason*.**

A negative control — a healthy cell, correctly not diagnosed — is passed by a diagnostic that never
diagnoses anything. v0.3 shipped one of those and did not know it. Its external bias gate projected
the very trace it was about to fit and called the projection "structural bias"; since a linear fit to
a residual *is* that projection, the gate's statistic was pinned at 1σ and `REFUSE_MODEL_BIAS` came
back for **any data whatsoever**.

This notebook is how that was found, and what replaced it. We inject a real fault into the PyBaMM
cell and ask the complementary question:

> When the external cell really is broken, does AstraCell find it — and can it still tell a real
> fault from its own model error?

> ⚠️ **Still synthetic.** PyBaMM is a sophisticated *model*, and the fault is one we injected. The
> healthy baseline used to correct the phantom is the *identical simulation*, which no workshop can
> supply — so every number here is an **upper bound** on what a real baseline could deliver. Read
> [`docs/POSITIVE_CONTROL.md`](../docs/POSITIVE_CONTROL.md) §5 before quoting any of it. PyBaMM is
> optional (`pip install -e '.[pybamm]'`); §1 runs without it, the rest skip cleanly.
"""),
    code("""
import matplotlib.pyplot as plt
import numpy as np

from astracell.calibration import (
    CAPACITY_TARGET,
    R0_TARGET,
    build_evidence,
    build_external_observer,
    detection_metrics,
    external_scenario,
    observer_voltage,
    prepare_external,
    pulse_profile,
    run_trials,
    verdict_distribution,
)
from astracell.cell.ocv import NMC_LIKE, ocv_from_table
from astracell.observability.decision import VerdictKind
from astracell.plant import PYBAMM_AVAILABLE

plt.rcParams["figure.dpi"] = 110
SEED, SOC0, N_TIME, CAPACITY_AH, N_TRIALS = 0, 0.9, 600, 5.0, 400
OBSERVER_R0_OHM = 0.025           # the observer's nominal R0; faults are injected in ohms
CATHODE_SCALE = 0.3
CURRENT = pulse_profile(CAPACITY_AH, mean_c_rate=0.5, pulse_c_rate=1.0, n_time=N_TIME)

observer = None                    # populated in §2 when PyBaMM is present
healthy_v = None

print("PyBaMM available:", PYBAMM_AVAILABLE)
"""),
    md("""
---
## 1. Why v0.3's external gate could never have found a fault

No PyBaMM needed. `prepare_external` projected the very trace it was about to fit:

$$b = \\mathrm{FIM}^{-1}S^\\top\\Sigma^{-1}(g - f(\\theta^*)) \\qquad
  \\hat\\delta = \\mathrm{FIM}^{-1}S^\\top\\Sigma^{-1}(g + \\varepsilon - f(\\theta^*))$$

Same projection, same trace, so `b ≡ E[δ̂]`. Substitute into the credibility statistic and it
collapses to something that has nothing to do with the battery:

$$\\mathrm{SNR}_{\\text{total}} = \\frac{|b+\\varepsilon|}{\\sqrt{\\sigma^2+b^2}}, \\qquad
  \\mathbb{E}\\!\\left[\\mathrm{SNR}_{\\text{total}}^2\\right]
  = \\frac{b^2+\\sigma^2}{\\sigma^2+b^2} = 1 \\quad \\textbf{exactly, for any } b$$

**It is a pure noise statistic.** Feed it a sine wave, an absurd 1 V ramp, or a whisper: RMS 1 every
time. Whenever `|b| ≫ σ` — the only regime in which a bias gate has a job — it concentrates on 1σ and
`REFUSE_MODEL_BIAS` is certain.
"""),
    code("""
control = build_external_observer(NMC_LIKE, CAPACITY_AH)
scn = external_scenario(name="degenerate", observer=control, current_a=CURRENT, soc0=SOC0)
nominal = observer_voltage(scn)
t = np.arange(N_TIME)

print(f"{'residual fed to the gate':30s} {'bias':>12s} {'mean':>7s} {'RMS':>7s} {'max':>7s}  refusals")
for label, residual in (("20 mV of sine wave", 0.02 * np.sin(t / 17.0)),
                        ("a 1 V linear ramp (absurd)", np.linspace(0.0, 1.0, N_TIME)),
                        ("a 1 uV whisper", 1e-6 * np.cos(t / 3.0))):
    prepared = prepare_external(scn, nominal + residual)     # no baseline: the v0.3 call
    res = run_trials(scn, 300, seed=SEED, estimator="linear", prepared=prepared)
    refused = res.verdicts.count(VerdictKind.REFUSE_MODEL_BIAS)
    rms = np.sqrt(np.mean(res.snr_bias ** 2))
    print(f"{label:30s} {prepared.target_bias:+12.2%} {res.snr_bias.mean():7.4f} {rms:7.4f} "
          f"{res.snr_bias.max():7.4f}  {refused:3d}/300")

print("\\nA 1 V ramp implies a 1718% capacity deviation. All refused, all at 1 sigma. The 1 uV row")
print("is the degenerate other end: b ~ 0, so the statistic collapses to |noise|/sigma and merely")
print("re-tests the noise. Uninformative in both directions.")
"""),
    md("""
v0.3 was right about a healthy cell for the wrong reason. Its 100% `REFUSE_MODEL_BIAS` was not
evidence about the cell; it was **arithmetic about the estimator**. In §3 we point the same gate at a
cell with a *doubled series resistance* and watch it call the fault "bias".

The fix is a **healthy baseline**. `prepare_external(..., baseline_voltage=g_h)` projects the bias
from a trace known to be healthy, restoring v0.2's convention that the residual is evaluated at the
healthy nominal while the fault lives in the data. The gate then discriminates rather than refuses.
"""),
    md("""
---
## 2. Two real faults, injected into the external plant

* **Contact resistance** `+ΔR` — a corroded tab weld. The differential voltage is `−ΔR·I` to machine
  precision, which is *exactly* the ECM's `∂V/∂R₀ = −I`. **The observer can express this fault.**
* **Cathode particle diffusivity ×0.3** — particle cracking. No lithium and no active material lost,
  so the true `(R₀, capacity)` deviation is `(0, 0)`. It lands squarely in the blind spot of a
  one-RC ECM. **The observer cannot express this change, and must not name it.**
"""),
    code("""
if PYBAMM_AVAILABLE:
    from astracell.plant import (
        HEALTHY, contact_resistance, pybamm_pseudo_ocv, simulate_pybamm_cell, slow_cathode,
    )

    soc, ocv = pybamm_pseudo_ocv(model="SPMe", n_points=200)
    curve = ocv_from_table(soc, ocv, name="pybamm_spme", chemistry="SPMe/Chen2020")
    observer = build_external_observer(curve, CAPACITY_AH)
    healthy_v = simulate_pybamm_cell(CURRENT, 1.0, model="SPMe", soc0=SOC0, fault=HEALTHY).voltage_v

    def evidence_for(fault, magnitude, target):
        faulty = simulate_pybamm_cell(CURRENT, 1.0, model="SPMe", soc0=SOC0, fault=fault).voltage_v
        s = external_scenario(name=fault.name, observer=observer, current_a=CURRENT, soc0=SOC0,
                              target_index=target, fault_magnitude=magnitude)
        return build_evidence(s, healthy_v, faulty), faulty

    delta_r = 0.20 * OBSERVER_R0_OHM
    strong, faulty_v = evidence_for(contact_resistance(delta_r), 0.20, R0_TARGET)
    confounder, cathode_v = evidence_for(slow_cathode(CATHODE_SCALE), 0.0, R0_TARGET)
    ohmic = -delta_r * CURRENT

    print(f"max |(g_f - g_h) + dR*I|   {np.abs((faulty_v - healthy_v) - ohmic).max():.2e} V  <- exactly ohmic")
    print(f"contact resistance, ||rho||  {strong.misfit_paired:.2e}   <- the ECM can say this exactly")
    print(f"cathode diffusivity, ||rho|| {confounder.misfit_paired:.1f}   <- no parameter setting says this")
else:
    print("PyBaMM not installed -- skipping. Install with: pip install -e '.[pybamm]'")
"""),
    code("""
if PYBAMM_AVAILABLE:
    t = np.arange(N_TIME)
    fig, (top, mid, bot) = plt.subplots(3, 1, figsize=(9, 7), sharex=True, height_ratios=[2, 1, 1])
    top.plot(t, healthy_v, color="#2e7d32", lw=1.1, label="healthy PyBaMM SPMe")
    top.plot(t, faulty_v, color="#1565c0", lw=1.0, label=f"+{1e3*delta_r:.2f} mOhm contact resistance")
    top.plot(t, cathode_v, color="#c62828", lw=1.0, ls="--", label=f"cathode diffusivity x{CATHODE_SCALE}")
    top.set(ylabel="terminal voltage (V)", title="Two real faults in a plant AstraCell did not write")
    top.legend(fontsize=8); top.grid(alpha=0.3)

    mid.plot(t, 1e3 * (faulty_v - healthy_v), color="#1565c0", lw=1.0, label="measured differential")
    mid.plot(t, 1e3 * ohmic, color="black", lw=0.8, ls=":", label=r"$-\\Delta R \\cdot I(t)$")
    mid.set(ylabel="(mV)", title="Contact resistance: the differential IS the ECM's R0 direction")
    mid.legend(fontsize=8); mid.grid(alpha=0.3)

    bot.plot(t, 1e3 * (cathode_v - healthy_v), color="#c62828", lw=1.0)
    bot.axhline(0, color="#455a64", lw=0.8, ls=":")
    bot.set(xlabel="time (s)", ylabel="(mV)",
            title="Cathode diffusivity: a shape no first-order ECM can produce")
    bot.grid(alpha=0.3)
    plt.show()
else:
    print("PyBaMM not installed -- skipping the trace figure.")
"""),
    md("""
---
## 3. Baseline subtraction and paired comparison are the same estimator

The brief offered these as alternatives. They are not — the projection is linear:

$$\\mathrm{FIM}^{-1}S^\\top\\Sigma^{-1}(g_f - f) \\;-\\; \\mathrm{FIM}^{-1}S^\\top\\Sigma^{-1}(g_h - f)
  \\;\\equiv\\; \\mathrm{FIM}^{-1}S^\\top\\Sigma^{-1}(g_f - g_h)$$

There was never a choice. What the identity *does* buy is honest noise: two measured traces carry two
independent draws, so the differential has covariance `2Σ` and the CRLB widens by `√2`. **Subtraction
is not free.**
"""),
    code("""
if PYBAMM_AVAILABLE:
    healthy_ev, _ = evidence_for(HEALTHY, 0.0, CAPACITY_TARGET)
    huge, huge_v = evidence_for(contact_resistance(1.00 * OBSERVER_R0_OHM), 1.00, R0_TARGET)

    # v0.3's baseline-free gate, pointed at a cell whose series resistance has DOUBLED.
    v03_scn = external_scenario(name="v0.3", observer=observer, current_a=CURRENT, soc0=SOC0,
                                target_index=R0_TARGET)
    v03 = prepare_external(v03_scn, huge_v)                      # no baseline: exactly the v0.3 call
    v03_res = run_trials(v03_scn, 300, seed=SEED, estimator="linear", prepared=v03)
    print("v0.3's gate on a doubled series resistance (+25 mOhm):")
    print(f"  bias == estimate  {v03.target_bias:+.2%}     SNR_total  "
          f"{np.sqrt(np.mean(v03_res.snr_bias ** 2)):.4f} sigma")
    print(f"  verdicts          {v03_res.verdicts.count(VerdictKind.REFUSE_MODEL_BIAS)}/300 "
          f"REFUSE_MODEL_BIAS   <- it calls the fault 'bias'\\n")

    print(f"healthy phantom on capacity       {healthy_ev.phantom:+.2%}   <- v0.3's headline")
    print(f"healthy phantom on R0             {strong.phantom:+.2%}")
    print(f"raw estimate (R0), 20% fault      {strong.raw_estimate:+.4%}")
    print(f"  minus the phantom               {strong.raw_estimate - strong.phantom:+.4%}")
    print(f"paired estimate (R0), 20% fault   {strong.paired_estimate:+.4%}   <- identical")
    print(f"residual of the identity          "
          f"{abs(strong.raw_estimate - strong.phantom - strong.paired_estimate):.2e}")
    print(f"\\nsigma: raw {strong.raw.target_sigma:.3%}  ->  paired {strong.paired.target_sigma:.3%}"
          f"  (x{strong.paired.target_sigma / strong.raw.target_sigma:.3f})")
else:
    print("PyBaMM not installed -- skipping.")
"""),
    md("""
### Diagnosis is not detection

A **true positive** requires DIAGNOSE *and* an interval that covers the injected truth *and* the right
sign. Scoring only harmful overclaim rewards silence; scoring only diagnosis rewards noise.

Watch what happens at a **100% fault** (a doubled series resistance). The raw path is finally loud
enough to clear its own phantom's gate — and diagnoses in every trial, reporting `+91.4%` against a
truth of `+100%`. An 8.6-point miss inside a 0.23%-wide interval, 145σ of misplaced confidence.
"""),
    code("""
if PYBAMM_AVAILABLE:
    rows = [("raw (20% fault)", strong.raw), ("paired (20% fault)", strong.paired),
            ("raw (100% fault)", huge.raw), ("paired (100% fault)", huge.paired)]
    print(f"{'path':22s} {'diagnose':>9s} {'true pos':>9s} {'overclaim':>10s} {'cov 95%':>8s}")
    for label, ctx in rows:
        m = detection_metrics(run_trials(ctx.scenario, N_TRIALS, seed=SEED, prepared=ctx))
        print(f"{label:22s} {m.diagnosis_rate:9.2f} {m.true_positive_rate:9.2f} "
              f"{m.harmful_overclaim_rate:10.2f} {m.coverage:8.2f}")
    print(f"\\nraw estimate at the 100% fault: {huge.raw_estimate:+.1%}   (truth +100%)")
    print("Right about the fault, wrong about the fault. Every diagnosis is a harmful overclaim.")
else:
    print("PyBaMM not installed -- skipping.")
"""),
    md("""
### Re-scoring v0.3's headline with the honest gate

The gate cannot use `parameter_bias` on data that might be faulted. What it *can* use is the part of
the residual **no parameter setting reproduces** — the out-of-span component `ρ̃ = (I−P)r̃`, which is
exactly independent of any fault present. Convert it to parameter units by Cauchy–Schwarz:

$$b^{\\mathrm{lof}}_i = \\lVert\\tilde\\rho\\rVert \\cdot \\sigma_i$$

It is invariant to the noise scale, and exactly zero when the model reproduces the data. Apply it to
v0.3's healthy cell, where we *know* the entire estimate is bias, and it under-warns by **3.2×**.
"""),
    code("""
if PYBAMM_AVAILABLE:
    b_lof = healthy_ev.raw_lack_of_fit
    truth_is_bias = abs(healthy_ev.raw_estimate)       # the cell is healthy: the estimate IS the bias
    snr_total = truth_is_bias / np.hypot(healthy_ev.raw.target_sigma, b_lof)

    print(f"capacity estimate      {healthy_ev.raw_estimate:+.2%}   <- and it is ALL bias")
    print(f"lack-of-fit bias       {b_lof:+.2%}   <- what the screen sees")
    print(f"SNR_total              {snr_total:.2f} sigma   <- WEAK_EVIDENCE, not REFUSE")
    print(f"fraction of bias seen  {b_lof / truth_is_bias:.0%}")
    print("\\nv0.3's '100% REFUSE_MODEL_BIAS' becomes 'WEAK'. Harmful overclaim is still 0% -- WEAK")
    print("is not DIAGNOSE -- so the headline is corrected, not retracted. But the margin was")
    print("thinner than reported, and a phantom three times smaller would have slipped through.")
else:
    print("PyBaMM not installed -- skipping.")
"""),
    md("""
---
## 4. The magnitude sweep

Where diagnosis begins, where it must not, and the transition between. `b_lof` is zero to
floating-point dust throughout — the ECM reproduces a series resistance *exactly* — so the
credibility gate is inert and the verdicts are pure Cramér–Rao. The transition sits where `σ(ΔR) =
0.021 mΩ` puts it: 5σ is 0.105 mΩ.

Note the true-positive rate saturates at **0.94, not 1.00**. A 95% interval misses its truth 5% of
the time by construction, so 5% of confident, *correct* diagnoses are counted harmful overclaims.
Overclaim floors at `1 − coverage_level` for any detector that diagnoses at all. v0.3's 0% overclaim
came from 0% diagnosis; it is not a number to aspire to.
"""),
    code("""
STYLE = {
    VerdictKind.DIAGNOSE: ("#2e7d32", "diagnose"),
    VerdictKind.WEAK_EVIDENCE: ("#f9a825", "weak"),
    VerdictKind.REFUSE_UNOBSERVABLE: ("#9e9e9e", "refuse: unobservable"),
    VerdictKind.REFUSE_CONFOUNDED: ("#6a1b9a", "refuse: confounded"),
    VerdictKind.REFUSE_MODEL_BIAS: ("#c62828", "refuse: model bias"),
}
SWEEP = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.2, 1.0)

if PYBAMM_AVAILABLE:
    fractions = {k: np.zeros(len(SWEEP)) for k in VerdictKind}
    tpr = []
    print(f"{'dR [mOhm]':>10s} {'SNR':>9s} {'diagnose':>9s} {'true pos':>9s} {'weak':>6s} {'refuse':>7s}")
    for i, frac in enumerate(SWEEP):
        ev, _ = evidence_for(contact_resistance(frac * OBSERVER_R0_OHM), frac, R0_TARGET)
        res = run_trials(ev.paired.scenario, N_TRIALS, seed=SEED, prepared=ev.paired)
        m = detection_metrics(res)
        for k, share in verdict_distribution(res).items():
            fractions[k][i] = share
        tpr.append(m.true_positive_rate)
        snr = abs(ev.paired_estimate) / ev.paired.target_sigma
        print(f"{1e3*frac*OBSERVER_R0_OHM:10.3f} {snr:9.1f} {m.diagnosis_rate:9.2f} "
              f"{m.true_positive_rate:9.2f} {m.weak_rate:6.2f} {m.refusal_rate:7.2f}")

    x = 1e3 * np.array(SWEEP) * OBSERVER_R0_OHM
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bottom = np.zeros(len(SWEEP))
    for k in VerdictKind:
        if fractions[k].max() == 0.0:
            continue
        colour, name = STYLE[k]
        ax.fill_between(x, bottom, bottom + fractions[k], color=colour, alpha=0.85, label=name)
        bottom = bottom + fractions[k]
    ax.plot(x, tpr, color="black", lw=1.6, marker="o", ms=3.5, label="true-positive rate")
    ax.axhline(0.95, color="black", lw=0.7, ls=":")
    ax.set(xscale="log", xlabel=r"injected contact resistance $\\Delta R$ (m$\\Omega$)",
           ylabel="fraction of trials", ylim=(0, 1),
           title="Baseline-corrected verdicts against a real, injected, external fault")
    ax.margins(x=0); ax.legend(fontsize=7, loc="center left", framealpha=0.92)
    plt.show()
else:
    print("PyBaMM not installed -- skipping the sweep.")
"""),
    md("""
---
## 5. The confounder, refused — by one percent

Cathode diffusivity `×0.3` is a **real physical degradation** whose true `(R₀, capacity)` deviation is
`(0, 0)`. That is not asserted from the parameter we set — it is measured. The faulted C/20 curve
shifts by 9.6 mV RMS (an earlier draft wrongly called it "unchanged"), but the shift is **linear in
current** — 199 mV per C over a 10× rate range — hence an overpotential, hence extrapolating to zero.
The equilibrium OCV–SOC relation, and with it the coulombic capacity, is untouched.

Variance alone would diagnose a 13% resistance fault at 158σ and a 119% capacity loss at 579σ. The
lack-of-fit says: *no setting of my parameters produces this differential.* Refuse.

**Watch the margin on the capacity row.**
"""),
    code("""
if PYBAMM_AVAILABLE:
    capacity_ev, _ = evidence_for(slow_cathode(CATHODE_SCALE), 0.0, CAPACITY_TARGET)
    print(f"{'hypothesis':12s} {'paired est':>12s} {'SNR var':>9s} {'b_lof':>9s} {'SNR tot':>9s}  verdict")
    for label, ev in (("R0", confounder), ("capacity", capacity_ev)):
        sv = abs(ev.paired_estimate) / ev.paired.target_sigma
        st = abs(ev.paired_estimate) / np.hypot(ev.paired.target_sigma, ev.lack_of_fit)
        res = run_trials(ev.paired.scenario, N_TRIALS, seed=SEED, prepared=ev.paired)
        top = max(verdict_distribution(res).items(), key=lambda kv: kv[1])[0]
        print(f"{label:12s} {ev.paired_estimate:12.2%} {sv:9.1f} {ev.lack_of_fit:9.2%} {st:9.2f}  "
              f"{top.value.upper()}")
    print("\\nThe capacity hypothesis clears the 2-sigma line at 1.98 sigma -- refused by 1%. The")
    print("screen is doing real work and has nothing to spare. Shorten the window to 300 s and the")
    print("very same hypothesis reaches 3.35 sigma and is merely WEAK.")
else:
    print("PyBaMM not installed -- skipping the confounder.")
"""),
    code("""
if PYBAMM_AVAILABLE:
    weak_ev, _ = evidence_for(contact_resistance(0.002 * OBSERVER_R0_OHM), 0.002, R0_TARGET)
    healthy_r0, _ = evidence_for(HEALTHY, 0.0, R0_TARGET)
    cases = [("healthy", healthy_r0), ("weak fault", weak_ev),
             ("real fault", strong), ("confounded", confounder)]

    labels, tprs, fprs, refusals = [], [], [], []
    for label, ev in cases:
        m = detection_metrics(run_trials(ev.paired.scenario, N_TRIALS, seed=SEED, prepared=ev.paired))
        labels.append(label); tprs.append(m.true_positive_rate)
        fprs.append(m.false_positive_rate); refusals.append(m.refusal_rate)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    idx = np.arange(len(labels))
    ax.bar(idx - 0.26, tprs, 0.25, color="#2e7d32", label="true positive")
    ax.bar(idx, fprs, 0.25, color="#c62828", label="false positive")
    ax.bar(idx + 0.26, refusals, 0.25, color="#9e9e9e", label="refusal")
    ax.set(ylabel="fraction of trials", ylim=(0, 1.05), xticks=idx, xticklabels=labels,
           title="Finds the fault it can express. Refuses the one it cannot.")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.show()
else:
    print("PyBaMM not installed -- skipping the rates figure.")
"""),
    md("""
---
## What this notebook validated, and what it did not

**Validated (synthetically, against a plant we did not design, with a fault we did):**

- v0.3's external bias gate refused **unconditionally**: its "structural bias" was algebraically equal
  to the estimate it gated, so `SNR_total ≡ 1` for any data. A negative control could never have
  revealed this. A positive control did, immediately.
- Baseline subtraction and paired comparison are the **same estimator**, exactly (identity residual
  `6 × 10⁻¹⁶`), at the price of a `√2` wider σ.
- With a healthy baseline the same gate **discriminates**: it recovers a real injected fault at its
  correct magnitude with nominal coverage, weakens a marginal one, and still refuses a real physical
  change it cannot name.
- **Diagnosis is not detection.** The raw path diagnoses a doubled series resistance in every trial
  while reporting it 145σ from the truth.

**Not validated, and not claimable:**

- Anything about a **real battery**. PyBaMM is a model; the fault is one we injected into it.
- **The baseline.** `g_h` here is the *identical simulation*. A real beginning-of-life fingerprint is
  a different day, temperature, SOC window, and cell. Every rate above is an **upper bound**.
- **The screen bounds nothing.** `b_lof` sees only the out-of-span residual. On the one case with known
  truth it captures **31%** of the bias. A structural error lying entirely *in* the span is invisible to
  it — and to every other method that looks only at this experiment.
- **The primary fault is in the model's span by construction.** A contact resistance *is* what an
  ECM's `R₀` is. Recovering it demonstrates the plumbing, not the physics. The confounder is where the
  physics is, and there the margin is **1%**.
- **Capacity fade is still not injected.** It cannot be, cleanly, without breaking the shared-OCV
  control. [`LIMITATIONS.md`](../LIMITATIONS.md) §15.

v0.4 does not make AstraCell better at diagnosing. It makes the refusal **mean something** — and it
found a place where v0.3's refusal did not.

Full accounting: [`docs/POSITIVE_CONTROL.md`](../docs/POSITIVE_CONTROL.md) and
[`LIMITATIONS.md`](../LIMITATIONS.md) §15.
"""),
]


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    n_code = sum(c["cell_type"] == "code" for c in cells)
    print(f"wrote {path} ({len(cells)} cells, {n_code} code)")


def main() -> None:
    write_notebook(NOTEBOOK_01, CELLS_01)
    write_notebook(NOTEBOOK_02, CELLS_02)
    write_notebook(NOTEBOOK_03, CELLS_03)
    write_notebook(NOTEBOOK_04, CELLS_04)


if __name__ == "__main__":
    main()
