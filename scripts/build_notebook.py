"""Generate ``notebooks/01_identifiability_study.ipynb``.

The notebook is a build artifact, not a hand-maintained source file. Cells live here
as plain strings, which means they are diffable, greppable, and cannot accumulate
stale outputs in version control.

Run:  python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_identifiability_study.ipynb"


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


CELLS = [
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


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    n_code = sum(c["cell_type"] == "code" for c in CELLS)
    print(f"wrote {NOTEBOOK} ({len(CELLS)} cells, {n_code} code)")


if __name__ == "__main__":
    main()
