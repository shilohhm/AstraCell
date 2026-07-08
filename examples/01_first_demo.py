"""AstraCell first demo: what can this pack's sensors actually see?

Run:  python examples/01_first_demo.py

Six acts:

1. Build a 4x8 pack and instrument it the way a real BMS is instrumented:
   every cell's voltage, four thermocouples, one pack-current shunt.
2. Inject a resistance fault, a capacity fault, a cooling fault, and two sensor
   biases. Simulate. Add noise.
3. Ask the Fisher information which of those faults the data could *possibly*
   support a claim about.
4. Render the pack map. Grey cells are refusals.
5. Render the detectability heatmap: SNR over (excitation, fault magnitude).
6. Take one cell where AstraCell refuses, and compute which single additional
   thermocouple would change the answer.

Nothing here detects a fault. This is the layer that decides whether detecting a
fault is a well-posed question, which is the layer that has to come first.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astracell.duty import pulse_train
from astracell.faults import (
    apply_physical_faults,
    apply_sensor_faults,
    cooling_weakness,
    high_internal_resistance,
    reduced_capacity,
    temp_sensor_bias,
    voltage_sensor_bias,
)
from astracell.observability import (
    ParamKind,
    all_specs,
    detectability_heatmap,
    grey_cell_map,
)
from astracell.observability.decision import assess, sensor_recommendation
from astracell.observability.sensitivity import ParameterSpec, sensitivities
from astracell.pack import PackTopology, nominal_pack, simulate
from astracell.sensors import NoiseModel, measure
from astracell.sensors.topology import realistic_topology
from astracell.viz.heatmap import plot_detectability_heatmap, plot_min_detectable
from astracell.viz.packmap import plot_pack_map

FIGURES = Path(__file__).resolve().parents[1] / "reports" / "figures"
SEED = 0

# Cells chosen to make the point. Cell 12 carries a thermocouple; cell 10 does not.
FAULT_CELL_R0 = 5
FAULT_CELL_CAPACITY = 17
FAULT_CELL_COOLING = 10  # no thermocouple -> AstraCell will refuse
SENSED_COOLING_CELL = 12  # has a thermocouple -> AstraCell will answer


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------------- act 1
    rule("1. Pack and sensor topology")
    pack = PackTopology(n_modules=4, cells_per_module=8)
    healthy = nominal_pack(pack, seed=SEED)
    topology = realistic_topology(pack, n_temp_sensors=4)
    noise = NoiseModel()

    print(f"  {topology.summary()}")
    print(f"  thermocouples on cells {topology.temp_cells}")
    print(
        f"  voltage noise 1-sigma {1e3 * noise.voltage_sigma_v:.1f} mV | "
        f"temperature noise 1-sigma {noise.temp_sigma_k:.1f} K"
    )
    print("\n  Note the asymmetry: 32 voltage channels, 4 temperature channels,")
    print("  1 current channel. That asymmetry is the whole story.")

    # ---------------------------------------------------------------- act 2
    rule("2. Inject faults, simulate, measure")
    duty = pulse_train(1200.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0)
    print(f"  duty cycle: {duty.name}, {duty.n_samples} samples at {duty.dt_s:.0f} s")
    print(f"  mean current {duty.mean_current_a:.1f} A, std {duty.current_std_a:.1f} A")

    physical = [
        high_internal_resistance(FAULT_CELL_R0, 0.20),
        reduced_capacity(FAULT_CELL_CAPACITY, 0.05),
        cooling_weakness(FAULT_CELL_COOLING, 0.40),
    ]
    sensor_faults = [
        voltage_sensor_bias(channel=2, bias_v=5e-3),
        temp_sensor_bias(channel=1, bias_k=2.0),
    ]
    for f in physical:
        print(f"  injected: {f.describe()}")
    for f in sensor_faults:
        print(f"  injected: {f.describe()}")

    faulted = apply_physical_faults(healthy, physical)
    assert healthy.r0_ohm[FAULT_CELL_R0] != faulted.r0_ohm[FAULT_CELL_R0], (
        "injector mutated nothing"
    )
    assert np.isclose(healthy.r0_ohm[FAULT_CELL_R0] * 1.20, faulted.r0_ohm[FAULT_CELL_R0])

    sim = simulate(faulted, duty.current_a, duty.dt_s)
    meas = apply_sensor_faults(measure(sim, topology, noise, rng), sensor_faults)
    print(
        f"\n  simulated: SOC {sim.soc.min():.3f}..{sim.soc.max():.3f}, "
        f"T {sim.temp_k.min() - 273.15:.2f}..{sim.temp_k.max() - 273.15:.2f} degC"
    )
    print(f"  measured : {meas.voltage_v.shape[1]} V channels, {meas.temp_k.shape[1]} T channels")

    # ---------------------------------------------------------------- act 3
    rule("3. What could this data possibly support a claim about?")
    print("  Cramer-Rao bounds. These are limits on EVERY unbiased estimator,")
    print("  not on a particular algorithm. Below 2 sigma, no detector can work.\n")

    specs = all_specs(pack.n_cells)
    print(
        f"  differentiating {len(specs)} parameters "
        f"({2 * len(specs)} simulations, central differences)..."
    )
    sens = sensitivities(healthy, duty.current_a, duty.dt_s, specs)

    hypotheses = [
        (ParameterSpec(FAULT_CELL_R0, ParamKind.R0), 0.20),
        (ParameterSpec(FAULT_CELL_CAPACITY, ParamKind.CAPACITY), 0.05),
        (ParameterSpec(SENSED_COOLING_CELL, ParamKind.HA), 0.40),
        (ParameterSpec(FAULT_CELL_COOLING, ParamKind.HA), 0.40),
    ]
    verdicts = []
    for target, magnitude in hypotheses:
        recommendation = None
        verdict = assess(sens, topology, noise, specs, target, magnitude)
        if verdict.refused and target.kind is ParamKind.HA:
            recommendation = sensor_recommendation(sens, topology, noise, specs, target, magnitude)
            verdict = assess(
                sens, topology, noise, specs, target, magnitude, recommendation=recommendation
            )
        verdicts.append(verdict)
        has_tc = "yes" if target.cell in topology.temp_cells else "no"
        print(f"  --- {target.label()}  (thermocouple on this cell: {has_tc})")
        print(verdict.render())
        print()

    refused = [v for v in verdicts if v.refused]
    print(
        f"  AstraCell answered {len(verdicts) - len(refused)} of {len(verdicts)} hypotheses "
        f"and refused {len(refused)}."
    )

    # ---------------------------------------------------------------- act 4
    rule("4. Pack maps")
    for kind, magnitude, name in [
        (ParamKind.R0, 0.20, "r0"),
        (ParamKind.CAPACITY, 0.05, "capacity"),
        (ParamKind.HA, 0.40, "cooling"),
    ]:
        grey = grey_cell_map(
            healthy, duty.current_a, duty.dt_s, topology, noise, kind=kind, magnitude=magnitude
        )
        counts = grey.counts()
        print(
            f"  {kind.value:12s} at {100 * magnitude:2.0f}%  ->  "
            + "  ".join(f"{k}: {v:2d}" for k, v in counts.items())
        )
        fault_cell = {
            "r0": FAULT_CELL_R0,
            "capacity": FAULT_CELL_CAPACITY,
            "cooling": FAULT_CELL_COOLING,
        }[name]
        fig = plot_pack_map(grey, pack, fault_cell=fault_cell)
        fig.savefig(FIGURES / f"packmap_{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        if kind is ParamKind.HA:
            n_obs, n_all = counts["observable"], pack.n_cells
            print(f"\n  Cooling faults are identifiable on {n_obs} of {n_all} cells.")
            print(f"  Those are exactly the cells carrying a thermocouple: {topology.temp_cells}")
            print("\n  A thermocouple informs about the cell it sits on, and almost nothing")
            print("  else. Conduction carries too little hA information to a neighbour.")
            print("  Note that the far corner BEATS the sensor's next-door neighbour:\n")
            for c in (4, 3, 9, 0):
                d = min(pack.grid_distance(c, s) for s in topology.temp_cells)
                note = "  <- thermocouple" if c in topology.temp_cells else ""
                print(f"    cell {c:2d}  grid distance {d}  ->  SNR {grey.snr[c]:6.2f} sigma{note}")
            print("\n  Distance is not a sufficient statistic. The Fisher information knows")
            print("  about thermal mass, anisotropy, boundary effects, excitation and noise.")
            print("  A hop count knows about none of them.")

    # ---------------------------------------------------------------- act 5
    rule("5. Detectability heatmaps: excitation vs fault magnitude")
    heatmaps = {}
    for kind, name in [(ParamKind.R0, "r0"), (ParamKind.HA, "cooling")]:
        cell = FAULT_CELL_R0 if kind is ParamKind.R0 else FAULT_CELL_COOLING
        result = detectability_heatmap(healthy, topology, noise, cell=cell, kind=kind)
        heatmaps[kind] = result
        lo, hi = result.crlb_std[0], result.crlb_std[-1]
        c_lo, c_hi = result.excitation_c_rate[0], result.excitation_c_rate[-1]
        print(
            f"  {kind.value:12s} on cell {cell}: "
            f"CRLB 1-sigma falls {100 * lo:.3g}% -> {100 * hi:.3g}% "
            f"as excitation goes {c_lo:.2f}C -> {c_hi:.2f}C"
        )
        print(
            f"    smallest fault visible at 5 sigma: "
            f"{100 * result.min_detectable_magnitude()[0]:.3g}% -> "
            f"{100 * result.min_detectable_magnitude()[-1]:.3g}%"
        )
        print(
            f"    cond(R0, capacity) pair: {result.r0_capacity_condition[0]:.3g} -> "
            f"{result.r0_capacity_condition[-1]:.3g}"
        )

        fig = plot_detectability_heatmap(result)
        fig.savefig(FIGURES / f"detectability_{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        fig = plot_min_detectable(result)
        fig.savefig(FIGURES / f"min_detectable_{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # ---------------------------------------------------------------- act 6
    rule("6. The refusal, and what would fix it")
    target = ParameterSpec(FAULT_CELL_COOLING, ParamKind.HA)
    before = assess(sens, topology, noise, specs, target, 0.40)
    print(f"  A 40% cooling fault was injected on cell {FAULT_CELL_COOLING}.")
    print("  It is really there. AstraCell still refuses:\n")
    print(before.render())

    from astracell.observability.mask import recommend_temp_sensor

    ranked = recommend_temp_sensor(sens, topology, noise, specs, target, 0.40)
    print("\n  Counterfactual sensor placement (no re-simulation, just a row mask):")
    for cell, snr in ranked[:5]:
        print(f"    + thermocouple on cell {cell:2d}  ->  SNR {snr:6.2f} sigma")

    best_cell, _ = ranked[0]
    improved = topology.with_temp_sensor_at(best_cell)
    after = assess(sens, improved, noise, specs, target, 0.40)
    print(f"\n  With one extra thermocouple on cell {best_cell}:\n")
    print(after.render())

    ratio = before.crlb_std / after.crlb_std if after.crlb_std > 0 else float("inf")
    print(
        f"\n  The Cramer-Rao floor improved {ratio:.1f}x, from "
        f"+/-{100 * before.crlb_std:.1f}% to +/-{100 * after.crlb_std:.1f}%."
    )
    print(f"  Verdict changed: {before.kind.value.upper()} -> {after.kind.value.upper()}")

    # A second, cheaper answer to the same question. The FIM gives both.
    cooling_map = heatmaps[ParamKind.HA]
    snr_by_excitation = 0.40 / cooling_map.crlb_std
    clears = np.flatnonzero(snr_by_excitation >= 5.0)
    print("\n  ...but adding hardware is not the only way to buy identifiability.")
    print("  Heat generation scales as I^2, so a harder current pulse makes the")
    print("  cell's own voltage work as a thermometer, through R0(T):\n")
    for k in (0, len(cooling_map.excitation_c_rate) // 2, -1):
        c_rate = cooling_map.excitation_c_rate[k]
        print(
            f"    excitation {c_rate:4.2f}C  ->  hA[{FAULT_CELL_COOLING}] SNR "
            f"{snr_by_excitation[k]:7.2f} sigma"
        )
    if clears.size:
        threshold = cooling_map.excitation_c_rate[clears[0]]
        print(
            f"\n  A {threshold:.2f}C pulse makes cell {FAULT_CELL_COOLING}'s cooling fault "
            f"observable with no new sensor."
        )
        print("  Two ways to answer 'what should I measure next?', from one Fisher matrix.")
        print("  (This heatmap ignores cross-cell confounding, so it is optimistic;")
        print("   see LIMITATIONS.md section 5.)")

    grey_before = grey_cell_map(
        healthy, duty.current_a, duty.dt_s, topology, noise, kind=ParamKind.HA, magnitude=0.40
    )
    grey_after = grey_cell_map(
        healthy, duty.current_a, duty.dt_s, improved, noise, kind=ParamKind.HA, magnitude=0.40
    )
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0))
    plot_pack_map(
        grey_before,
        pack,
        ax=axes[0],
        fault_cell=FAULT_CELL_COOLING,
        title=f"Cooling fault identifiability: {topology.n_temp} thermocouples",
    )
    plot_pack_map(
        grey_after,
        pack,
        ax=axes[1],
        fault_cell=FAULT_CELL_COOLING,
        title=f"...and with one more, on cell {best_cell}",
    )
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.tight_layout()
    fig.savefig(FIGURES / "sensor_placement_before_after.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    rule("Figures written")
    for path in sorted(FIGURES.glob("*.png")):
        print(f"  {path.relative_to(FIGURES.parents[1])}")
    print("\n  Read LIMITATIONS.md before believing any of these numbers.")


if __name__ == "__main__":
    main()
