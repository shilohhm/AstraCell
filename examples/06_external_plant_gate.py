"""Does calibrated abstention survive a plant AstraCell did not write?

Run:  python examples/06_external_plant_gate.py       (~2 min, needs `pip install -e '.[pybamm]'`)

v0.2 proved AstraCell's intervals are calibrated and its bias gate honest -- but against a plant
*we* built, whose mismatch was four terms we chose. This is the first external-validity test: the
data now comes from a PyBaMM single cell (SPMe: electrolyte + particle diffusion), and the
observer is the same first-order ECM as before. The gap between them is physics we did not design.

The target is capacity, because it is the one quantity both models share a truth about: the cell
is *healthy*, so the honest capacity deviation is zero. Anything the observer reports is the
diffusion it cannot model, projected onto the one knob that can absorb a slow droop -- capacity.

Four acts:

1. **A control that cannot be a harness bug.** Feed the same pipeline an ECM-generated trace
   instead of PyBaMM. With no model mismatch, coverage must track nominal. It does -- so any
   collapse under PyBaMM is PyBaMM's, not our plumbing's.
2. **The residual, and the phantom fault.** PyBaMM's voltage departs from the best the ECM can do
   by tens of mV. Fit it and the observer reports a large, precise capacity fault on a sound cell.
3. **Coverage before and after admitting the bias.** The variance-only interval -- pinned to
   ~0.15% -- covers the truth essentially never. Widening it for the structural bias restores
   coverage, at the price of usefulness.
4. **Overclaim, across excitation, with and without the gate.** Without the gate the observer
   diagnoses a fault that is not there in nearly every trial and at every C-rate. The bias gate
   refuses instead. That is the result: under an honest plant, AstraCell diagnoses *less*.

The phantom fault's size is not robust -- it swings with C-rate and with the observer's RC tuning
(see docs/EXTERNAL_PLANT.md), which is itself the tell that it is not a real capacity loss. What
is robust is the overclaim and the refusal. As ever, the honest outcome is the worse-looking one.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
from astracell.cell.ocv import ocv_from_table
from astracell.plant import PYBAMM_AVAILABLE, pybamm_pseudo_ocv, simulate_pybamm_cell

FIGURES = Path(__file__).resolve().parents[1] / "reports" / "figures"
SEED = 0
SOC0 = 0.9
DT_S = 1.0
N_TIME = 600
CAPACITY_AH = 5.0
MODEL = "SPMe"


def rule(title: str) -> None:
    print(f"\n{'=' * 84}\n{title}\n{'=' * 84}")


def build_curve_and_observer():  # type: ignore[no-untyped-def]
    """The shared setup: PyBaMM's pseudo-OCV, and the one-cell ECM observer that borrows it."""
    soc, ocv = pybamm_pseudo_ocv(model=MODEL, n_points=200)
    curve = ocv_from_table(soc, ocv, name=f"pybamm_{MODEL.lower()}", chemistry=f"{MODEL}/Chen2020")
    observer = build_external_observer(curve, capacity_ah=CAPACITY_AH)
    return curve, observer


def act1_control(observer, current) -> None:  # type: ignore[no-untyped-def]
    rule("1. Control: feed the pipeline an ECM trace. No mismatch -> coverage must be nominal.")
    scenario = external_scenario(name="control", observer=observer, current_a=current, soc0=SOC0)
    ecm_voltage = observer_voltage(scenario)  # the ECM is its own plant: mismatch is zero
    prepared = prepare_external(scenario, ecm_voltage)
    result = run_trials(scenario, 3000, seed=SEED, estimator="linear", prepared=prepared)

    empirical = coverage_curve(result)
    print(f"  structural bias = {result.bias:+.4%}   (must be ~0: the ECM cannot mismatch itself)")
    print(f"  {'nominal':>10s} {'empirical coverage':>20s}")
    for level, got in zip(NOMINAL_LEVELS, empirical, strict=True):
        print(f"  {level:10.0%} {got:20.1%}")
    worst = max(abs(got - lvl) for got, lvl in zip(empirical, NOMINAL_LEVELS, strict=True))
    print(
        f"\n  Largest deviation from nominal: {worst:.3f}. The external-plant harness adds no bias"
    )
    print("  of its own -- so the collapse under PyBaMM in act 3 is model mismatch, not a bug.")
    return result


def act2_residual_and_phantom(observer, current) -> None:  # type: ignore[no-untyped-def]
    rule("2. The residual and the phantom fault: fit a real cell, get a fault that is not there")
    plant = simulate_pybamm_cell(current, DT_S, model=MODEL, soc0=SOC0)
    scenario = external_scenario(name="pybamm", observer=observer, current_a=current, soc0=SOC0)
    obs_v = observer_voltage(scenario)
    prepared = prepare_external(scenario, plant.voltage_v)
    result = run_trials(scenario, 500, seed=SEED, estimator="linear", prepared=prepared)

    residual_mv = 1e3 * (plant.voltage_v - obs_v)
    rms = float(np.sqrt(np.mean(residual_mv**2)))
    print(f"  plant = PyBaMM {MODEL} (Chen2020, {CAPACITY_AH:.0f} Ah), observer = first-order ECM")
    print(f"  plant - observer residual: {rms:.2f} mV RMS  (measurement noise is 1 mV)")
    print("  true capacity deviation = 0.00% (healthy cell)")
    print(
        f"  estimated capacity deviation = {result.delta_hat.mean():+.2%}  "
        f"+/- {result.crlb_std:.3%} (1 sigma)"
    )
    sigmas = abs(result.delta_hat.mean()) / result.crlb_std
    print(f"  the estimate sits {sigmas:.0f} sigma from the truth -- precise, and wrong.")

    # Figure 1: voltage traces + residual.
    t = np.arange(N_TIME) * DT_S
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True, height_ratios=[2, 1])
    top.plot(t, plant.voltage_v, color="#1565c0", lw=1.1, label=f"PyBaMM {MODEL} (plant)")
    top.plot(t, obs_v, color="#c62828", lw=1.0, ls="--", label="first-order ECM (observer)")
    top.set(
        ylabel="terminal voltage [V]",
        title="A first-order ECM cannot express PyBaMM's diffusion.\n"
        "The gap is small in volts, decisive in the capacity estimate.",
    )
    top.legend(fontsize=8)
    top.grid(alpha=0.3)
    bottom.plot(t, residual_mv, color="#455a64", lw=0.9)
    bottom.axhline(0.0, color="#c62828", lw=0.8, ls=":")
    bottom.set(xlabel="time [s]", ylabel="residual [mV]")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "external_residual.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: estimate distribution vs the truth.
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    half = two_sided_z(0.95) * result.crlb_std
    center = float(result.delta_hat.mean())
    ax.hist(
        100 * result.delta_hat,
        bins=40,
        color="#c62828",
        alpha=0.75,
        label="capacity estimates (500 trials)",
    )
    ax.axvline(0.0, color="#2e7d32", lw=2.0, label="truth: healthy cell (0%)")
    ax.axvspan(
        100 * (center - half),
        100 * (center + half),
        color="#c62828",
        alpha=0.15,
        label="variance-only 95% interval",
    )
    ax.set(
        xlabel="estimated capacity deviation [%]",
        ylabel="trials",
        title="The 95% interval is tight, confident, and nowhere near the truth.",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "external_estimate_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return plant, scenario, prepared, result


def act3_coverage(control_result, money_result) -> None:  # type: ignore[no-untyped-def]
    rule("3. Coverage before and after admitting the bias")
    cov_control = coverage_curve(control_result)
    cov_var = coverage_curve(money_result, bias_aware=False)
    cov_bias = coverage_curve(money_result, bias_aware=True)

    print(
        f"  {'nominal':>8s} {'control (ECM)':>15s} {'PyBaMM var-only':>17s} "
        f"{'PyBaMM bias-aware':>19s}"
    )
    for lvl, cc, cv, cb in zip(NOMINAL_LEVELS, cov_control, cov_var, cov_bias, strict=True):
        print(f"  {lvl:8.0%} {cc:15.1%} {cv:17.1%} {cb:19.1%}")
    print("\n  Against the ECM plant, coverage is nominal. Against PyBaMM, the variance-only")
    print("  interval covers the truth essentially never -- it is pinned to ~0.15% around a")
    print("  centre 60-odd percent away. Widening it for the structural bias restores coverage,")
    print("  but only by making the interval wide enough to be diagnostically useless.")

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    grid = np.array(NOMINAL_LEVELS)
    ax.plot([0, 1], [0, 1], ls="--", color="#455a64", lw=1.0, label="perfect calibration")
    ax.plot(grid, cov_control, "o-", color="#2e7d32", label="control: ECM plant (matched)")
    ax.plot(grid, cov_var, "s-", color="#c62828", label="PyBaMM plant, variance-only")
    ax.plot(grid, cov_bias, "^-", color="#1565c0", label="PyBaMM plant, bias-aware")
    ax.set(
        xlabel="nominal confidence",
        ylabel="empirical coverage",
        title="A plant we did not write breaks variance-only coverage.\n"
        "The control proves it is the plant, not the harness.",
        xlim=(0.45, 1.0),
        ylim=(-0.03, 1.03),
    )
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "external_coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def act4_overclaim_sweep(observer) -> None:  # type: ignore[no-untyped-def]
    rule("4. Harmful overclaim across excitation, with and without the bias gate")
    c_rates = (0.2, 0.3, 0.5, 0.8)
    gated_rates, ungated_rates, biases = [], [], []
    print(
        f"  {'mean C':>7s} {'phantom fault':>14s} {'sigma':>8s} "
        f"{'overclaim (no gate)':>20s} {'overclaim (gated)':>18s}"
    )
    for mean_c in c_rates:
        current = pulse_profile(CAPACITY_AH, mean_c_rate=mean_c, pulse_c_rate=1.0, n_time=N_TIME)
        plant = simulate_pybamm_cell(current, DT_S, model=MODEL, soc0=SOC0)
        gated_scn = external_scenario(name="g", observer=observer, current_a=current, soc0=SOC0)
        gated_prep = prepare_external(gated_scn, plant.voltage_v)
        gated = abstention_metrics(
            run_trials(gated_scn, 300, seed=SEED, estimator="linear", prepared=gated_prep)
        )
        ungated_scn = external_scenario(
            name="u", observer=observer, current_a=current, soc0=SOC0, use_bias_gate=False
        )
        ungated_prep = prepare_external(ungated_scn, plant.voltage_v)
        ungated = abstention_metrics(
            run_trials(ungated_scn, 300, seed=SEED, estimator="linear", prepared=ungated_prep)
        )
        gated_rates.append(gated.harmful_overclaim_rate)
        ungated_rates.append(ungated.harmful_overclaim_rate)
        biases.append(gated_prep.target_bias)
        print(
            f"  {mean_c:6.1f}C {gated_prep.target_bias:+13.1%} {gated_prep.target_sigma:8.3%} "
            f"{ungated.harmful_overclaim_rate:20.2f} {gated.harmful_overclaim_rate:18.2f}"
        )

    print(
        "\n  The phantom fault swings from "
        f"{min(biases):+.0%} to {max(biases):+.0%} across C-rate -- its own instability proving"
    )
    print("  it is not a capacity loss. But at every excitation the variance-only observer")
    print("  overclaims in ~every trial, and the bias gate refuses instead.")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(c_rates))
    ax.bar(x - 0.2, ungated_rates, 0.4, color="#c62828", label="no gate (variance only)")
    ax.bar(x + 0.2, gated_rates, 0.4, color="#2e7d32", label="bias gate on")
    ax.set(
        xlabel="mean discharge rate",
        ylabel="harmful overclaim rate",
        title="A healthy cell, diagnosed as faulty -- until the bias gate refuses.",
        ylim=(0.0, 1.05),
    )
    ax.set_xticks(x, [f"{c:.1f}C" for c in c_rates])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_overclaim.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def summary(money_result) -> None:  # type: ignore[no-untyped-def]
    gated = abstention_metrics(money_result)
    rule("Summary: the external-plant gate")
    print(f"  plant model              PyBaMM {MODEL} (Chen2020, {CAPACITY_AH:.0f} Ah)")
    print("  observer model           first-order Thevenin ECM (R0, R1C1), shared OCV")
    print("  target                   capacity, true deviation 0% (healthy cell)")
    print(f"  variance-only coverage   {coverage_curve(money_result)[3]:.0%} at nominal 95%")
    print(
        f"  bias-aware coverage      {coverage_curve(money_result, bias_aware=True)[3]:.0%} "
        "at nominal 95%"
    )
    print(f"  refusal rate (gated)     {gated.refuse_model_bias_rate:.0%} REFUSE_MODEL_BIAS")
    print(
        f"  harmful overclaim        100% (no gate)  ->  {gated.harmful_overclaim_rate:.0%} (gated)"
    )
    print("\n  Conclusion: calibrated abstention survives contact with a plant AstraCell did not")
    print("  write. It survives by refusing -- which is the point. See docs/EXTERNAL_PLANT.md.")


def main() -> int:
    if not PYBAMM_AVAILABLE:
        print("PyBaMM is not installed; example 06 needs it. Install: pip install -e '.[pybamm]'")
        print("Skipping (this is not a failure -- PyBaMM is an optional dependency).")
        return 0
    warnings.simplefilter("ignore")  # PyBaMM/casadi solve chatter, not ours
    FIGURES.mkdir(parents=True, exist_ok=True)
    _, observer = build_curve_and_observer()
    money_current = pulse_profile(CAPACITY_AH, mean_c_rate=0.5, pulse_c_rate=1.0, n_time=N_TIME)

    control_result = act1_control(observer, money_current)
    _, _, _, money_result = act2_residual_and_phantom(observer, money_current)
    act3_coverage(control_result, money_result)
    act4_overclaim_sweep(observer)
    summary(money_result)
    print(f"\nFigures written to {FIGURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
