"""What the Cramer-Rao bound cannot see.

Run:  python examples/04_model_mismatch.py       (~60 s)

Every number this repository has published assumes the observer's model is *correct*. The
CRLB bounds the variance of an unbiased estimator; it says nothing about an estimator that
is biased because its model is wrong. And the model is wrong -- a first-order ECM with one
lumped thermal node, fitted to a cell that has SOC-dependent resistance, a slow diffusion
branch, a core-to-surface gradient and a laggy thermocouple.

Five acts:

1. The residual the observer cannot explain, and which of its blind spots produce it.
2. The bias that residual creates, and two ways to compute it, one of which is a real fit.
3. **The money plot.** Repeat the experiment a hundred times: the Cramer-Rao floor falls to
   nothing and the bias does not move by one bit. Confidence grows without bound; accuracy
   stops dead at a ceiling. Verdicts flip from DIAGNOSE to REFUSE_MODEL_BIAS.
4. Harder excitation does not remove structural error. It decides which parameter absorbs
   it -- which means the Ds-optimal test planner of ``examples/03`` is optimising a quantity
   that can be traded against credibility, and does not know it.
5. A nuisance parameter is where model error goes to hide. Free the ammeter offset and the
   per-cell biases collapse, in exchange for a fictitious shunt calibration.

The point is not that AstraCell now looks better. Four of its published numbers get worse.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astracell.duty import pulse_train
from astracell.observability import (
    BiasConvergenceError,
    ParameterSpec,
    ParamKind,
    bias_aware_snr,
    bias_ceiling,
    crlb,
    detection_snr,
    fisher_information,
    local_specs,
    parameter_bias,
    pseudo_true_bias,
    residual_score,
    sensitivities,
    solve_bias,
    structural_residual,
    with_current_bias,
)
from astracell.observability.decision import assess_under_mismatch
from astracell.pack import PackTopology, nominal_pack
from astracell.pack.simulate import simulate
from astracell.plant import NO_MISMATCH, REALISTIC_MISMATCH, MismatchModel, simulate_plant
from astracell.sensors import NoiseModel
from astracell.sensors.topology import realistic_topology

FIGURES = Path(__file__).resolve().parents[1] / "reports" / "figures"
SEED = 0
SENSED_CELL = 12  # carries a thermocouple
BLIND_CELL = 10  # does not

R0, CAPACITY, HA = 0, 1, 2  # indices within local_specs(cell)
FAULTS = (0.20, 0.05, 0.40)  # the magnitudes the README hunts for


def rule(title: str) -> None:
    print(f"\n{'=' * 84}\n{title}\n{'=' * 84}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pack = PackTopology(n_modules=4, cells_per_module=8)
    params = nominal_pack(pack, seed=SEED)
    topology = realistic_topology(pack, n_temp_sensors=4)
    noise = NoiseModel()
    duty = pulse_train(1200.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0)

    # ------------------------------------------------------------------- act 1
    rule("1. The residual the observer cannot explain")
    control = structural_residual(params, NO_MISMATCH, duty.current_a, duty.dt_s)
    print("  control: with the mismatch switched off, the residual is exactly zero")
    print(f"           non-zero entries = {np.count_nonzero(control)}  (bitwise, not 'small')")
    print("           so anything below is mismatch, not a bug in the harness.\n")

    residual = structural_residual(params, REALISTIC_MISMATCH, duty.current_a, duty.dt_s)
    plant = simulate_plant(params, duty.current_a, duty.dt_s, REALISTIC_MISMATCH)
    print(
        f"  residual at t = 0          : {np.abs(residual[0]).max():.1e}   "
        "(zero by construction, which is what makes theta* well defined)"
    )
    print(
        f"  voltage residual, RMS      : {1e3 * np.sqrt((residual[:, :, 0] ** 2).mean()):7.3f} mV"
        f"   vs {1e3 * noise.voltage_sigma_v:.2f} mV of sensor noise"
    )
    print(f"  voltage residual, peak     : {1e3 * np.abs(residual[:, :, 0]).max():7.3f} mV")
    print(
        f"  temperature residual, RMS  : {np.sqrt((residual[:, :, 1] ** 2).mean()):7.3f} K "
        f"   vs {noise.temp_sigma_k:.2f} K"
    )
    print(f"  core-surface gradient, peak: {np.abs(plant.core_surface_gradient_k).max():7.3f} K")
    print("\n  The model error is four times the noise floor. The estimator will not")
    print("  ignore it -- it will explain it, using the only vocabulary it has: parameters.")

    print("\n  Which blind spot produces which bias? (one knob at a time, cell 10)")
    specs = local_specs(BLIND_CELL)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    knobs = {
        "R0 varies with SOC": MismatchModel(r0_soc_slope=REALISTIC_MISMATCH.r0_soc_slope),
        "slow diffusion branch": MismatchModel(
            second_rc_r_ohm=REALISTIC_MISMATCH.second_rc_r_ohm,
            second_rc_tau_s=REALISTIC_MISMATCH.second_rc_tau_s,
        ),
        "core/surface gradient": MismatchModel(
            core_surface_resistance_k_per_w=REALISTIC_MISMATCH.core_surface_resistance_k_per_w
        ),
        "thermocouple lag": MismatchModel(temp_sensor_tau_s=REALISTIC_MISMATCH.temp_sensor_tau_s),
        "all four together": REALISTIC_MISMATCH,
    }
    print(f"    {'blind spot':<24s} {'R0 bias':>10s} {'capacity bias':>15s} {'hA bias':>12s}")
    for name, mismatch in knobs.items():
        piece = structural_residual(params, mismatch, duty.current_a, duty.dt_s)
        bias = parameter_bias(sens, piece, topology, noise)
        print(
            f"    {name:<24s} {100 * bias[R0]:+9.3f}% {100 * bias[CAPACITY]:+14.3f}% "
            f"{100 * bias[HA]:+11.1f}%"
        )

    print("\n  Capacity's bias is almost entirely the missing diffusion branch: a slow voltage")
    print("  droop is indistinguishable from coulombs that never left the cell. And hA's bias")
    print("  is dominated by the ELECTRICAL blind spots, not the thermal ones -- because an")
    print("  uninstrumented cell's cooling is only ever seen through R0(T). Model error in")
    print("  the voltage channel arrives disguised as a thermal fault.")

    # ------------------------------------------------------------------- act 2
    rule("2. The bias, and a real fit to check it against")
    print("  parameter_bias   : one Gauss-Newton step from theta*. Cheap, first-order.")
    print("  pseudo_true_bias : that step iterated to where the fit actually rests.\n")

    print(f"  {'cell':>5s} {'param':>9s} {'linearised':>12s} {'iterated fit':>14s} {'gap':>8s}")
    for cell in (SENSED_CELL, BLIND_CELL):
        cell_specs = local_specs(cell)
        cell_sens = sensitivities(params, duty.current_a, duty.dt_s, cell_specs)
        linear = parameter_bias(cell_sens, residual, topology, noise)
        try:
            fitted = pseudo_true_bias(
                params, REALISTIC_MISMATCH, duty.current_a, duty.dt_s, topology, noise, cell_specs
            )
        except BiasConvergenceError:
            print(f"  {cell:5d} {'(all three)':>11s}  no pseudo-true value exists: cell {cell}")
            print("           carries no thermocouple, so its cooling coefficient is")
            print("           unidentifiable and the fit has a whole ray of equally good")
            print("           answers. parameter_bias reports inf there rather than pick one.")
            continue
        for i, spec in enumerate(cell_specs):
            gap = abs(linear[i] - fitted[i]) / max(abs(fitted[i]), 1e-12)
            print(
                f"  {cell:5d} {spec.label():>9s} {100 * linear[i]:+11.3f}% "
                f"{100 * fitted[i]:+13.3f}% {100 * gap:7.1f}%"
            )

    print("\n  The linearisation is a scale estimate, not a correction. Where the two differ")
    print("  by tens of percent, the iterated fit is the one to quote. They converge as the")
    print("  mismatch shrinks (tests/test_mismatch.py pins that at 0.4% by 3% mismatch).")

    # ------------------------------------------------------------------- act 3
    rule("3. More data. The floor falls; the bias does not move.")
    fim = fisher_information(sens, topology, noise)
    score = residual_score(sens, residual, topology, noise)
    replicas = np.array([1, 2, 5, 10, 25, 100, 1000, 10000], dtype=float)

    print(
        f"  {'replicas':>9s} {'R0 CRLB':>10s} {'R0 bias':>10s} "
        f"{'R0 SNR_var':>12s} {'R0 SNR_tot':>12s}"
    )
    curves: dict[str, list[float]] = {"variance": [], "total": []}
    for k in replicas:
        variance = crlb(k * fim)
        bias = solve_bias(k * fim, k * score)
        snr_v = float(detection_snr(variance, FAULTS[R0])[R0])
        snr_t = float(bias_aware_snr(variance, bias, FAULTS[R0])[R0])
        curves["variance"].append(snr_v)
        curves["total"].append(snr_t)
        print(
            f"  {k:9.0f} {100 * np.sqrt(variance[R0]):9.4f}% {100 * bias[R0]:+9.4f}% "
            f"{snr_v:11.1f}s {snr_t:11.2f}s"
        )

    ceilings = bias_ceiling(solve_bias(fim, score), FAULTS[R0])
    print("\n  The bias is bit-for-bit constant. SNR_variance grows as sqrt(k), without bound.")
    print(f"  SNR_total stops at the ceiling m/|b| = {ceilings[R0]:.2f} sigma and stays there.")
    print("  Ten thousand repetitions of this experiment buy you nothing but false confidence.")

    print("\n  Verdicts on cell 10, with the ammeter believed (I_bias held fixed):")
    print(
        f"  {'hypothesis':<12s} {'SNR_var':>9s} {'bias':>11s} {'SNR_tot':>9s} {'ceiling':>9s}"
        f"  {'variance-only':>14s} -> {'bias-aware':<18s}"
    )
    flipped = 0
    for i, spec in enumerate(specs):
        magnitude = FAULTS[i]
        naive = assess_under_mismatch(
            sens, topology, noise, specs, spec, magnitude, np.zeros_like(residual)
        )
        honest = assess_under_mismatch(sens, topology, noise, specs, spec, magnitude, residual)
        flipped += naive.kind is not honest.kind
        ceiling = honest.bias_ceiling
        print(
            f"  {spec.label():<12s} {naive.snr:8.2f}s {100 * (honest.bias or 0):+10.2f}% "
            f"{honest.snr_total:8.2f}s {ceiling:8.2f}s  "
            f"{naive.kind.value.upper():>14s} -> {honest.kind.value.upper():<18s}"
        )
    print(f"\n  {flipped} of {len(specs)} verdicts changed. The full refusal:\n")
    capacity_target = ParameterSpec(BLIND_CELL, ParamKind.CAPACITY)
    print(
        assess_under_mismatch(
            sens, topology, noise, specs, capacity_target, FAULTS[CAPACITY], residual
        ).render()
    )

    # ------------------------------------------------------------------- act 4
    rule("4. Excitation does not remove structural error. It routes it.")
    print("  Raising the pulse amplitude pins R0 down through the IR drop, so the slow")
    print("  polarisation residual has nowhere left to go except capacity.\n")
    print(
        f"  {'pulse':>7s} {'I std':>7s} | {'R0 CRLB':>9s} {'R0 bias':>9s} {'R0 ceiling':>11s}"
        f" | {'Q CRLB':>8s} {'Q bias':>9s} {'Q ceiling':>10s}"
    )
    amplitudes = np.array([0.25, 0.5, 1.0, 1.5, 1.75, 2.0, 2.5])
    sweep: dict[str, list[float]] = {"r0_ceiling": [], "q_ceiling": [], "r0_bias": []}
    for amplitude in amplitudes:
        profile = pulse_train(1200.0, 1.0, mean_c_rate=0.2, pulse_c_rate=float(amplitude))
        s = sensitivities(params, profile.current_a, profile.dt_s, specs)
        r = structural_residual(params, REALISTIC_MISMATCH, profile.current_a, profile.dt_s)
        b = parameter_bias(s, r, topology, noise)
        std = np.sqrt(crlb(fisher_information(s, topology, noise)))
        ceil = bias_ceiling(b, 1.0)  # per unit magnitude; scale below
        sweep["r0_ceiling"].append(FAULTS[R0] * ceil[R0])
        sweep["q_ceiling"].append(FAULTS[CAPACITY] * ceil[CAPACITY])
        sweep["r0_bias"].append(float(b[R0]))
        print(
            f"  {amplitude:6.2f}C {profile.current_std_a:6.1f}A | {100 * std[R0]:8.3f}% "
            f"{100 * b[R0]:+8.3f}% {FAULTS[R0] * ceil[R0]:10.2f}s | {100 * std[CAPACITY]:7.3f}% "
            f"{100 * b[CAPACITY]:+8.3f}% {FAULTS[CAPACITY] * ceil[CAPACITY]:9.2f}s"
        )

    crossing = int(np.argmax(sweep["r0_ceiling"]))
    print("\n  Both CRLBs improve monotonically with excitation, as example 03 promised.")
    print(
        f"  R0's bias shrinks with it too, from {100 * sweep['r0_bias'][0]:.1f}% and through "
        f"zero near {amplitudes[crossing]:.2f}C."
    )
    print(
        f"  Capacity's ceiling falls from {sweep['q_ceiling'][0]:.2f} to "
        f"{sweep['q_ceiling'][-1]:.2f} sigma over the same sweep, and never recovers."
    )

    print(
        f"\n  Do NOT read the {sweep['r0_ceiling'][crossing]:.0f} sigma at "
        f"{amplitudes[crossing]:.2f}C as an operating point. It is a zero crossing, and a"
    )
    print("  ceiling at a zero crossing is a pole in a quantity we computed by ASSUMING we")
    print("  know the plant. Perturb one assumed plant parameter and watch the pole move:\n")

    # Re-derive R0's bias at the crossing under a +/-10% error in the assumed diffusion
    # resistance. This is the whole warning, so it is measured here rather than asserted.
    at_crossing = pulse_train(
        1200.0, 1.0, mean_c_rate=0.2, pulse_c_rate=float(amplitudes[crossing])
    )
    sens_c = sensitivities(params, at_crossing.current_a, at_crossing.dt_s, specs)
    print(f"    {'assumed R2':>12s} {'R0 bias at ' + f'{amplitudes[crossing]:.2f}C':>20s}")
    for factor in (0.9, 1.0, 1.1):
        guess = replace(
            REALISTIC_MISMATCH, second_rc_r_ohm=REALISTIC_MISMATCH.second_rc_r_ohm * factor
        )
        r_guess = structural_residual(params, guess, at_crossing.current_a, at_crossing.dt_s)
        b_guess = parameter_bias(sens_c, r_guess, topology, noise)
        print(f"    {1e3 * guess.second_rc_r_ohm:9.2f} mOhm {100 * b_guess[R0]:+19.3f}%")

    print("\n  A ten percent error in one number we guessed flips the SIGN of R0's bias at the")
    print("  crossing, so the crossing relocates by a large fraction of a C-rate. Tuning a duty")
    print("  cycle to null a bias inferred from your own guess of the plant calibrates against")
    print("  the guess, not the cell. The robust reading is the trend, not the pole.")

    print("\n  And the trend is enough to matter: the D-optimal / Ds-optimal planner in")
    print(f"  examples/03 will happily recommend a {amplitudes[-1]:.1f}C pulse train to sharpen a")
    print(
        f"  capacity estimate, driving its ceiling from {sweep['q_ceiling'][0]:.2f} to "
        f"{sweep['q_ceiling'][-1]:.2f} sigma as it does so."
    )
    print("  It optimises variance. It cannot see bias. A real defect, not a nuance.")

    # ------------------------------------------------------------------- act 5
    rule("5. A nuisance parameter is where model error hides")
    nuisance = with_current_bias(specs)
    sens_n = sensitivities(params, duty.current_a, duty.dt_s, nuisance)
    fixed = parameter_bias(sens, residual, topology, noise)
    freed = parameter_bias(sens_n, residual, topology, noise, nuisance)

    print(f"  {'param':>9s} {'ammeter believed':>18s} {'ammeter free':>15s}")
    for i, spec in enumerate(specs):
        print(f"  {spec.label():>9s} {100 * fixed[i]:+17.3f}% {100 * freed[i]:+14.3f}%")
    print(f"  {'I_bias':>9s} {'--':>18s} {freed[3]:+14.4f} A")

    print(
        f"\n  Capacity's bias collapses from {100 * fixed[CAPACITY]:+.2f}% to "
        f"{100 * freed[CAPACITY]:+.2f}%. The capacity estimate really does get better:"
    )
    print("  a common-mode voltage residual looks like a mis-calibrated shunt, and freeing")
    print("  a nuisance regressor that spans the residual is the textbook cure for")
    print("  omitted-variable bias. This is not a trick. It works.")
    print(f"\n  The price: the fit reports a shunt offset of {freed[3]:+.3f} A that is not a")
    print("  shunt offset at all. It is model error, comfortably inside the 2 A prior, and")
    print("  nothing in the Cramer-Rao bound will ever flag it. Calibrate your ammeter")
    print("  against this number and you have calibrated in your own modelling mistake.")
    print("\n  Choosing what to let float chooses which parameters are protected from model")
    print(
        f"  error, and which lie you tell instead. hA's bias stays near "
        f"{100 * freed[HA]:+.0f}% either way, because no current offset can mimic cooling."
    )

    # ------------------------------------------------------------------- figure
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 4.6))

    left.plot(
        replicas,
        curves["variance"],
        marker="o",
        color="#c62828",
        label=r"CRLB only:  $m/\sqrt{\mathrm{CRLB}}$",
    )
    left.plot(
        replicas,
        curves["total"],
        marker="s",
        color="#1565c0",
        label=r"bias-aware: $m/\sqrt{\mathrm{CRLB}+b^2}$",
    )
    left.axhline(
        ceilings[R0],
        ls="--",
        color="#37474f",
        lw=1.1,
        label=rf"ceiling $m/|b|$ = {ceilings[R0]:.1f}$\sigma$",
    )
    left.axhline(5.0, ls=":", color="#2e7d32", lw=1.1, label=r"5$\sigma$ decision threshold")
    left.set(
        xscale="log",
        yscale="log",
        xlabel="replications of the same experiment",
        ylabel=r"detection SNR ($\sigma$)",
        title="Confidence grows without bound.\nAccuracy stops at the bias ceiling.",
    )
    left.legend(fontsize=8, loc="upper left")
    left.grid(alpha=0.3, which="both")

    right.plot(
        amplitudes,
        sweep["r0_ceiling"],
        marker="o",
        color="#1565c0",
        label=r"$R_0$ ceiling (20% fault)",
    )
    right.plot(
        amplitudes,
        sweep["q_ceiling"],
        marker="s",
        color="#c62828",
        label="capacity ceiling (5% fault)",
    )
    right.axhline(5.0, ls=":", color="#2e7d32", lw=1.1, label=r"5$\sigma$ threshold")
    right.set(
        yscale="log",
        xlabel="pulse amplitude (C-rate)",
        ylabel=r"bias ceiling ($\sigma$)",
        title=(
            "Excitation does not remove structural error.\nIt decides which parameter absorbs it."
        ),
    )
    right.legend(fontsize=8)
    right.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(FIGURES / "model_mismatch.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # residual trace, because a picture of the thing itself is worth having
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    observer = simulate(params, duty.current_a, duty.dt_s)
    ax.plot(
        observer.time_s,
        1e3 * residual[:, BLIND_CELL, 0],
        color="#c62828",
        lw=0.9,
        label=f"structural residual, cell {BLIND_CELL}",
    )
    ax.axhspan(
        -1e3 * noise.voltage_sigma_v,
        1e3 * noise.voltage_sigma_v,
        color="#9e9e9e",
        alpha=0.35,
        label=r"$\pm 1\sigma$ sensor noise",
    )
    ax.set(
        xlabel="time (s)",
        ylabel="plant $-$ observer (mV)",
        title=(
            "What the ECM cannot say, said anyway:\nthe residual it must explain using parameters"
        ),
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "structural_residual.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(
        "\n  figures: reports/figures/model_mismatch.png, reports/figures/structural_residual.png"
    )


if __name__ == "__main__":
    main()
