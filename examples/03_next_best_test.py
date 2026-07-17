"""What should I measure next?

Run:  python examples/03_next_best_test.py

AstraCell refused to diagnose a 40% cooling fault on cell 10 (no thermocouple). Refusing
is half an answer. This is the other half: rank the diagnostic tests a BMS could actually
command, by how much each would sharpen *the hypothesis we asked about*.

Six acts:

1. Rank by D-optimality (``det FIM``) and watch it crown the wrong test -- because it is
   optimising the volume of a confidence ellipsoid whose other axes nobody asked about.
2. Rank by Ds-optimality (the target's marginal) and watch it crown the right one.
3. Note that neither score is the *decision* rule. Nats-per-minute picks the most
   information-efficient test; a decision needs the **cheapest test that crosses the
   threshold**. Run that one, and watch the verdict flip.
4. Turn on realistic correlated noise. Ten minutes of pulsing now beats a permanently
   installed thermocouple, because under 1/f noise the thermocouple's own signature is
   DC and whitening destroys it. The recommendation inverts.
5. Shrink the fault until no test, no sensor, and no affordable protocol reaches the
   threshold. Watch AstraCell refuse, and say what it would take.
6. Ask the same question of v0.9's fitted dynamics: which excitation makes the RC branch
   ``(R1, C1)`` identifiable? A pulse train the BMS can command drops ``VIF(R1)`` below the
   gate a 1C discharge leaves it far above -- the prescription that earns the branch. Its
   honest limit: the same test does not earn the *capacity* verdict, whose refusal is model
   bias, not confounding, and no excitation removes bias.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astracell.calibration import (
    C1_TARGET,
    CAPACITY_TARGET,
    R1_TARGET,
    build_external_observer,
    constant_profile,
    external_specs_4param,
    pulse_profile,
    vif_under_excitation,
)
from astracell.cell.ocv import NMC_LIKE
from astracell.duty import pulse_train
from astracell.observability import (
    ParameterSpec,
    ParamKind,
    all_specs,
    crlb,
    default_test_library,
    detection_snr,
    local_specs,
    rank_tests,
    recommend_temp_sensor,
    render_ranking,
    sensitivities,
    with_current_bias,
)
from astracell.observability.experiment import plan_tests, recommend_test
from astracell.pack import PackTopology, nominal_pack
from astracell.sensors import NoiseModel
from astracell.sensors.topology import realistic_topology

FIGURES = Path(__file__).resolve().parents[1] / "reports" / "figures"
SEED = 0
BLIND_CELL = 10
MAGNITUDE = 0.40

# Act 6: the single-cell ECM the v0.9 dynamics fit runs on, mirroring the positive control --
# a ~5 Ah cell discharged over a 600 s window from 90% SOC.
ECM_CAP = 5.0
ECM_N = 600
ECM_SOC0 = 0.9


def rule(title: str) -> None:
    print(f"\n{'=' * 84}\n{title}\n{'=' * 84}")


def snr_of(fim, index: int, magnitude: float) -> float:
    return float(detection_snr(np.array([crlb(fim)[index]]), magnitude)[0])


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pack = PackTopology(n_modules=4, cells_per_module=8)
    params = nominal_pack(pack, seed=SEED)
    topology = realistic_topology(pack, n_temp_sensors=4)
    baseline = pulse_train(1200.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0)
    library = default_test_library()

    target = ParameterSpec(BLIND_CELL, ParamKind.HA)
    specs = with_current_bias(local_specs(BLIND_CELL))
    index = specs.index(target)

    print(f"  hypothesis: {target.label()} at {100 * MAGNITUDE:.0f}%")
    print(f"  cell {BLIND_CELL} carries no thermocouple. Thermocouples: {topology.temp_cells}")
    print(f"\n  test library ({len(library)} candidates):")
    for t in library:
        print(
            f"    {t.name:<24s} {t.cost_s:5.0f} s  I_std {t.profile.current_std_a:5.1f} A"
            f"  -- {t.rationale}"
        )

    # ------------------------------------------------------------------- act 1&2
    rule("1+2. D-optimality crowns the wrong test. Ds-optimality crowns the right one.")
    noise = NoiseModel()
    scores, fim_before, fim_per_test = rank_tests(
        params, topology, noise, specs, target, MAGNITUDE, library=library, baseline=baseline
    )
    print(render_ranking(scores, fim_before, target))

    by_d = max(scores, key=lambda s: s.eig_nats)
    by_ds = max(scores, key=lambda s: s.eig_target_nats)
    print("\n  Ranked by TOTAL information gained:")
    print(
        f"    D-optimal  (det FIM, all params) : {by_d.name:<24s} {by_d.eig_nats:5.2f} nats"
        f"  -> SNR {by_d.snr_after:.2f} sigma"
    )
    print(
        f"    Ds-optimal (hA[{BLIND_CELL}] alone)      : {by_ds.name:<24s} "
        f"{by_ds.eig_target_nats:5.2f} nats  -> SNR {by_ds.snr_after:.2f} sigma"
    )
    if by_d.name != by_ds.name:
        print(
            f"\n  They disagree, and Ds is right: {by_ds.snr_after:.2f} sigma beats "
            f"{by_d.snr_after:.2f} sigma."
        )
        print(f"  D-optimality crowns '{by_d.name}' because its {by_d.cost_s:.0f} seconds sharpen")
        print("  R0 and capacity, inflating det(FIM). Neither is the parameter we asked about.")
        print("  You are diagnosing one thing. Optimise that axis, not the ellipsoid volume.")

    control = next(s for s in scores if s.name == "rest_60s")
    print(
        f"\n  negative control 'rest_60s': {control.eig_target_nats:.4f} nats, "
        f"SNR {snr_of(fim_before, index, MAGNITUDE):.3f} -> {control.snr_after:.3f}"
    )
    print("  Doing nothing gains nothing. If it ever outranks a 2C pulse, the score is broken.")

    # ------------------------------------------------------------------- act 3
    rule("3. Neither score is the decision rule. Pick the cheapest crossing.")
    snr_before = snr_of(fim_before, index, MAGNITUDE)
    var_before = crlb(fim_before)[index]
    print(
        f"  before: SNR {snr_before:.2f} sigma, 1sig +/-{100 * np.sqrt(var_before):.2f}%  -> REFUSE"
    )

    efficient = scores[0]  # already sorted by Ds nats/min
    print(
        f"\n  most nats-per-minute : {efficient.name:<24s} -> {efficient.snr_after:5.2f} sigma"
        f"  {'(clears)' if efficient.resolves() else '(STILL REFUSED)'}"
    )

    best = recommend_test(scores)
    clearing = [s for s in scores if s.resolves()]
    if best is not None:
        print(
            f"  cheapest crossing 5s : {best.name:<24s} -> {best.snr_after:5.2f} sigma"
            f"   [{len(clearing)} of {len(scores)} tests clear]"
        )
        print(f"\n  Run '{best.name}' for {best.cost_s:.0f} s:")
        print(f"    SNR   {snr_before:6.2f}  ->  {best.snr_after:6.2f} sigma")
        print(f"    1sig  {100 * np.sqrt(var_before):5.2f}%  ->  {100 * best.crlb_std_after:5.2f}%")
        print(
            f"    gained {best.eig_target_nats:.2f} nats about hA[{BLIND_CELL}]; VIF "
            f"{best.vif_after:.2f}, still separable"
        )
        print("\n  Verdict: REFUSE -> DIAGNOSE, with no new hardware.")

    plan = plan_tests(fim_before, fim_per_test, library, index, MAGNITUDE)
    print("\n  Greedy sequential plan (information adds; no re-simulation):")
    print(plan.render())

    # ------------------------------------------------------------------- act 4
    rule("4. Realistic correlated noise (rho = 0.9). Excite; do not instrument.")
    noisy = NoiseModel(voltage_rho=0.9, temp_rho=0.9)
    scores_n, fim_before_n, fim_per_test_n = rank_tests(
        params, topology, noisy, specs, target, MAGNITUDE, library=library, baseline=baseline
    )
    print(render_ranking(scores_n, fim_before_n, target))

    snr_n = snr_of(fim_before_n, index, MAGNITUDE)
    print(f"\n  before: SNR {snr_n:.2f} sigma  -> REFUSE")
    print(
        f"  most nats-per-minute : {scores_n[0].name:<24s} -> {scores_n[0].snr_after:5.2f} sigma"
        f"  {'(clears)' if scores_n[0].resolves() else '(STILL REFUSED)'}"
    )

    best_n = recommend_test(scores_n)
    full_specs = all_specs(pack.n_cells)
    full_sens = sensitivities(params, baseline.current_a, baseline.dt_s, full_specs)
    ranked = recommend_temp_sensor(full_sens, topology, noisy, full_specs, target, MAGNITUDE)

    print("\n  Two levers, priced against each other:")
    if best_n is not None:
        print(
            f"    excite   : run '{best_n.name}' for {best_n.cost_s / 60:.0f} min "
            f"-> {best_n.snr_after:5.2f} sigma"
        )
    print(
        f"    instrument: fit the best possible thermocouple (cell {ranked[0][0]}) "
        f"-> {ranked[0][1]:5.2f} sigma"
    )

    if best_n is not None and best_n.snr_after > ranked[0][1]:
        print(
            f"\n  Ten minutes of pulsing beats a permanently installed sensor, "
            f"{best_n.snr_after:.2f} vs {ranked[0][1]:.2f} sigma."
        )
        print("  Under 1/f noise the thermocouple's own signature is DC, so whitening")
        print("  destroys it (see examples/02, part 3). The correct engineering answer")
        print("  is a protocol, not a part -- and it is the opposite of the white-noise answer.")

    # ------------------------------------------------------------------- act 5
    rule("5. Shrink the fault until nothing works. Then say so.")
    print("  The 40% fault above is easy. Walk the magnitude down and watch every")
    print("  intervention fail in turn. A 30-minute service-bay budget is 3 tests.\n")

    print(
        f"  {'fault':>7s} {'single test':>14s} {'best sensor':>13s} "
        f"{'30-min protocol':>17s} {'verdict':>18s}"
    )
    for magnitude in (0.40, 0.15, 0.08):
        sc, f0, fs = rank_tests(
            params, topology, noisy, specs, target, magnitude, library=library, baseline=baseline
        )
        single = recommend_test(sc)
        sensor = recommend_temp_sensor(full_sens, topology, noisy, full_specs, target, magnitude)[0]
        plan = plan_tests(f0, fs, library, index, magnitude, max_tests=3)
        long_plan = plan_tests(f0, fs, library, index, magnitude, max_tests=16)

        if single is not None:
            outcome = f"run 1 test ({single.cost_s / 60:.0f} min)"
        elif plan.resolved:
            outcome = f"{plan.total_cost_s / 60:.0f}-min protocol"
        elif long_plan.resolved:
            outcome = f"needs {long_plan.total_cost_s / 60:.0f} min"
        else:
            outcome = "REFUSE"

        print(
            f"  {100 * magnitude:6.0f}% {(single.name if single else 'none clears'):>14s} "
            f"{sensor[1]:12.2f}s {plan.final_snr:16.2f}s {outcome:>18s}"
        )

    print("\n  40%: one 10-minute pulse train. Do not buy the thermocouple.")
    print("  15%: no single test, no sensor -- but information adds, so ~48 min of")
    print("       repeated pulsing gets there. AstraCell reports the schedule.")
    print("   8%: nothing in the library, no sensor, and no affordable protocol reaches")
    print("       5 sigma. The honest output is a refusal plus the reason.")
    print("\n  A diagnostic that cannot say 'I cannot see this' is not a diagnostic.")

    # ------------------------------------------------------------------- act 6
    rule("6. The same question for v0.9's fitted dynamics: what excitation earns R1, C1?")
    print("  Acts 1-5 planned the next-best-test for a thermal fault on the pack. v0.9 (fit the")
    print("  dynamics) asks it of a different fault: a single cell's fast RC branch, now fitted as")
    print("  (R0, capacity, R1, C1). VIF is read off the FIM, so it scores a profile before the")
    print("  BMS ever commands it -- the same Ds-logic, a new target.\n")

    ecm = build_external_observer(NMC_LIKE, ECM_CAP)
    specs4 = external_specs_4param()
    excitations = {
        "constant 1.0C (Oxford)": constant_profile(ECM_CAP, 1.0, ECM_N),
        "pulse 60s @0.5C": pulse_profile(ECM_CAP, mean_c_rate=0.4, pulse_c_rate=0.5, n_time=ECM_N),
        "pulse 60s @1C": pulse_profile(ECM_CAP, mean_c_rate=0.5, pulse_c_rate=1.0, n_time=ECM_N),
        "pulse 120s @1C": pulse_profile(
            ECM_CAP, mean_c_rate=0.5, pulse_c_rate=1.0, n_time=ECM_N, period_s=120.0
        ),
    }
    vifs = {
        name: vif_under_excitation(ecm, cur, soc0=ECM_SOC0, specs=specs4)
        for name, cur in excitations.items()
    }
    peaks = {name: float(np.max(np.abs(cur)) / ECM_CAP) for name, cur in excitations.items()}

    print(f"    {'excitation':<22s} {'peak':>5s} {'VIF R1':>8s} {'VIF C1':>8s}   verdict")
    for name, v in vifs.items():
        clears = v[R1_TARGET] < 10.0
        print(
            f"    {name:<22s} {peaks[name]:4.2f}C {v[R1_TARGET]:8.2f} {v[C1_TARGET]:8.2f}   "
            f"{'identifiable' if clears else 'CONFOUNDED'}"
        )

    base = vifs["constant 1.0C (Oxford)"]
    print(f"\n  1C discharge: VIF(R1) {base[R1_TARGET]:.1f} >> 10 -- R1 is collinear with R0.")
    print("  On the real Oxford cell it is worse: a median ~287 (REAL_CELL.md, Fit the dynamics).")

    crossings = [(peaks[n], n, v) for n, v in vifs.items() if v[R1_TARGET] < 10.0]
    if crossings:
        cheap = min(crossings, key=lambda c: c[0])
        best = min(crossings, key=lambda c: c[2][R1_TARGET])
        print(
            f"  cheapest crossing : {cheap[1]} at {cheap[0]:.2f}C peak -> VIF(R1) "
            f"{cheap[2][R1_TARGET]:.2f} (just clears 10)."
        )
        print(
            f"  most identifiable : {best[1]} -> VIF(R1) {best[2][R1_TARGET]:.2f}"
            " -- identifiable under a pulse the BMS can command."
        )

    caps = np.array([v[CAPACITY_TARGET] for v in vifs.values()])
    print("\n  But this earns the DYNAMICS, not the capacity verdict. Capacity's VIF barely moves")
    print(
        f"  across every excitation ({caps.min():.1f}..{caps.max():.1f}) -- it was never the"
        " confounded parameter."
    )
    print("  The real-cell capacity verdict is REFUSE_MODEL_BIAS, a bias gate blind to VIF, and")
    print("  harder excitation only routes that bias, never removes it (examples/04, part 4).")
    print("  Two failures, two fixes: excitation for the dynamics, an OCV-drift observer for Q.")

    # ------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, (sc, before, title) in zip(
        axes,
        [(scores, snr_before, "white noise"), (scores_n, snr_n, "AR(1), rho = 0.9")],
        strict=True,
    ):
        names = [s.name for s in sc][::-1]
        vals = [s.snr_after for s in sc][::-1]
        colours = ["#2e7d32" if v >= 5 else ("#f9a825" if v >= 2 else "#9e9e9e") for v in vals]
        ax.barh(names, vals, color=colours)
        ax.axvline(5.0, ls="--", color="#2e7d32", lw=1.2)
        ax.axvline(2.0, ls="--", color="#f9a825", lw=1.2)
        ax.axvline(before, ls=":", color="#37474f", lw=1.4)
        ax.set_xscale("log")
        ax.set_xlabel(f"SNR after one test (sigma)  --  {title}")
        ax.grid(alpha=0.3, axis="x", which="both")
    axes[0].set_title(
        f"Next-best-test for hA[{BLIND_CELL}] at {100 * MAGNITUDE:.0f}%\n"
        "dotted = before any test; dashed = 2 and 5 sigma thresholds",
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "next_best_test.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  figure: {(FIGURES / 'next_best_test.png').relative_to(FIGURES.parents[1])}")


if __name__ == "__main__":
    main()
