"""Is the headline claim robust, or is it an artifact of the noise model?

Run:  python examples/02_noise_robustness.py       (~40 s)

``examples/01_first_demo.py`` reports that a 20% resistance fault is identifiable to
+/-0.13% (1 sigma). That number leans on two assumptions:

1. Measurement noise is **white**, so 1200 samples average it down by sqrt(1200) = 35x.
2. The pack **current is known exactly**, rather than measured with a ~0.5% error.

Both are false. This script measures what they cost. The answer is not "everything gets
worse":

* Capacity degrades exactly as the effective-sample-size formula predicts.
* Resistance and cooling **get better** under strongly correlated noise.
* A thermocouple's contribution to a cooling diagnosis collapses from 90% to 1%, and the
  instrumented-beats-uninstrumented ordering inverts.

Everything below is computed on the **same 97-parameter spec set as the README's headline
table** -- every cell's R0, capacity and hA, plus the current bias -- so cross-cell
confounding is carried and the comparison is apples-to-apples. Using ``local_specs`` here
instead would silently inflate the resistance SNR from 149 to 174 sigma by assuming every
other cell in the pack is already known.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astracell.duty import constant_current, pulse_train
from astracell.observability import (
    ParameterSpec,
    ParamKind,
    all_specs,
    crlb,
    fisher_information,
    prior_information,
    sensitivities,
    variance_inflation,
    with_current_bias,
)
from astracell.pack import PackTopology, nominal_pack
from astracell.sensors import NoiseModel
from astracell.sensors.topology import SensorTopology, realistic_topology

FIGURES = Path(__file__).resolve().parents[1] / "reports" / "figures"
SEED = 0
SENSED_CELL = 12  # carries a thermocouple
BLIND_CELL = 10  # does not


def rule(title: str) -> None:
    print(f"\n{'=' * 82}\n{title}\n{'=' * 82}")


def verdict(snr: float) -> str:
    return "DIAGNOSE" if snr >= 5 else ("WEAK_EVIDENCE" if snr >= 2 else "REFUSE")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pack = PackTopology(n_modules=4, cells_per_module=8)
    params = nominal_pack(pack, seed=SEED)
    topology = realistic_topology(pack, n_temp_sensors=4)
    duty = pulse_train(1200.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0)

    specs = with_current_bias(all_specs(pack.n_cells))  # CURRENT_BIAS_SPEC is last
    bias = len(specs) - 1
    print(f"  {len(specs)} parameters: 3 per cell x {pack.n_cells} cells, + 1 global current bias")
    print("  differentiating the pack (this is the slow part) ...")
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)

    def index_of(cell: int, kind: ParamKind) -> int:
        return specs.index(ParameterSpec(cell, kind))

    # ------------------------------------------------------------------ part 1
    rule("1. The pack current is not known exactly")
    print("  The BMS records a current that differs from the one that actually flowed by")
    print("  an unknown constant. Carry it as a nuisance parameter with a Gaussian prior")
    print("  equal to the shunt's accuracy (2 A), and marginalise over it.\n")
    print("  Because the bias is the LAST parameter, one sensitivity tensor gives both:")
    print("    current known      = crlb( fim[:-1, :-1] )   (bias held fixed)")
    print("    current unknown    = crlb( fim )[:-1]        (bias marginalised out)\n")

    flat = constant_current(1200.0, 1.0, c_rate=0.2)
    sens_flat = sensitivities(params, flat.current_a, flat.dt_s, specs)

    for name, s, profile in (
        ("pulse train, 1.0C", sens, duty),
        ("constant current", sens_flat, flat),
    ):
        data = fisher_information(s, topology, NoiseModel())  # no prior
        known = np.sqrt(crlb(data[:-1, :-1]))
        full = data + prior_information(specs, NoiseModel())
        unknown = np.sqrt(crlb(full))
        vif = variance_inflation(full)

        print(f"  --- {name}  (current std {profile.current_std_a:.1f} A)")
        print(f"      {'param':10s} {'I known':>10s} {'I unknown':>11s} {'cost':>8s} {'VIF':>8s}")
        for kind in (ParamKind.R0, ParamKind.CAPACITY, ParamKind.HA):
            i = index_of(BLIND_CELL, kind)
            print(
                f"      {specs[i].label():10s} {100 * known[i]:9.3f}% {100 * unknown[i]:10.3f}% "
                f"{unknown[i] / known[i]:7.2f}x {vif[i]:8.2f}"
            )
        print(f"      I_bias     {'':10s} {unknown[bias]:10.4f} A   (prior was 2.0000 A)\n")

    print("  Under pulsed excitation the cost is near nil: 32 voltage channels pin a")
    print("  common-mode offset ~400x better than the shunt's own spec. R0 and hA pay")
    print("  nothing; capacity pays 18%.")
    print("\n  Under constant current the whole design collapses. A constant current offset")
    print("  is indistinguishable from every cell being slightly more resistive, and from a")
    print("  slow capacity drift. Every VIF crosses the multicollinearity threshold of 10 --")
    print("  by one to two orders of magnitude -- and R0 alone degrades 6.5x. With no")
    print("  excitation there is no way to tell the ammeter from the pack.")
    print("\n  Excitation buys isolation, not merely precision.")

    # ------------------------------------------------------------------ part 2
    rule("2. Measurement noise is not white")
    print("  Real AFE noise has a 1/f component. Model it as AR(1) with lag-1 correlation")
    print("  rho. Whitening is a scaled first difference, (x[t]-rho*x[t-1])/sqrt(1-rho^2),")
    print("  whose effect on information is an exact reciprocal pair:")
    print("      a CONSTANT    sensitivity keeps  (1-rho)/(1+rho)  of its information")
    print("      an ALTERNATING one         GAINS (1+rho)/(1-rho)")
    print("  At rho=0.9 that is a 19x loss and a 19x gain. The question is not how much")
    print("  information we lose. It is which parameters lose it.\n")

    rhos = np.array([0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    kinds = (ParamKind.R0, ParamKind.CAPACITY, ParamKind.HA)
    curves: dict[ParamKind, list[float]] = {k: [] for k in kinds}
    for rho in rhos:
        noise = NoiseModel(voltage_rho=float(rho), temp_rho=float(rho))
        std = np.sqrt(crlb(fisher_information(sens, topology, noise, specs=specs)))
        for kind in kinds:
            curves[kind].append(float(std[index_of(BLIND_CELL, kind)]))

    print(
        f"  {'rho':>6s} {'N_eff':>8s} {'DC ref':>9s} {'R0 1sig':>10s} "
        f"{'Q 1sig':>10s} {'hA 1sig':>10s}"
    )
    for j, rho in enumerate(rhos):
        n_eff = duty.n_samples * (1 - rho) / (1 + rho)
        print(
            f"  {rho:6.2f} {n_eff:8.0f} {np.sqrt((1 + rho) / (1 - rho)):8.1f}x "
            f"{100 * curves[ParamKind.R0][j]:9.3f}% {100 * curves[ParamKind.CAPACITY][j]:9.3f}% "
            f"{100 * curves[ParamKind.HA][j]:9.2f}%"
        )

    def degrade(kind: ParamKind) -> float:
        return curves[kind][-1] / curves[kind][0]

    print("\n  At rho = 0.99, relative to white noise:")
    print(
        f"    capacity  x{degrade(ParamKind.CAPACITY):5.1f}   "
        f"(DC reference predicts x{np.sqrt(1.99 / 0.01):.1f})"
    )
    print(f"    R0        x{degrade(ParamKind.R0):5.2f}")
    print(f"    hA        x{degrade(ParamKind.HA):5.2f}")

    monotone = curves[ParamKind.CAPACITY] == sorted(curves[ParamKind.CAPACITY])
    print(f"\n  Capacity degrades monotonically ({monotone}) and tracks the DC reference:")
    print("  its signature is a slow SOC ramp, which is exactly what a difference destroys.")
    print("  R0 and hA are NON-monotone. They worsen out to rho ~ 0.7, then recover, and at")
    print(
        f"  rho = 0.99 they are {1 / degrade(ParamKind.R0):.1f}x and "
        f"{1 / degrade(ParamKind.HA):.1f}x TIGHTER than under white noise."
    )
    print("\n  Not a paradox: R0's signature lives in the current pulse edges, and")
    print("  differencing destroys the noise faster than it destroys an edge. Pulsed")
    print("  excitation is lock-in detection. It buys immunity to 1/f noise, not merely")
    print("  precision -- and the duty cycle chooses which faults survive the noise.")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for kind, colour in (
        (ParamKind.R0, "#1565c0"),
        (ParamKind.CAPACITY, "#c62828"),
        (ParamKind.HA, "#2e7d32"),
    ):
        base = curves[kind][0]
        ax.plot(rhos, [v / base for v in curves[kind]], marker="o", color=colour, label=kind.value)
    ax.plot(
        rhos,
        np.sqrt((1 + rhos) / (1 - rhos)),
        ls="--",
        color="#616161",
        label=r"DC signature: $\sqrt{(1+\rho)/(1-\rho)}$",
    )
    ax.axhline(1.0, color="#37474f", lw=0.8, ls=":")
    ax.set(
        xlabel=r"AR(1) noise correlation $\rho$",
        ylabel="CRLB degradation vs white noise (x)",
        yscale="log",
        title="Correlated noise does not uniformly hurt. It reallocates.",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "noise_robustness.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ part 3
    rule("3. A thermocouple stops being a thermocouple")
    print("  A cooling fault's thermal signature evolves on the pack time constant (~200 s),")
    print("  which against a 1 Hz sampler is essentially DC. The thermocouple is therefore")
    print("  precisely the channel a first difference destroys.\n")

    temp_only = SensorTopology(topology.n_cells, voltage_cells=(), temp_cells=topology.temp_cells)
    ha_sensed = index_of(SENSED_CELL, ParamKind.HA)

    print(f"  {'rho':>6s} {'thermocouple share of hA[' + str(SENSED_CELL) + '] information':>44s}")
    for rho in (0.0, 0.5, 0.9, 0.99):
        noise = NoiseModel(voltage_rho=rho, temp_rho=rho)
        total = fisher_information(sens, topology, noise)[ha_sensed, ha_sensed]
        thermal = fisher_information(sens, temp_only, noise)[ha_sensed, ha_sensed]
        print(f"  {rho:6.2f} {100 * thermal / total:43.1f}%")

    print("\n  At rho = 0 the thermocouple carries ~90% of the cooling information. At")
    print("  rho = 0.99 it carries ~1%, and hA is read entirely through R0(T)'s leak into")
    print("  the VOLTAGE channel -- which rides the pulses, and is amplified.")

    print("\n  Consequence -- does 'instrumented beats uninstrumented' hold?")
    print(
        f"  {'rho':>6s} {'hA[' + str(SENSED_CELL) + '] (has TC)':>18s} "
        f"{'hA[' + str(BLIND_CELL) + '] (none)':>18s}   ordering"
    )
    for rho in (0.0, 0.9, 0.99):
        noise = NoiseModel(voltage_rho=rho, temp_rho=rho)
        std = np.sqrt(crlb(fisher_information(sens, topology, noise, specs=specs)))
        a = 0.40 / std[ha_sensed]
        b = 0.40 / std[index_of(BLIND_CELL, ParamKind.HA)]
        print(f"  {rho:6.2f} {a:17.2f}s {b:17.2f}s   {'holds' if a > b else '*** INVERTED ***'}")

    print("\n  Once the thermocouple contributes nothing, cell position decides instead")
    print("  (a corner cell has fewer conduction paths, so a bigger dT). The sensor-topology")
    print("  story we told at rho = 0 does not merely weaken. It inverts.")

    # ------------------------------------------------------------------ part 4
    rule("4. Which README headline verdicts survive? (rho = 0.9, current bias carried)")
    hypotheses = [
        (ParameterSpec(5, ParamKind.R0), 0.20),
        (ParameterSpec(17, ParamKind.CAPACITY), 0.05),
        (ParameterSpec(SENSED_CELL, ParamKind.HA), 0.40),
        (ParameterSpec(BLIND_CELL, ParamKind.HA), 0.40),
    ]
    std_by_rho = {
        rho: np.sqrt(
            crlb(
                fisher_information(
                    sens, topology, NoiseModel(voltage_rho=rho, temp_rho=rho), specs=specs
                )
            )
        )
        for rho in (0.0, 0.9, 0.99)
    }

    print(
        f"  {'hypothesis':<16s} {'TC':>3s} {'white':>9s} {'rho=0.9':>9s} {'rho=0.99':>9s}"
        f" {'white':>16s} {'rho=0.9':>16s}"
    )
    flipped: list[str] = []
    for target, magnitude in hypotheses:
        i = specs.index(target)
        w, c, c99 = (magnitude / std_by_rho[r][i] for r in (0.0, 0.9, 0.99))
        changed = verdict(c) != verdict(w)
        if changed:
            flipped.append(f"{target.label()}: {verdict(w)} ({w:.2f}s) -> {verdict(c)} ({c:.2f}s)")
        tc = "TC" if target.cell in topology.temp_cells else "--"
        print(
            f"  {target.label():<16s} {tc:>3s} {w:8.2f}s {c:8.2f}s {c99:8.2f}s "
            f"{verdict(w):>16s} {verdict(c):>16s}{'  <-- FLIPPED' if changed else ''}"
        )

    print(f"\n  {len(flipped)} of {len(hypotheses)} verdicts change at rho = 0.9.")
    for f in flipped:
        print(f"    {f}")

    print("\n  What survives, robustly:")
    print("    * resistance faults are identifiable everywhere, at every rho tested")
    print("    * cooling on an uninstrumented cell is never diagnosable")
    print("    * capacity stays diagnosable at rho=0.9, though its margin collapses ~4x")
    print("\n  What does NOT survive:")
    print("    * 'cooling is identifiable exactly where the thermocouples are'. At rho=0.9")
    print("      the instrumented cell is refused too, so cooling is identifiable NOWHERE;")
    print("      at rho=0.99 the instrumented/uninstrumented ordering inverts outright.")
    print("    * the assumption that correlated noise can only hurt. It helps R0 and hA.")
    print("\n  We do not know rho for any real pack. We know rho = 0 is the most optimistic")
    print("  choice available, and that this repo made it silently until this script existed.")

    print(f"\n  figure: {(FIGURES / 'noise_robustness.png').relative_to(FIGURES.parents[1])}")


if __name__ == "__main__":
    main()
