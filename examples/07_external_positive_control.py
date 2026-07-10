"""When the external cell really is broken, does AstraCell find it?

Run:  python examples/07_external_positive_control.py   (~40 s, needs `pip install -e '.[pybamm]'`)

v0.3 fitted a first-order ECM to a *healthy* PyBaMM cell and watched it report a 67.6% capacity
loss at 466 sigma. The bias gate refused, harmful overclaim went 100% -> 0%, and that was the
result. It was also a negative control, and a negative control cannot distinguish a system that
declines wisely from one that declines always.

v0.3's external gate was the second kind, and did not know it. ``prepare_external`` projected the
very trace it was about to fit and called the projection "structural bias" -- but a linear fit to
a residual *is* that projection, so ``b == E[delta_hat]``, ``SNR_total == 1``, and
``REFUSE_MODEL_BIAS`` came back for any data at all. Feed it a sine wave: it refuses. Feed it a
shattered cell: it refuses. This example is how that was found.

Five acts:

1. **The degenerate gate.** Demonstrated on a residual made of sine waves, no PyBaMM needed.
2. **A real fault, injected into PyBaMM.** A contact resistance -- a corroded tab weld. The
   differential voltage is ``-dR * I`` to machine precision, which is *exactly* the ECM's R0
   sensitivity. The observer can express this fault. Whether it finds it is another matter.
3. **Raw vs paired.** With a healthy baseline the phantom subtracts off exactly, because the
   projection is linear. The corrected estimate recovers the fault; the raw one does not, and at
   large faults it diagnoses confidently at the wrong magnitude.
4. **The sweep.** Where diagnosis begins, where it should not, and the transition between.
5. **The confounder.** A cathode diffusivity fault: real, physical, and invisible to a one-RC
   ECM, which reads it as a 119% capacity loss. AstraCell must refuse, and does -- by 1%.

The positive control's fault is, by construction, exactly in the observer's model span. That is
what a positive control is for. It establishes that the pipeline can recover a fault the model can
express; it establishes nothing whatever about faults the model cannot. Act 5 is the other half,
and it is the one where the gate has no margin. See docs/POSITIVE_CONTROL.md.
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
from astracell.cell.ocv import ocv_from_table
from astracell.observability.decision import VerdictKind
from astracell.plant import (
    HEALTHY,
    PYBAMM_AVAILABLE,
    contact_resistance,
    pybamm_pseudo_ocv,
    simulate_pybamm_cell,
    slow_cathode,
)

FIGURES = Path(__file__).resolve().parents[1] / "reports" / "figures"
SEED = 0
SOC0 = 0.9
DT_S = 1.0
N_TIME = 600
CAPACITY_AH = 5.0
MODEL = "SPMe"
N_TRIALS = 400

#: The observer's nominal series resistance. Faults are injected in ohms -- the electrochemical
#: cell has no parameter called "R0" -- and converted to the observer's relative units here.
OBSERVER_R0_OHM = 0.025

#: The primary fault, swept. Fractions of the observer's nominal R0; 0.02 is 0.5 mOhm.
SWEEP = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.20, 1.00)
STRONG, WEAK, HUGE = 0.20, 0.002, 1.00
CATHODE_SCALE = 0.3

VERDICT_STYLE = {
    VerdictKind.DIAGNOSE: ("#2e7d32", "diagnose"),
    VerdictKind.WEAK_EVIDENCE: ("#f9a825", "weak"),
    VerdictKind.REFUSE_UNOBSERVABLE: ("#9e9e9e", "refuse: unobservable"),
    VerdictKind.REFUSE_CONFOUNDED: ("#6a1b9a", "refuse: confounded"),
    VerdictKind.REFUSE_MODEL_BIAS: ("#c62828", "refuse: model bias"),
}


def rule(title: str) -> None:
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")


def current() -> np.ndarray:
    return pulse_profile(CAPACITY_AH, mean_c_rate=0.5, pulse_c_rate=1.0, n_time=N_TIME)


# --------------------------------------------------------------------------------------- act 1


def act1_degenerate_gate() -> None:
    rule("1. Why v0.3's external gate could never have found a fault (no PyBaMM needed)")
    from astracell.cell.ocv import NMC_LIKE

    observer = build_external_observer(NMC_LIKE, CAPACITY_AH)
    scenario = external_scenario(name="d", observer=observer, current_a=current(), soc0=SOC0)
    nominal = observer_voltage(scenario)
    t = np.arange(N_TIME)

    print("  `parameter_bias` of a residual IS the linear fit to that residual. Feed the v0.3 gate")
    print("  the same trace it is about to fit and it sets b = E[delta_hat]. Then, exactly,")
    print("\n      SNR_total = |b + noise| / sqrt(sigma^2 + b^2)     and     E[SNR_total^2] = 1")
    print("\n  for ANY residual. The statistic carries no information about the data at all.\n")
    print(f"  {'residual':28s} {'bias':>12s} {'mean':>7s} {'RMS':>7s} {'max':>7s}  refusals")
    for label, residual in (
        ("20 mV of sine wave", 0.02 * np.sin(t / 17.0)),
        ("a 1 V linear ramp (absurd)", np.linspace(0.0, 1.0, N_TIME)),
        ("a 1 uV whisper", 1e-6 * np.cos(t / 3.0)),
    ):
        prepared = prepare_external(scenario, nominal + residual)
        result = run_trials(scenario, 300, seed=SEED, estimator="linear", prepared=prepared)
        refused = result.verdicts.count(VerdictKind.REFUSE_MODEL_BIAS)
        rms = float(np.sqrt(np.mean(result.snr_bias**2)))
        print(
            f"  {label:30s} {prepared.target_bias:+12.2%} {result.snr_bias.mean():7.4f} "
            f"{rms:7.4f} {result.snr_bias.max():7.4f}  {refused:3d}/300"
        )

    print(
        "\n  Not a battery, not a fault, not physics -- and the 1 V ramp implies a 1718% capacity"
    )
    print("  deviation. Both are refused, both at 1 sigma, for the same reason and with the same")
    print(
        "  certainty. Whenever the bias exceeds the noise -- the only regime in which a bias gate"
    )
    print("  has a job -- the statistic concentrates on 1 and REFUSE_MODEL_BIAS is guaranteed.")
    print("\n  The 1 uV row is the degenerate other end: b -> 0, so SNR_total collapses to")
    print("  |noise|/sigma, refuses nothing, and merely re-tests the noise. Uninformative in both")
    print("  directions -- which is what 'carries no information about the data' means.")
    print(
        "\n  v0.3 was right about a healthy cell for the wrong reason. Its 100% REFUSE_MODEL_BIAS"
    )
    print("  was not evidence about the cell; it was arithmetic about the estimator.")


# --------------------------------------------------------------------------------------- setup


def build_observer():  # type: ignore[no-untyped-def]
    soc, ocv = pybamm_pseudo_ocv(model=MODEL, n_points=200)
    curve = ocv_from_table(soc, ocv, name=f"pybamm_{MODEL.lower()}", chemistry=f"{MODEL}/Chen2020")
    return build_external_observer(curve, capacity_ah=CAPACITY_AH)


def evidence_for(observer, healthy_v, fault, magnitude, target):  # type: ignore[no-untyped-def]
    faulty_v = simulate_pybamm_cell(current(), DT_S, model=MODEL, soc0=SOC0, fault=fault).voltage_v
    scenario = external_scenario(
        name=fault.name,
        observer=observer,
        current_a=current(),
        soc0=SOC0,
        target_index=target,
        fault_magnitude=magnitude,
    )
    return build_evidence(scenario, healthy_v, faulty_v), faulty_v


# --------------------------------------------------------------------------------------- act 2


def act2_the_injected_fault(observer, healthy_v):  # type: ignore[no-untyped-def]
    rule("2. A real fault in the external plant: +0.5 mOhm of contact resistance (20% of R0)")
    delta_r = STRONG * OBSERVER_R0_OHM
    evidence, faulty_v = evidence_for(
        observer, healthy_v, contact_resistance(delta_r), STRONG, R0_TARGET
    )
    confounder, cathode_v = evidence_for(
        observer, healthy_v, slow_cathode(CATHODE_SCALE), 0.0, R0_TARGET
    )

    diff_r0 = faulty_v - healthy_v
    diff_cat = cathode_v - healthy_v
    ohmic = -delta_r * current()
    print(f"  injected contact resistance   {1e3 * delta_r:.2f} mOhm = {STRONG:.0%} of observer R0")
    print(
        f"  max |(g_f - g_h) + dR*I|      {np.abs(diff_r0 - ohmic).max():.2e} V   <- exactly ohmic"
    )
    print(f"  differential lack of fit      {evidence.misfit_paired:.2e}  <- ECM says this exactly")
    print()
    print(f"  injected cathode diffusivity  x{CATHODE_SCALE}")
    print(f"  differential RMS              {1e3 * np.sqrt(np.mean(diff_cat**2)):.2f} mV")
    print(
        f"  differential lack of fit      {confounder.misfit_paired:.1f}  <- no parameters say this"
    )

    t = np.arange(N_TIME) * DT_S
    fig, (top, mid, bot) = plt.subplots(
        3, 1, figsize=(9.2, 7.6), sharex=True, height_ratios=[2, 1, 1]
    )
    top.plot(t, healthy_v, color="#2e7d32", lw=1.1, label="healthy PyBaMM SPMe")
    top.plot(
        t, faulty_v, color="#1565c0", lw=1.0, label=f"+{1e3 * delta_r:.2f} mOhm contact resistance"
    )
    top.plot(
        t,
        cathode_v,
        color="#c62828",
        lw=1.0,
        ls="--",
        label=f"cathode diffusivity x{CATHODE_SCALE}",
    )
    top.set(
        ylabel="terminal voltage [V]", title="Two real faults in a plant AstraCell did not write"
    )
    top.legend(fontsize=8)
    top.grid(alpha=0.3)

    mid.plot(t, 1e3 * diff_r0, color="#1565c0", lw=1.0, label="measured differential")
    mid.plot(t, 1e3 * ohmic, color="#000000", lw=0.8, ls=":", label=r"$-\Delta R \cdot I(t)$")
    mid.set(ylabel="[mV]", title="Contact resistance: the differential IS the ECM's R0 direction")
    mid.legend(fontsize=8)
    mid.grid(alpha=0.3)

    bot.plot(t, 1e3 * diff_cat, color="#c62828", lw=1.0)
    bot.axhline(0.0, color="#455a64", lw=0.8, ls=":")
    bot.set(
        xlabel="time [s]",
        ylabel="[mV]",
        title="Cathode diffusivity: a shape no first-order ECM can produce",
    )
    bot.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "positive_control_traces.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return evidence, confounder


# --------------------------------------------------------------------------------------- act 3


def act3_raw_vs_paired(observer, healthy_v, evidence):  # type: ignore[no-untyped-def]
    rule("3. Raw vs baseline-corrected: the phantom subtracts off, exactly")
    healthy_ev, _ = evidence_for(observer, healthy_v, HEALTHY, 0.0, CAPACITY_TARGET)
    huge, huge_v = evidence_for(
        observer, healthy_v, contact_resistance(HUGE * OBSERVER_R0_OHM), HUGE, R0_TARGET
    )

    # The sharpest statement of act 1, now on a genuinely shattered cell: doubled series resistance,
    # 25 mOhm of it, and v0.3's baseline-free gate declines to see it for the same reason it
    # declined the sine wave.
    v03_scenario = external_scenario(
        name="v0.3-gate", observer=observer, current_a=current(), soc0=SOC0, target_index=R0_TARGET
    )
    v03 = prepare_external(v03_scenario, huge_v)  # no baseline: exactly the v0.3 call
    v03_result = run_trials(v03_scenario, 300, seed=SEED, estimator="linear", prepared=v03)
    refused = v03_result.verdicts.count(VerdictKind.REFUSE_MODEL_BIAS)
    print(
        f"  v0.3's gate on a DOUBLED series resistance (+{1e3 * HUGE * OBSERVER_R0_OHM:.0f} mOhm):"
    )
    print(f"    bias == estimate       {v03.target_bias:+.2%}")
    print(f"    SNR_total              {np.sqrt(np.mean(v03_result.snr_bias**2)):.4f} sigma")
    print(
        f"    verdicts               {refused}/300 REFUSE_MODEL_BIAS   <- it calls the fault 'bias'"
    )
    print()

    print(f"  {'quantity':38s} {'value':>14s}")
    print(f"  {'healthy phantom on capacity':38s} {healthy_ev.phantom:+14.2%}   <- v0.3's headline")
    print(f"  {'healthy phantom on R0':38s} {evidence.phantom:+14.2%}")
    print(f"  {'raw estimate (R0), 20% fault':38s} {evidence.raw_estimate:+14.4%}")
    print(f"  {'  minus the phantom':38s} {evidence.raw_estimate - evidence.phantom:+14.4%}")
    print(
        f"  {'paired estimate (R0), 20% fault':38s} {evidence.paired_estimate:+14.4%}  <- identical"
    )
    print(
        f"  {'residual of the identity':38s} "
        f"{abs(evidence.raw_estimate - evidence.phantom - evidence.paired_estimate):14.2e}"
    )
    print(
        "\n  The projection is linear, so subtracting the healthy baseline's bias and fitting the"
    )
    print("  paired difference are the SAME estimator. There was never a choice between them. The")
    print(
        f"  price is honest noise: two traces, two draws, so sigma widens by sqrt(2) to "
        f"{evidence.paired.target_sigma:.3%}."
    )

    results = {}
    for label, ctx in (("raw", evidence.raw), ("paired", evidence.paired)):
        results[label] = run_trials(
            ctx.scenario, N_TRIALS, seed=SEED, estimator="linear", prepared=ctx
        )
    huge_raw = run_trials(
        huge.raw.scenario, N_TRIALS, seed=SEED, estimator="linear", prepared=huge.raw
    )

    print(f"\n  {'path':28s} {'diagnose':>9s} {'true pos':>9s} {'overclaim':>10s} {'cov 95%':>8s}")
    for label in ("raw", "paired"):
        m = detection_metrics(results[label])
        print(
            f"  {label + ' (20% fault)':28s} {m.diagnosis_rate:9.2f} {m.true_positive_rate:9.2f} "
            f"{m.harmful_overclaim_rate:10.2f} {m.coverage:8.2f}"
        )
    m = detection_metrics(huge_raw)
    print(
        f"  {'raw (100% fault)':28s} {m.diagnosis_rate:9.2f} {m.true_positive_rate:9.2f} "
        f"{m.harmful_overclaim_rate:10.2f} {m.coverage:8.2f}"
    )
    print(
        "\n  At a 100% fault the raw path is loud enough to clear its own phantom's gate. It then"
    )
    print(
        f"  DIAGNOSES in every trial, reporting {huge.raw_estimate:+.1%} against a truth of "
        f"{HUGE:+.0%} --"
    )
    print(
        f"  an {100 * abs(huge.raw_estimate - HUGE):.1f}-point miss inside a "
        f"{2 * 1.96 * huge.raw.target_sigma:.2%}-wide interval. Right about the fault, wrong"
    )
    print("  about the fault. Every one of those diagnoses is a harmful overclaim.")

    # Re-score v0.3's headline with the honest gate, and price how much of the bias it sees.
    b_lof = healthy_ev.raw_lack_of_fit
    sigma = healthy_ev.raw.target_sigma
    truth_is_bias = abs(healthy_ev.raw_estimate)  # the cell is healthy: the estimate IS the bias
    snr_total = truth_is_bias / float(np.hypot(sigma, b_lof))
    print(
        "\n  Re-scoring v0.3's healthy cell with the honest gate (no baseline, lack-of-fit only):"
    )
    print(f"    capacity estimate      {healthy_ev.raw_estimate:+.2%}   <- and it is ALL bias")
    print(f"    lack-of-fit bias       {b_lof:+.2%}   <- what the screen sees")
    print(f"    SNR_total              {snr_total:.2f} sigma   <- WEAK_EVIDENCE, not REFUSE")
    print(f"    fraction of bias seen  {b_lof / truth_is_bias:.0%}")
    print("\n  So the screen under-warns by 3.2x on the one case whose truth we know, and v0.3's")
    print("  '100% REFUSE_MODEL_BIAS' becomes 'WEAK'. Harmful overclaim is still 0% -- WEAK is not")
    print("  DIAGNOSE -- so the headline is corrected, not retracted. But the margin was thinner")
    print("  than v0.3 reported, and a phantom three times smaller would have slipped through.")

    # Every distribution is narrower than a bin on this scale -- that is the point, and the reason
    # the phantom is dangerous: these estimates are exquisitely precise and two of them are wrong.
    healthy_hat = run_trials(
        healthy_ev.raw.scenario, N_TRIALS, seed=SEED, prepared=healthy_ev.raw
    ).delta_hat
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    bins = np.linspace(-80, 30, 220)
    for values, colour, label in (
        (healthy_hat, "#c62828", "healthy cell, raw fit (capacity)"),
        (results["raw"].delta_hat, "#f9a825", "+20% R0 fault, raw fit (R0)"),
        (results["paired"].delta_hat, "#2e7d32", "+20% R0 fault, baseline-corrected (R0)"),
    ):
        ax.hist(100 * values, bins=bins, color=colour, alpha=0.9, label=label)

    ax.axvline(0.0, color="#212121", lw=1.4, ls="--", zorder=5, label="truth: healthy cell (0%)")
    ax.axvline(
        100 * STRONG,
        color="#212121",
        lw=1.4,
        ls=":",
        zorder=5,
        label=f"truth: +{100 * STRONG:.0f}% R0",
    )
    top = ax.get_ylim()[1]
    ax.annotate(
        "",
        xy=(100 * STRONG, 0.62 * top),
        xytext=(100 * evidence.raw_estimate, 0.62 * top),
        arrowprops=dict(arrowstyle="<->", color="#212121", lw=1.1),
    )
    ax.text(
        0.5 * 100 * (STRONG + evidence.raw_estimate),
        0.66 * top,
        f"{100 * (STRONG - evidence.raw_estimate):.1f} pts\n= the R0 phantom",
        ha="center",
        fontsize=7.5,
        color="#212121",
    )
    ax.text(
        100 * healthy_ev.raw_estimate + 2.5,
        0.70 * top,
        f"a phantom fault:\nthe cell is healthy\n({healthy_ev.raw_estimate:+.1%})",
        ha="left",
        fontsize=7.5,
        color="#c62828",
    )
    ax.set(
        xlabel="estimated deviation [%]",
        ylabel=f"trials (of {N_TRIALS})",
        title="The phantom displaces every raw estimate. A healthy baseline subtracts it exactly.",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "positive_control_estimates.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return healthy_ev, huge


# --------------------------------------------------------------------------------------- act 4


def act4_sweep(observer, healthy_v):  # type: ignore[no-untyped-def]
    rule("4. Fault magnitude sweep: where AstraCell diagnoses, weakens, and refuses")
    print(
        f"  {'dR [mOhm]':>10s} {'dR/R0':>8s} {'paired est':>11s} {'SNR var':>9s} {'b_lof':>9s} "
        f"{'SNR tot':>8s} {'diagnose':>9s} {'true pos':>9s} {'weak':>6s} {'refuse':>7s}"
    )

    fractions = {kind: np.zeros(len(SWEEP)) for kind in VerdictKind}
    tpr, snr_var, snr_tot, sigma_mohm = [], [], [], 0.0
    for i, frac in enumerate(SWEEP):
        ev, _ = evidence_for(
            observer, healthy_v, contact_resistance(frac * OBSERVER_R0_OHM), frac, R0_TARGET
        )
        result = run_trials(
            ev.paired.scenario, N_TRIALS, seed=SEED, estimator="linear", prepared=ev.paired
        )
        m = detection_metrics(result)
        for kind, share in verdict_distribution(result).items():
            fractions[kind][i] = share

        sv = abs(ev.paired_estimate) / ev.paired.target_sigma
        st = abs(ev.paired_estimate) / float(np.hypot(ev.paired.target_sigma, ev.lack_of_fit))
        sigma_mohm = 1e3 * ev.paired.target_sigma * OBSERVER_R0_OHM
        tpr.append(m.true_positive_rate)
        snr_var.append(sv)
        snr_tot.append(st)
        print(
            f"  {1e3 * frac * OBSERVER_R0_OHM:10.3f} {frac:8.2%} {ev.paired_estimate:11.4%} "
            f"{sv:9.1f} {ev.lack_of_fit:9.2e} {st:8.1f} {m.diagnosis_rate:9.2f} "
            f"{m.true_positive_rate:9.2f} {m.weak_rate:6.2f} {m.refusal_rate:7.2f}"
        )

    print("\n  The transition sits exactly where the Cramer-Rao bound puts it. sigma on dR is")
    print(
        f"  {sigma_mohm:.3f} mOhm, so 5 sigma is {5 * sigma_mohm:.3f} mOhm: diagnosis begins there."
    )
    print("  Nothing about the 20 mV phantom enters: the differential cancelled it. AstraCell is")
    print("  resolving two ten-thousandths of the series resistance because that is what 600")
    print("  samples of a 1 mV sensor buy against a 2.5 A current, and no more.")
    print("\n  `b_lof` is zero to floating-point dust throughout -- the ECM reproduces a series")
    print("  resistance exactly -- so the credibility gate has nothing to object to and correctly")
    print("  stays out of the way. That is a fact about this fault, not a property of the gate.")
    print(
        f"\n  True-positive rate saturates at ~{max(tpr):.2f}, not 1.00, and that is correct: a 95%"
    )
    print("  interval misses its truth 5% of the time by construction, so 5% of confident, correct")
    print("  diagnoses are counted 'harmful overclaims'. Overclaim floors at 1 - coverage_level.")

    x = 1e3 * np.array(SWEEP) * OBSERVER_R0_OHM
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bottom = np.zeros(len(SWEEP))
    for kind in VerdictKind:
        colour, name = VERDICT_STYLE[kind]
        if fractions[kind].max() == 0.0:
            continue
        ax.fill_between(x, bottom, bottom + fractions[kind], color=colour, alpha=0.85, label=name)
        bottom = bottom + fractions[kind]
    ax.plot(x, tpr, color="#000000", lw=1.6, marker="o", ms=3.5, label="true-positive rate")
    ax.axhline(0.95, color="#000000", lw=0.7, ls=":")
    ax.set(
        xscale="log",
        xlabel=r"injected contact resistance $\Delta R$ [m$\Omega$]",
        ylabel="fraction of trials",
        ylim=(0, 1),
        title="Baseline-corrected verdicts against a real, injected, external fault",
    )
    ax.margins(x=0)
    ax.legend(fontsize=7, loc="center left", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(FIGURES / "positive_control_sweep.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return x, np.array(snr_var), np.array(snr_tot)


# --------------------------------------------------------------------------------------- act 5


def _snr_pair(evidence) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    """``(variance-only SNR, bias-aware SNR)`` of the paired estimate on its target."""
    magnitude = abs(evidence.paired_estimate)
    sigma = evidence.paired.target_sigma
    return magnitude / sigma, magnitude / float(np.hypot(sigma, evidence.lack_of_fit))


def plot_snr(sweep, scenarios) -> None:  # type: ignore[no-untyped-def]
    """Where the gate stands aside, and where it bites. The two halves of the same statistic.

    Left: on a fault the observer can express, the lack of fit is zero, so the bias-aware SNR is
    the variance-only SNR and the credibility gate does nothing. Right: on the confounder, the same
    gate cuts 158 sigma of confidence to 0.54 -- and on capacity, to 1.98: a refusal by one percent.
    """
    x, snr_var, snr_tot = sweep
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.6), width_ratios=[1.15, 1])

    left.loglog(x, snr_var, "o-", color="#1565c0", lw=1.4, label="variance-only SNR")
    left.loglog(x, snr_tot, "s--", color="#2e7d32", lw=1.4, ms=4, label="bias-aware SNR")
    left.axhline(5.0, color="#2e7d32", ls=":", lw=0.9)
    left.axhline(2.0, color="#f9a825", ls=":", lw=0.9)
    left.text(x[0], 5.6, "5 sigma: diagnose", fontsize=7, color="#2e7d32")
    left.text(x[0], 1.45, "2 sigma: refuse below", fontsize=7, color="#f9a825")
    left.set(
        xlabel=r"injected contact resistance $\Delta R$ [m$\Omega$]",
        ylabel="detection SNR [sigma]",
        title="A fault the observer CAN express:\nthe two SNRs coincide, the gate stands aside",
    )
    left.legend(fontsize=8, loc="upper left")
    left.grid(alpha=0.3, which="both")

    # The healthy scenario's differential is identically zero, so both its SNRs are exactly zero and
    # a log axis cannot draw them. That is not a small bar; it is the absence of a change to detect.
    # Its false-positive rate is the number that matters, and positive_control_rates.png reports it.
    drawable = [(name, pair) for name, pair, _ in scenarios if pair[0] > 1e-6]
    labels = [name for name, _ in drawable]
    var = [pair[0] for _, pair in drawable]
    tot = [pair[1] for _, pair in drawable]
    idx = np.arange(len(labels))
    right.bar(idx - 0.19, var, 0.36, color="#1565c0", label="variance-only SNR")
    right.bar(idx + 0.19, tot, 0.36, color="#2e7d32", label="bias-aware SNR")
    right.axhline(5.0, color="#2e7d32", ls=":", lw=0.9)
    right.axhline(2.0, color="#f9a825", ls=":", lw=0.9)
    for i, (v, t) in enumerate(zip(var, tot, strict=True)):
        right.text(i - 0.19, v * 1.15, f"{v:.1f}", ha="center", fontsize=7, color="#1565c0")
        right.text(i + 0.19, t * 1.15, f"{t:.2f}", ha="center", fontsize=7, color="#2e7d32")
    right.set(
        yscale="log",
        ylim=(0.2, 4000),
        ylabel="detection SNR [sigma]",
        title="A change it CANNOT express:\nthe gate cuts 579 sigma to 1.98, and refuses",
    )
    right.set_xticks(idx, labels, fontsize=7.5)
    right.legend(fontsize=8, loc="upper left")
    right.grid(alpha=0.3, axis="y", which="both")

    fig.tight_layout()
    fig.savefig(FIGURES / "positive_control_snr.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def act5_confounder(observer, healthy_v, confounder):  # type: ignore[no-untyped-def]
    rule("5. The confounder: a real degradation the observer cannot express")
    capacity, _ = evidence_for(
        observer, healthy_v, slow_cathode(CATHODE_SCALE), 0.0, CAPACITY_TARGET
    )

    print(f"  injected: cathode particle diffusivity x{CATHODE_SCALE}. No lithium lost, no active")
    print("  material lost. True (R0, capacity) deviation: (0%, 0%). Only transport changed.\n")
    print(
        f"  {'hypothesis':14s} {'paired est':>12s} {'sigma':>9s} {'SNR var':>9s} {'b_lof':>9s} "
        f"{'SNR tot':>9s}  verdict"
    )
    rows = []
    for label, ev in (("R0", confounder), ("capacity", capacity)):
        sv = abs(ev.paired_estimate) / ev.paired.target_sigma
        st = abs(ev.paired_estimate) / float(np.hypot(ev.paired.target_sigma, ev.lack_of_fit))
        result = run_trials(
            ev.paired.scenario, N_TRIALS, seed=SEED, estimator="linear", prepared=ev.paired
        )
        top = max(verdict_distribution(result).items(), key=lambda kv: kv[1])[0]
        rows.append((label, ev, sv, st, result))
        print(
            f"  {label:14s} {ev.paired_estimate:12.2%} {ev.paired.target_sigma:9.3%} {sv:9.1f} "
            f"{ev.lack_of_fit:9.2%} {st:9.2f}  {top.value.upper()}"
        )

    print(
        "\n  Variance alone would diagnose a 13% resistance fault at 158 sigma, and a 119% capacity"
    )
    print("  loss at 579 sigma, on a cell whose resistance and capacity are exactly nominal. The")
    print(
        "  lack-of-fit bias says: no setting of my parameters produces this differential. Refuse."
    )
    print(
        f"\n  And note the margin. The capacity hypothesis clears the 2-sigma line at "
        f"{rows[1][3]:.2f} sigma --"
    )
    print("  refused by 1%. The screen is doing real work and has nothing to spare. Shorten the")
    print("  window to 300 s and the very same hypothesis reaches 3.35 sigma and is merely WEAK.")

    healthy_ev, _ = evidence_for(observer, healthy_v, HEALTHY, 0.0, R0_TARGET)
    weak_ev, _ = evidence_for(
        observer, healthy_v, contact_resistance(WEAK * OBSERVER_R0_OHM), WEAK, R0_TARGET
    )
    strong_ev, _ = evidence_for(
        observer, healthy_v, contact_resistance(STRONG * OBSERVER_R0_OHM), STRONG, R0_TARGET
    )

    print()
    print(
        f"  {'scenario':32s} {'TPR':>6s} {'FPR':>6s} {'refuse':>7s} {'weak':>6s} {'overclaim':>9s}"
    )
    labels, tprs, fprs, weaks, refusals, snr_pairs = [], [], [], [], [], []
    for label, short, ev in (
        ("healthy (null)", "healthy", healthy_ev),
        (f"weak fault (+{1e3 * WEAK * OBSERVER_R0_OHM:.2f} mOhm)", "weak fault", weak_ev),
        (f"real fault (+{1e3 * STRONG * OBSERVER_R0_OHM:.2f} mOhm)", "real fault", strong_ev),
        (f"confounded (D x{CATHODE_SCALE}, R0)", "confounded\n(R0)", confounder),
        (f"confounded (D x{CATHODE_SCALE}, capacity)", "confounded\n(capacity)", capacity),
    ):
        m = detection_metrics(
            run_trials(ev.paired.scenario, N_TRIALS, seed=SEED, prepared=ev.paired)
        )
        labels.append(short)
        tprs.append(m.true_positive_rate)
        fprs.append(m.false_positive_rate)
        weaks.append(m.weak_rate)
        refusals.append(m.refusal_rate)
        snr_pairs.append((short, _snr_pair(ev), ev))
        print(
            f"  {label:32s} {m.true_positive_rate:6.2f} {m.false_positive_rate:6.2f} "
            f"{m.refusal_rate:8.2f} {m.weak_rate:6.2f} {m.harmful_overclaim_rate:10.2f}"
        )

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    idx = np.arange(len(labels))
    series = (
        (tprs, -0.30, "#2e7d32", "true positive"),
        (fprs, -0.10, "#c62828", "false positive"),
        (weaks, 0.10, "#f9a825", "weak evidence"),
        (refusals, 0.30, "#9e9e9e", "refusal"),
    )
    for values, offset, colour, label in series:
        ax.bar(idx + offset, values, 0.19, color=colour, label=label)
        # A rate of exactly zero must read as a *measurement*, not as a missing bar. The two zeros
        # that matter -- no false positive on either null -- are the point of the whole figure.
        for i, value in enumerate(values):
            ax.text(
                i + offset,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                fontsize=6.5,
                color=colour if value > 0 else "#212121",
                fontweight="bold" if value == 0.0 and colour == "#c62828" else "normal",
            )
    ax.set(
        ylabel="fraction of trials",
        ylim=(0, 1.12),
        title="Baseline-corrected: finds the fault it can express, refuses the one it cannot.\n"
        "Zero false positives on both nulls: the healthy cell, and the change it cannot name.",
    )
    ax.set_xticks(idx, labels, fontsize=7.5)
    ax.legend(fontsize=8, ncols=4, loc="upper center")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "positive_control_rates.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return snr_pairs


def summary(healthy_ev, evidence, confounder) -> None:  # type: ignore[no-untyped-def]
    rule("Summary: the external positive control")
    confounder_snr = abs(confounder.paired_estimate) / float(
        np.hypot(confounder.paired.target_sigma, confounder.lack_of_fit)
    )
    print(f"  plant                    PyBaMM {MODEL} (Chen2020, {CAPACITY_AH:.0f} Ah), one cell")
    print(
        "  observer                 first-order Thevenin ECM (R0, R1C1), shared healthy pseudo-OCV"
    )
    print("  primary fault            contact resistance, exactly in the observer's model span")
    print(f"  confounder               cathode diffusivity x{CATHODE_SCALE}, exactly outside it")
    print()
    print(f"  healthy phantom          {healthy_ev.phantom:+.1%} capacity on a sound cell (v0.3)")
    print("  v0.3 external gate       REFUSE_MODEL_BIAS, unconditionally, for any data (act 1)")
    print(
        f"  paired estimate          {evidence.paired_estimate:+.4%} vs an injected {STRONG:+.0%}"
    )
    print("  transition               diagnosis begins at 0.10 mOhm; below 0.05 mOhm, refusal")
    print(f"  confounder, R0           refused at {confounder_snr:.2f} sigma")
    print("  confounder, capacity     refused at 1.98 sigma -- by one percent")
    print()
    print(
        "  v0.4 does not make AstraCell better at diagnosing. It makes the refusal mean something:"
    )
    print("  the same gate that declines a phantom now passes a real fault, and still declines a")
    print("  real change it cannot name. What it cost was a healthy baseline and sqrt(2) of sigma.")
    print("\n  Read section 5 of docs/POSITIVE_CONTROL.md before quoting any of this. The baseline")
    print(
        "  here is the identical simulation, which no workshop can supply. The lack-of-fit screen"
    )
    print("  bounds nothing: on the one case with known truth it captures 31% of the bias it warns")
    print(
        "  of. And the primary fault is in the model's span by construction -- that is what makes"
    )
    print("  it a control, and what makes it silent about every fault that is not.")


def main() -> int:
    if not PYBAMM_AVAILABLE:
        print("PyBaMM is not installed; example 07 needs it. Install: pip install -e '.[pybamm]'")
        print("Running act 1 only (it needs no external plant), then skipping.\n")
        act1_degenerate_gate()
        print("\nSkipping acts 2-5 (this is not a failure -- PyBaMM is an optional dependency).")
        return 0

    warnings.simplefilter("ignore")  # PyBaMM/casadi solve chatter, not ours
    FIGURES.mkdir(parents=True, exist_ok=True)

    act1_degenerate_gate()
    observer = build_observer()
    healthy_v = simulate_pybamm_cell(
        current(), DT_S, model=MODEL, soc0=SOC0, fault=HEALTHY
    ).voltage_v

    evidence, confounder = act2_the_injected_fault(observer, healthy_v)
    healthy_ev, _ = act3_raw_vs_paired(observer, healthy_v, evidence)
    sweep = act4_sweep(observer, healthy_v)
    scenarios = act5_confounder(observer, healthy_v, confounder)
    plot_snr(sweep, scenarios)
    summary(healthy_ev, evidence, confounder)
    print(f"\nFigures written to {FIGURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
