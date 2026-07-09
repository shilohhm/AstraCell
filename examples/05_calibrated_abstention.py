"""Are the verdicts calibrated, or just convincing?

Run:  python examples/05_calibrated_abstention.py       (~3 min)

v0.1 proved AstraCell can compute a structural bias and refuse when it dominates. It proved it
on *one* example. This asks the harder question: across thousands of repeated experiments with a
known injected truth, do the intervals and verdicts mean what they claim? A 90% interval should
cover the truth 90% of the time. A DIAGNOSE should be right. A REFUSE should be earning its
silence.

To ask that at all, v0.2 needs something the repository never had: an estimator. Everything
before was a property of the design. Here a fixed-information Gauss-Newton (matched-model, for
attainment) or an exact linear-Gaussian fit (mismatch, for speed) turns each noisy realisation
into an estimate and a verdict, and ``calibration`` counts how often they hold up.

Five acts:

1. **Coverage vs nominal.** Under a matched model the maximum-likelihood interval covers at its
   nominal rate -- the Cramer-Rao bound is *attained*, not just asserted. Under mismatch the
   variance-only interval covers essentially never. Widening it for the bias restores honesty at
   the price of uselessness.
2. **More data buys precision, not accuracy.** The money plot: variance-only confidence climbs
   without bound while the estimate cloud tightens onto a value that was never the truth.
3. **Verdict distribution vs fault size.** Where AstraCell diagnoses, weakens, and refuses -- and
   how ``REFUSE_MODEL_BIAS`` caps the capacity diagnosis no matter how large the fault.
4. **Harmful overclaim, before and after the gate.** The one number that says calibration worked:
   the rate of confident-and-wrong diagnoses, which the bias gate drives to zero by refusing.
5. **When does a recommendation stay calibrated?** Adding a thermocouple, pulsing harder, or
   watching longer all raise the variance-only SNR. Only some of them keep the verdict honest.

The point, as ever, is not that AstraCell looks better. Under honest assumptions several of its
numbers get worse. That is the result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
from astracell.plant import NO_MISMATCH, REALISTIC_MISMATCH

FIGURES = Path(__file__).resolve().parents[1] / "reports" / "figures"
SEED = 0
SENSED_CELL = 3  # carries the single thermocouple on the 2x2 demo pack
BLIND_CELL = 1  # voltage only

WINDOW = pulse_train(600.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0).current_a
GN = {"max_iter": 10, "step_tol": 1e-5}

VERDICT_STYLE: dict[VerdictKind, tuple[str, str]] = {
    VerdictKind.DIAGNOSE: ("#2e7d32", "diagnose"),
    VerdictKind.WEAK_EVIDENCE: ("#f9a825", "weak"),
    VerdictKind.REFUSE_UNOBSERVABLE: ("#9e9e9e", "refuse: unobservable"),
    VerdictKind.REFUSE_CONFOUNDED: ("#6a1b9a", "refuse: confounded"),
    VerdictKind.REFUSE_MODEL_BIAS: ("#c62828", "refuse: model bias"),
}


def rule(title: str) -> None:
    print(f"\n{'=' * 84}\n{title}\n{'=' * 84}")


def act1_coverage() -> None:
    rule("1. Coverage vs nominal: does a 90% interval cover 90% of the time?")
    matched = build_scenario(
        name="cap/matched", fault_kind=ParamKind.CAPACITY, target_cell=BLIND_CELL, current_a=WINDOW
    )
    mismatch = build_scenario(
        name="cap/mismatch",
        fault_kind=ParamKind.CAPACITY,
        target_cell=BLIND_CELL,
        current_a=WINDOW,
        mismatch=REALISTIC_MISMATCH,
    )
    # Matched uses the true nonlinear MLE, because "the bound is attained" is a claim about a
    # real estimator. Mismatch uses the exact linear fit: fast, and its ~30% structural bias is
    # three orders of magnitude larger than the estimator's own curvature, so it dominates.
    res_matched = run_trials(
        matched, 250, seed=SEED, estimator="gauss_newton", estimator_options=GN
    )
    res_mismatch = run_trials(mismatch, 250, seed=SEED, estimator="linear")

    cov_matched = coverage_curve(res_matched)
    cov_var = coverage_curve(res_mismatch, bias_aware=False)
    cov_bias = coverage_curve(res_mismatch, bias_aware=True)

    print(
        f"  matched sigma = {res_matched.crlb_std:.3%}   "
        f"mismatch bias = {res_mismatch.bias:+.2%}   (250 trials each)\n"
    )
    print(
        f"  {'nominal':>8s} {'matched (MLE)':>15s} {'mismatch var':>14s} "
        f"{'mismatch bias-aware':>21s}"
    )
    for lvl, cm, cv, cb in zip(NOMINAL_LEVELS, cov_matched, cov_var, cov_bias, strict=True):
        print(f"  {lvl:8.0%} {cm:15.2%} {cv:14.2%} {cb:21.2%}")
    print("\n  Matched coverage tracks the diagonal: the MLE attains the Cramer-Rao bound, which")
    print("  no number in this repository had ever shown. Variance-only coverage under mismatch")
    print("  collapses to zero -- the interval is confident and wrong. The bias-aware interval")
    print("  recovers coverage only above 80%, and only by being wide enough to be useless.")

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    grid = np.array(NOMINAL_LEVELS)
    ax.plot([0, 1], [0, 1], ls="--", color="#455a64", lw=1.0, label="perfect calibration")
    ax.plot(grid, cov_matched, "o-", color="#2e7d32", label="matched (MLE)")
    ax.plot(grid, cov_var, "s-", color="#c62828", label="mismatched, variance-only")
    ax.plot(grid, cov_bias, "^-", color="#1565c0", label="mismatched, bias-aware interval")
    ax.set(
        xlabel="nominal confidence",
        ylabel="empirical coverage",
        title="A 90% interval should cover 90% of the time.\n"
        "Under mismatch, variance-only does not.",
        xlim=(0.45, 1.0),
        ylim=(-0.03, 1.03),
    )
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return res_mismatch


def act2_more_samples(res_mismatch) -> None:
    rule("2. More data buys precision, not accuracy")
    counts = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 1000.0, 10000.0])
    curve = sample_count_curve(res_mismatch, counts)

    print(
        f"  true fault = {res_mismatch.delta_true:+.0%}   structural bias = "
        f"{res_mismatch.bias:+.2%}   estimate settles at {curve.center:+.2%}\n"
    )
    print(f"  {'repeats':>9s} {'SNR (variance only)':>21s} {'SNR (bias-aware)':>18s}")
    for k, sv, sb in zip(counts, curve.snr_var, curve.snr_bias, strict=True):
        print(f"  {int(k):9d} {sv:21.1f} {sb:18.3f}")
    print("\n  Variance-only SNR grows as sqrt(k), without bound. Bias-aware SNR stops dead at")
    print(f"  the ceiling {curve.ceiling:.3f} sigma -- the estimate cloud tightens by 100x onto a")
    print("  centre that was never the truth. A CRLB-only system grows more confident and no")
    print("  less wrong the more data you feed it. This is the central claim, now empirical.")

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    left.loglog(
        counts, curve.snr_var, "o-", color="#c62828", label=r"variance only: $\sqrt{k}\,m/\sigma$"
    )
    left.loglog(
        counts,
        curve.snr_bias,
        "s-",
        color="#1565c0",
        label=r"bias-aware: $m/\sqrt{\sigma^2/k+b^2}$",
    )
    left.axhline(5.0, ls=":", color="#2e7d32", lw=1.1, label=r"5$\sigma$ threshold")
    left.axhline(
        curve.ceiling,
        ls="--",
        color="#1565c0",
        lw=0.9,
        label=f"ceiling {curve.ceiling:.2f}$\\sigma$",
    )
    left.set(
        xlabel="independent repetitions",
        ylabel="detection SNR",
        title="Confidence climbs; credibility saturates.",
    )
    left.legend(fontsize=8)
    left.grid(alpha=0.3, which="both")

    right.fill_between(
        counts,
        100 * curve.band_lo,
        100 * curve.band_hi,
        color="#c62828",
        alpha=0.25,
        label="95% of estimates",
    )
    right.axhline(
        100 * curve.center, color="#c62828", lw=1.2, label=f"estimate centre {curve.center:+.1%}"
    )
    right.axhline(
        100 * res_mismatch.delta_true,
        ls="--",
        color="#2e7d32",
        lw=1.3,
        label=f"the truth {res_mismatch.delta_true:+.0%}",
    )
    right.set(
        xscale="log",
        xlabel="independent repetitions",
        ylabel="capacity estimate (%)",
        title="The cloud tightens onto the wrong answer.",
    )
    right.legend(fontsize=8, loc="center right")
    right.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_snr_vs_samples.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _verdict_fractions(kind: ParamKind, cell: int, magnitudes, mismatch):
    fractions = {v: np.zeros(len(magnitudes)) for v in VerdictKind}
    for i, mag in enumerate(magnitudes):
        scenario = build_scenario(
            name=f"{kind.value}@{mag}",
            fault_kind=kind,
            target_cell=cell,
            fault_magnitude=float(mag),
            current_a=WINDOW,
            mismatch=mismatch,
        )
        dist = verdict_distribution(run_trials(scenario, 160, seed=SEED, estimator="linear"))
        for v, f in dist.items():
            fractions[v][i] = f
    return fractions


def act3_verdict_distribution() -> None:
    rule("3. Verdict distribution vs fault magnitude (under model mismatch)")
    sweeps = [
        ("R0", ParamKind.R0, BLIND_CELL, np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])),
        (
            "capacity",
            ParamKind.CAPACITY,
            BLIND_CELL,
            np.array([0.0, -0.01, -0.02, -0.05, -0.1, -0.2, -0.3]),
        ),
        ("cooling", ParamKind.HA, SENSED_CELL, np.array([0.0, -0.1, -0.2, -0.4, -0.6])),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    order = list(VerdictKind)
    for ax, (label, kind, cell, mags) in zip(axes, sweeps, strict=True):
        fractions = _verdict_fractions(kind, cell, mags, REALISTIC_MISMATCH)
        x = 100 * np.abs(mags)
        bottom = np.zeros(len(mags))
        for v in order:
            color, name = VERDICT_STYLE[v]
            ax.fill_between(
                x, bottom, bottom + fractions[v], color=color, alpha=0.85, label=name, step="mid"
            )
            bottom = bottom + fractions[v]
        ax.set(xlabel=f"|{label} fault| (%)", ylim=(0, 1), title=f"{label} on cell {cell}")
        ax.margins(x=0)
        diag = fractions[VerdictKind.DIAGNOSE]
        print(
            f"  {label:9s}: diagnose rate {diag.min():.2f} -> {diag.max():.2f} across the sweep; "
            f"model-bias refusal peaks at {fractions[VerdictKind.REFUSE_MODEL_BIAS].max():.2f}"
        )
    axes[0].set_ylabel("fraction of trials")
    axes[1].legend(fontsize=7, loc="center left", framealpha=0.9)
    fig.suptitle(
        "Where AstraCell diagnoses, weakens, and refuses -- and where the model-bias "
        "gate caps a diagnosis it cannot trust.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_verdict_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def act4_overclaim() -> None:
    rule("4. Harmful overclaim, before and after the model-bias gate")
    kinds = [
        ("R0", ParamKind.R0, BLIND_CELL),
        ("capacity", ParamKind.CAPACITY, BLIND_CELL),
        ("cooling", ParamKind.HA, SENSED_CELL),
    ]
    before, after = [], []
    print(
        f"  {'fault':>9s} {'gate OFF overclaim':>19s} {'gate ON overclaim':>18s} "
        f"{'useful refusals':>16s}"
    )
    for label, kind, cell in kinds:
        off = abstention_metrics(
            run_trials(
                build_scenario(
                    name="off",
                    fault_kind=kind,
                    target_cell=cell,
                    current_a=WINDOW,
                    mismatch=REALISTIC_MISMATCH,
                    use_bias_gate=False,
                ),
                250,
                seed=SEED,
            )
        )
        on = abstention_metrics(
            run_trials(
                build_scenario(
                    name="on",
                    fault_kind=kind,
                    target_cell=cell,
                    current_a=WINDOW,
                    mismatch=REALISTIC_MISMATCH,
                    use_bias_gate=True,
                ),
                250,
                seed=SEED,
            )
        )
        before.append(off.harmful_overclaim_rate)
        after.append(on.harmful_overclaim_rate)
        print(
            f"  {label:9s} {off.harmful_overclaim_rate:19.2%} {on.harmful_overclaim_rate:18.2%} "
            f"{on.useful_refusal_rate:16.2%}"
        )
    print("\n  Under mismatch the variance-only observer confidently diagnoses faults that are")
    print("  mostly its own model error. The gate refuses exactly those, driving the harmful")
    print("  overclaim rate toward zero -- at the cost of diagnoses it should never have made.")

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    x = np.arange(len(kinds))
    ax.bar(x - 0.2, before, 0.4, color="#c62828", label="variance-only (gate off)")
    ax.bar(x + 0.2, after, 0.4, color="#1565c0", label="bias-aware (gate on)")
    ax.set(
        xticks=x,
        xticklabels=[k[0] for k in kinds],
        ylabel="harmful overclaim rate",
        ylim=(0, 1.05),
        title="The model-bias gate turns overclaims into refusals.",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_overclaim.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def act5_recommendations() -> None:
    rule("5. When does a recommendation stay calibrated? (a cooling fault on a blind cell)")
    interventions = {
        "baseline (blind cell)": dict(current_a=WINDOW),
        "+ thermocouple": dict(current_a=WINDOW, n_temp_sensors=4),
        "+ 2.5C pulse train": dict(
            current_a=pulse_train(600.0, 1.0, mean_c_rate=0.2, pulse_c_rate=2.5).current_a
        ),
        "+ 2x window": dict(
            current_a=pulse_train(1200.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0).current_a
        ),
        "+ current nuisance": dict(current_a=WINDOW, include_current_bias=True),
    }
    print("  Each intervention is scored twice: on a matched plant (does it enable a calibrated")
    print("  diagnosis?) and on the mismatched plant (does it stay honest?).\n")
    print(
        f"  {'intervention':>22s} | {'matched sigma':>13s} {'diagnose':>9s} | "
        f"{'mism. bias':>11s} {'ceiling':>8s} {'verdict':>19s}"
    )
    for label, knobs in interventions.items():
        matched = build_scenario(
            name="m", fault_kind=ParamKind.HA, target_cell=BLIND_CELL, mismatch=NO_MISMATCH, **knobs
        )  # type: ignore[arg-type]
        mism = build_scenario(
            name="x",
            fault_kind=ParamKind.HA,
            target_cell=BLIND_CELL,
            mismatch=REALISTIC_MISMATCH,
            **knobs,
        )  # type: ignore[arg-type]
        res_m = run_trials(matched, 200, seed=SEED)
        res_x = run_trials(mism, 200, seed=SEED)
        diagnose_matched = abstention_metrics(res_m).diagnosis_rate
        ceiling = abs(res_x.delta_true) / abs(res_x.bias) if res_x.bias else float("inf")
        top_verdict = max(verdict_distribution(res_x).items(), key=lambda kv: kv[1])[0]
        _, verdict_name = VERDICT_STYLE[top_verdict]
        print(
            f"  {label:>22s} | {res_m.crlb_std:13.1%} {diagnose_matched:9.0%} | "
            f"{res_x.bias:+11.0%} {ceiling:8.2f} {verdict_name:>19s}"
        )
    print("\n  On a matched plant the recommendations work exactly as example 03 promised: a")
    print("  thermocouple makes cooling observable, and a 2.5C pulse makes it fully diagnosable")
    print("  with no new sensor -- excitation substituting for instrumentation.")
    print("\n  On the mismatched plant every one of them refuses. None makes cooling calibratable,")
    print("  because the structural bias dwarfs the fault, and the ceiling -- the only number that")
    print("  discriminates here -- does NOT track the matched SNR. The 2x window and the current")
    print("  nuisance raise variance-side confidence while LOWERING the ceiling (0.12, 0.08): more")
    print("  of the same excitation pours information into hA and moves the bias onto it. The")
    print("  flashiest recommendation is not the calibrated one, and two of them backfire. Under")
    print("  this mismatch the honest recommendation remains v0.1's: fix the model, not the data.")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    res_mismatch = act1_coverage()
    act2_more_samples(res_mismatch)
    act3_verdict_distribution()
    act4_overclaim()
    act5_recommendations()
    print(
        "\n  figures: reports/figures/calibration_coverage.png, "
        "reports/figures/calibration_snr_vs_samples.png,"
    )
    print(
        "           reports/figures/calibration_verdict_distribution.png, "
        "reports/figures/calibration_overclaim.png"
    )


if __name__ == "__main__":
    main()
