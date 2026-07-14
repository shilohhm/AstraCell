"""AstraCell's first measured cell: the Oxford Battery Degradation Dataset (Tier 3).

Run:  python examples/08_real_cell.py      (needs the data first: python scripts/fetch_oxford.py)

Every result before this one was synthetic. Tier 1 tested the ECM against itself and against a
hand-built mismatch; Tier 2 tested it against PyBaMM, an electrochemical simulator we did not write.
Both are models. This is eight real Kokam pouch cells (Howey & Birkl 2017), cycled at 40 degC to
end of life, characterised every 100 drive cycles with a 1C discharge and a slow pseudo-OCV sweep.
It is the only thing in this repository that has ever touched a battery.

**What is new here is a real ground truth.** The dataset measures each age's capacity directly, from
the charge its own 1C discharge delivered. So for the first time AstraCell's capacity estimate can
be scored against a number nobody chose -- ``measured_fade`` -- rather than against an injected
truth or a simulator's parameter. Two observers are run against each aged cell:

* **shared-OCV (the deployable one).** Calibrate the pseudo-OCV once on the fresh cell and track.
  A real cell's OCV curve *moves* as it ages; the first-order ECM cannot express that motion, so the
  lack-of-fit screen should fire and capacity should be refused. This is the honest field setting.
* **per-age OCV (a control).** Re-measure the pseudo-OCV at every age, removing the OCV drift the
  ECM cannot model. Not deployable -- you cannot re-characterise a pack in a car -- but it isolates
  how much of the mismatch was the moving OCV versus everything else a real cell does.

**The honest expectation, stated before the run.** Continuing the v0.3 -> v0.4 arc, a real cell
should mismatch the ECM at least as badly as PyBaMM did, so the likely outcome is that AstraCell
*refuses capacity even harder* on a real cell. That is not a failure. A diagnostic that abstains on
data it cannot trust is the entire thesis. Whatever prints below is the measured outcome of this
run -- no real-cell number is hard-coded anywhere in this repository. See docs/REAL_CELL.md.

The data is ODC-ODbL and is never committed here; ``scripts/fetch_oxford.py`` fetches it, and this
script skips cleanly when it is absent -- exactly as examples 06-07 skip without PyBaMM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astracell.calibration import (
    CAPACITY_TARGET,
    build_evidence,
    build_external_observer,
    detection_metrics,
    external_scenario,
    run_trials,
    verdict_distribution,
)
from astracell.observability.decision import VerdictKind
from astracell.plant.oxford import (
    OxfordAge,
    aligned_pair,
    default_mat_path,
    load_cell,
    measured_fade,
    pseudo_ocv_curve,
)

FIGURES = Path(__file__).resolve().parents[1] / "reports" / "figures"
CELL = "Cell1"
SEED = 0
N_TRIALS = 400
N_SCORE = 14  # Cell1 has ~78 characterisation ages; score a readable spread across the cell's life

#: Keep the replay inside the ECM's representable SOC band even for the most-faded age, and inside
#: the shortest discharge. ``aligned_pair`` starts the observer at SOC 0.98 (the ECM's ceiling), so
#: the window may not empty the cell past ~SOC 0.03. Both bounds are read from data, not guessed.
_SOC0 = 0.98
_SOC_FLOOR = 0.03

#: A real dataset carries a few malformed characterisation cycles (partial or empty 1C discharges);
#: drop any age that is not a usable ~1C run rather than let one poison the shared window or a fade.
_MIN_CAPACITY_AH = 0.1
_MIN_DURATION_S = 600.0

VERDICT_STYLE = {
    VerdictKind.DIAGNOSE: ("#2e7d32", "diagnose"),
    VerdictKind.WEAK_EVIDENCE: ("#f9a825", "weak"),
    VerdictKind.REFUSE_UNOBSERVABLE: ("#9e9e9e", "refuse: unobservable"),
    VerdictKind.REFUSE_CONFOUNDED: ("#6a1b9a", "refuse: confounded"),
    VerdictKind.REFUSE_MODEL_BIAS: ("#c62828", "refuse: model bias"),
}


def rule(title: str) -> None:
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")


def _safe_window(ages: dict[int, OxfordAge]) -> float:
    """The longest window that stays inside both the shortest discharge and the ECM's SOC band.

    The most-faded cell empties fastest, so it sets the SOC-band limit; the shortest recorded
    discharge sets the other. Take 90% of the tighter of the two. Derived from the data.
    """
    durations = [float(a.time_s[-1] - a.time_s[0]) for a in ages.values()]
    q_min = min(a.capacity_ah for a in ages.values())
    i_mean = float(np.mean([float(np.mean(a.current_a)) for a in ages.values()]))
    band_limit = (_SOC0 - _SOC_FLOOR) * q_min * 3600.0 / max(i_mean, 1e-9)
    return 0.9 * min(min(durations), band_limit)


def _usable_ages(ages: dict[int, OxfordAge]) -> dict[int, OxfordAge]:
    """Keep only ages with a usable ~1C discharge.

    Real characterisation data carries a few partial or empty discharges, and one malformed age
    (zero capacity, zero duration) would drag the shared window to zero or fake an infinite fade.
    """
    return {
        cyc: age
        for cyc, age in ages.items()
        if age.capacity_ah > _MIN_CAPACITY_AH
        and age.current_a.size >= 4
        and float(age.time_s[-1] - age.time_s[0]) > _MIN_DURATION_S
    }


def _score_selection(ages: dict[int, OxfordAge]) -> list[int]:
    """At most ``N_SCORE`` ages, evenly spread across life (baseline and end always included)."""
    cycs = sorted(ages)
    if len(cycs) <= N_SCORE:
        return cycs
    picks = np.linspace(0, len(cycs) - 1, N_SCORE).round().astype(int)
    return sorted({cycs[i] for i in picks})


def _evaluate(
    baseline: OxfordAge, aged: OxfordAge, observer: object, window_s: float
) -> dict[str, object]:
    """Score one aged cell: the observer's capacity estimate against the *measured* fade."""
    current, base_v, aged_v, soc0 = aligned_pair(baseline, aged, window_s=window_s)
    fade = measured_fade(baseline, aged)
    scenario = external_scenario(
        name=f"oxford_{CELL.lower()}_cyc{aged.cyc:04d}",
        observer=observer,
        current_a=current,
        soc0=soc0,
        fault_magnitude=fade,
        target_index=CAPACITY_TARGET,
    )
    evidence = build_evidence(scenario, base_v, aged_v)
    result = run_trials(
        evidence.paired.scenario, N_TRIALS, seed=SEED, estimator="linear", prepared=evidence.paired
    )
    metrics = detection_metrics(result)
    top_verdict = max(verdict_distribution(result).items(), key=lambda kv: kv[1])[0]
    return {
        "cyc": aged.cyc,
        "fade": fade,
        "estimate": evidence.paired_estimate,
        "misfit": evidence.misfit_paired,
        "sigma": evidence.paired.target_sigma,
        "verdict": top_verdict,
        "coverage": metrics.coverage,
        "diagnose": metrics.diagnosis_rate,
    }


def _run_mode(
    ages: dict[int, OxfordAge],
    baseline: OxfordAge,
    selection: list[int],
    window_s: float,
    *,
    per_age_ocv: bool,
) -> list[dict[str, object]]:
    """Evaluate the selected aged cells under one OCV policy: shared baseline OCV, or each age's."""
    shared_observer = build_external_observer(pseudo_ocv_curve(baseline), baseline.capacity_ah)
    rows = []
    for cyc in selection:
        if cyc == baseline.cyc:
            continue
        aged = ages[cyc]
        observer = (
            build_external_observer(pseudo_ocv_curve(aged), baseline.capacity_ah)
            if per_age_ocv
            else shared_observer
        )
        rows.append(_evaluate(baseline, aged, observer, window_s))
    return rows


def _print_table(title: str, rows: list[dict[str, object]]) -> None:
    rule(title)
    print(
        f"  {'cyc':>5s} {'measured fade':>14s} {'observer est':>13s} {'lack-of-fit':>12s} "
        f"{'sigma':>8s} {'coverage':>9s}  verdict"
    )
    for r in rows:
        print(
            f"  {r['cyc']:5d} {r['fade']:14.2%} {r['estimate']:13.2%} {r['misfit']:12.2e} "
            f"{r['sigma']:8.2%} {r['coverage']:9.2f}  {r['verdict'].value.upper()}"  # type: ignore[union-attr]
        )


def _figure(
    baseline: OxfordAge, shared: list[dict[str, object]], per_age: list[dict[str, object]]
) -> None:
    cyc = np.array([r["cyc"] for r in shared], dtype=float)
    fade = 100.0 * np.array([r["fade"] for r in shared])
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True, height_ratios=[1.4, 1])

    top.plot(cyc, fade, "k-o", lw=1.6, ms=4, label="measured capacity fade (ground truth)")
    for rows, colour, name in (
        (shared, "#c62828", "observer estimate (shared OCV)"),
        (per_age, "#1565c0", "observer estimate (per-age OCV)"),
    ):
        est = 100.0 * np.array([r["estimate"] for r in rows])
        top.plot(cyc, est, "--s", color=colour, lw=1.2, ms=4, label=name)
    top.axhline(0.0, color="#455a64", lw=0.7, ls=":")
    top.set(
        ylabel="capacity deviation [%]",
        title=f"Oxford {CELL}: the ECM's capacity estimate vs the cell's measured fade",
    )
    top.legend(fontsize=8)
    top.grid(alpha=0.3)

    for rows, colour, name in (
        (shared, "#c62828", "shared OCV"),
        (per_age, "#1565c0", "per-age OCV"),
    ):
        mis = np.array([float(r["misfit"]) for r in rows])  # type: ignore[arg-type]
        bot.semilogy(cyc, np.maximum(mis, 1e-9), "-o", color=colour, lw=1.2, ms=4, label=name)
    bot.set(
        xlabel="characterisation cycle",
        ylabel="differential lack-of-fit",
        title="What the observer cannot reproduce: the screen that gates the estimate above",
    )
    bot.legend(fontsize=8)
    bot.grid(alpha=0.3, which="both")

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "real_cell_capacity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _summary(
    baseline: OxfordAge, shared: list[dict[str, object]], per_age: list[dict[str, object]]
) -> None:
    rule("Summary: AstraCell's first contact with a measured cell")
    refused = {VerdictKind.REFUSE_MODEL_BIAS, VerdictKind.REFUSE_CONFOUNDED}
    n_shared_refused = sum(1 for r in shared if r["verdict"] in refused)
    n_age_refused = sum(1 for r in per_age if r["verdict"] in refused)
    n = len(shared)
    print(f"  cell                     Oxford {CELL}, Kokam 740 mAh pouch, 40 degC to end of life")
    print(
        f"  baseline age             cyc{baseline.cyc:04d}, "
        f"measured capacity {baseline.capacity_ah:.3f} Ah"
    )
    print(f"  ages scored              {n}, against their own measured 1C fade")
    print("  observer                 first-order Thevenin ECM (R0, R1C1), measured pseudo-OCV")
    print()
    print(f"  shared-OCV capacity      refused in {n_shared_refused}/{n} ages (deployable path)")
    print(f"  per-age-OCV capacity     refused in {n_age_refused}/{n} ages (OCV drift removed)")
    print()
    print("  These are the measured outcomes of THIS run, not asserted targets. Read them against")
    print("  docs/REAL_CELL.md and LIMITATIONS.md: one cell of eight, a first-order ECM, a shared")
    print("  baseline that is a different day and temperature -- every reason to distrust a")
    print("  confident diagnosis is present, which is exactly why the refusal is the point. Tier 3")
    print("  is contact, not validation: nothing here yet says the ECM is RIGHT about a real cell,")
    print("  only whether AstraCell knows when it is not.")


def main() -> int:
    path = default_mat_path()
    if not path.exists():
        print(f"No Oxford dataset at {path}.")
        print("Fetch it first (ODC-ODbL, ~254 MB):  python scripts/fetch_oxford.py")
        print("Skipping (this is not a failure -- the dataset is an optional, licensed download).")
        return 0

    try:
        ages = load_cell(path, CELL)
    except ImportError as exc:
        print(f"The Oxford loader needs the optional [oxford] extra: {exc}")
        print("Install it:  pip install -e '.[oxford]'   then re-run.")
        return 0

    usable = _usable_ages(ages)
    dropped = len(ages) - len(usable)
    if len(usable) < 2:
        print(f"Only {len(usable)} usable ages in {CELL}; need >= 2 to score a fade. Aborting.")
        return 0
    baseline = usable[min(usable)]
    selection = _score_selection(usable)
    window_s = _safe_window({c: usable[c] for c in selection})
    scored = [c for c in selection if c != baseline.cyc]

    rule("0. The measured cell")
    note = f"; dropped {dropped} malformed" if dropped else ""
    print(f"  loaded {len(ages)} characterisation ages of Oxford {CELL} from {path.name}{note}")
    print(f"  scoring {len(scored)} ages: {', '.join(f'cyc{c:04d}' for c in selection)}")
    print(f"  baseline cyc{baseline.cyc:04d}: measured capacity {baseline.capacity_ah:.4f} Ah")
    print(f"  paired-fit window: {window_s:.0f} s (inside the shortest discharge and the SOC band)")

    shared = _run_mode(usable, baseline, selection, window_s, per_age_ocv=False)
    per_age = _run_mode(usable, baseline, selection, window_s, per_age_ocv=True)

    _print_table("1. Shared baseline OCV (deployable): capacity estimate vs measured fade", shared)
    _print_table("2. Per-age OCV (control): the same, with OCV drift removed", per_age)
    _figure(baseline, shared, per_age)
    _summary(baseline, shared, per_age)
    print(f"\nFigure written to {FIGURES / 'real_cell_capacity.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
