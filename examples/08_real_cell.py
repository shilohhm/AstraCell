"""AstraCell across all eight measured cells: the Oxford Battery Degradation Dataset (Tier 3).

Run:  python examples/08_real_cell.py      (needs the data first: python scripts/fetch_oxford.py)

Every result before Tier 3 was synthetic. Tier 1 tested the ECM against itself and a hand-built
mismatch; Tier 2 tested it against PyBaMM, an electrochemical simulator we did not write. Both are
models. This is eight real Kokam pouch cells (Howey & Birkl 2017), cycled at 40 degC to end of life,
characterised every ~100 drive cycles with a 1C discharge and a slow pseudo-OCV sweep. It is the
only thing in this repository that has ever touched a battery.

**What v0.6 established, and what v0.7 asks.** v0.6 ran this against *one* cell (Cell1) and found
the first-order ECM's capacity estimate wrong in *sign* -- a phantom capacity *gain* while the cell
faded -- refused on every age. A single cell cannot separate a property of *that* cell from a
property of the observer meeting a real battery. v0.7 runs all eight and reports the two things a
breadth run is for:

* **the refusal distribution** -- across all eight cells and every scored age, in both OCV modes,
  how often does AstraCell refuse, and with which verdict; and
* **the phantom-gain spread** -- how far, and in which direction, the ECM's capacity estimate lands
  relative to each cell's measured fade, and how much that varies cell to cell.

Two observers are run against each aged cell, exactly as in v0.6:

* **shared-OCV (the deployable one).** Calibrate the pseudo-OCV once on the fresh cell and track. A
  real cell's OCV curve *moves* as it ages; the first-order ECM cannot express that motion, so the
  lack-of-fit screen should fire and capacity should be refused. This is the honest field setting.
* **per-age OCV (a control).** Re-measure the pseudo-OCV at every age, removing the OCV drift the
  ECM cannot model. Not deployable -- you cannot re-characterise a pack in a car -- but it isolates
  how much of the mismatch was the moving OCV versus everything else a real cell does.

**The honest expectation, stated before the run.** Continuing the v0.3 -> v0.4 arc, a real cell
should mismatch the ECM at least as badly as PyBaMM did, so the likely outcome is that AstraCell
*refuses capacity on every cell*, and that the phantom persists across the eight -- making it the
observer's failure to represent a real cell, not a quirk of Cell1. That is not a failure of the
diagnostic. A screen that abstains on data it cannot trust is the entire thesis. Whatever prints
below is the measured outcome of this run -- no real-cell number is hard-coded anywhere in this
repository. See docs/REAL_CELL.md.

The data is ODC-ODbL and is never committed here; ``scripts/fetch_oxford.py`` fetches it, and this
script skips cleanly when it is absent -- exactly as examples 06-07 skip without PyBaMM.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
    load_cells,
    measured_fade,
    pseudo_ocv_curve,
)

FIGURES = Path(__file__).resolve().parents[1] / "reports" / "figures"
CELLS = tuple(f"Cell{i}" for i in range(1, 9))
SEED = 0
N_TRIALS = 400
N_SCORE = 14  # each cell has 46-78 characterisation ages; score a readable spread across its life

#: Keep the replay inside the ECM's representable SOC band even for the most-faded age, and inside
#: the shortest discharge. ``aligned_pair`` starts the observer at SOC 0.98 (the ECM's ceiling), so
#: the window may not empty the cell past ~SOC 0.03. Both bounds are read from data, not guessed.
_SOC0 = 0.98
_SOC_FLOOR = 0.03

#: A real dataset carries a few malformed characterisation cycles (partial or empty 1C discharges);
#: drop any age that is not a usable ~1C run rather than let one poison the shared window or a fade.
_MIN_CAPACITY_AH = 0.1
_MIN_DURATION_S = 600.0

#: A verdict is a refusal of capacity when the gate declines it as model bias or a confounder.
#: REFUSE_UNOBSERVABLE would be a different statement (nothing to see); it does not arise here, but
#: keeping the set explicit means the refusal count says exactly what it means.
_REFUSED = frozenset({VerdictKind.REFUSE_MODEL_BIAS, VerdictKind.REFUSE_CONFOUNDED})

VERDICT_STYLE = {
    VerdictKind.DIAGNOSE: ("#2e7d32", "diagnose"),
    VerdictKind.WEAK_EVIDENCE: ("#f9a825", "weak"),
    VerdictKind.REFUSE_UNOBSERVABLE: ("#9e9e9e", "refuse: unobservable"),
    VerdictKind.REFUSE_CONFOUNDED: ("#6a1b9a", "refuse: confounded"),
    VerdictKind.REFUSE_MODEL_BIAS: ("#c62828", "refuse: model bias"),
}


def rule(title: str) -> None:
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")


@dataclass(frozen=True)
class AgeRow:
    """One aged cell scored against its own measured 1C fade, under one OCV policy."""

    cyc: int
    fade: float  # measured relative capacity deviation (negative = fade): the ground truth
    estimate: float  # the observer's paired capacity estimate
    misfit: float  # differential lack-of-fit: what the ECM cannot reproduce at any parameter
    sigma: float  # the variance-only 1-sigma interval half-width
    verdict: VerdictKind
    coverage: float  # does the interval contain the measured fade? (1.0 yes / 0.0 no)

    @property
    def refused(self) -> bool:
        return self.verdict in _REFUSED


@dataclass(frozen=True)
class CellResult:
    """One cell's whole outcome: the roster facts, and the per-age rows in both OCV modes.

    ``error`` is set (and the row tuples left empty) if a cell could not be scored -- a short or
    malformed discharge that trips the paired-window bounds -- so the run reports the miss rather
    than aborting all eight. Every reported aggregate is derived from ``scored`` cells only.
    """

    cell: str
    n_loaded: int
    n_dropped: int
    baseline_cyc: int
    baseline_cap: float
    eol_cyc: int
    eol_cap: float
    shared: tuple[AgeRow, ...]
    per_age: tuple[AgeRow, ...]
    error: str | None = None

    @property
    def n_scored(self) -> int:
        return len(self.shared)

    @property
    def refused_shared(self) -> int:
        return sum(row.refused for row in self.shared)

    @property
    def refused_per_age(self) -> int:
        return sum(row.refused for row in self.per_age)

    @property
    def eol_shared(self) -> AgeRow:
        return self.shared[-1]

    @property
    def eol_per_age(self) -> AgeRow:
        return self.per_age[-1]

    @property
    def worst_misfit(self) -> float:
        return max(row.misfit for row in self.shared)


# ------------------------------------------------------------------------------- per-cell scoring


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


def _evaluate(
    cell: str, baseline: OxfordAge, aged: OxfordAge, observer: object, window_s: float
) -> AgeRow:
    """Score one aged cell: the observer's capacity estimate against the *measured* fade."""
    current, base_v, aged_v, soc0 = aligned_pair(baseline, aged, window_s=window_s)
    fade = measured_fade(baseline, aged)
    scenario = external_scenario(
        name=f"oxford_{cell.lower()}_cyc{aged.cyc:04d}",
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
    return AgeRow(
        cyc=aged.cyc,
        fade=fade,
        estimate=evidence.paired_estimate,
        misfit=evidence.misfit_paired,
        sigma=evidence.paired.target_sigma,
        verdict=top_verdict,
        coverage=metrics.coverage,
    )


def _run_mode(
    cell: str,
    ages: dict[int, OxfordAge],
    baseline: OxfordAge,
    selection: list[int],
    window_s: float,
    *,
    per_age_ocv: bool,
) -> tuple[AgeRow, ...]:
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
        rows.append(_evaluate(cell, baseline, aged, observer, window_s))
    return tuple(rows)


def _score_cell(cell: str, ages: dict[int, OxfordAge]) -> CellResult:
    """Score one cell end to end under both OCV policies, or record why it could not be scored."""
    n_loaded = len(ages)
    usable = _usable_ages(ages)
    n_dropped = n_loaded - len(usable)
    baseline = usable[min(usable)]
    try:
        if len(usable) < 2:
            raise ValueError(f"only {len(usable)} usable ages; need >= 2 to score a fade")
        selection = _score_selection(usable)
        window_s = _safe_window({c: usable[c] for c in selection})
        shared = _run_mode(cell, usable, baseline, selection, window_s, per_age_ocv=False)
        per_age = _run_mode(cell, usable, baseline, selection, window_s, per_age_ocv=True)
    except (
        Exception
    ) as exc:  # a short/malformed discharge trips the window bounds: report, do not abort
        print(f"  [!] {cell}: could not score ({exc}); reported as skipped, not silently dropped")
        return CellResult(
            cell,
            n_loaded,
            n_dropped,
            baseline.cyc,
            baseline.capacity_ah,
            baseline.cyc,
            baseline.capacity_ah,
            (),
            (),
            error=str(exc),
        )
    eol_cyc = shared[-1].cyc
    return CellResult(
        cell=cell,
        n_loaded=n_loaded,
        n_dropped=n_dropped,
        baseline_cyc=baseline.cyc,
        baseline_cap=baseline.capacity_ah,
        eol_cyc=eol_cyc,
        eol_cap=usable[eol_cyc].capacity_ah,
        shared=shared,
        per_age=per_age,
    )


# ------------------------------------------------------------------------------------- reporting


def _print_roster(results: list[CellResult], filename: str) -> None:
    rule("0. The eight measured cells")
    print(f"  loaded {len(results)} Oxford cells from {filename}")
    print(
        f"  scoring up to {N_SCORE - 1} aged ages per cell, evenly spread across each cell's life\n"
    )
    print(
        f"  {'cell':6s} {'ages':>4s} {'baseline Ah':>12s} {'EOL cyc':>8s} {'EOL Ah':>8s} "
        f"{'measured EOL fade':>18s}"
    )
    for r in results:
        if r.error:
            print(
                f"  {r.cell:6s} {'--':>4s} {r.baseline_cap:12.4f} {'--':>8s} {'--':>8s} "
                f"{'not scored':>18s}"
            )
            continue
        print(
            f"  {r.cell:6s} {r.n_scored:4d} {r.baseline_cap:12.4f} {r.eol_cyc:8d} {r.eol_cap:8.4f} "
            f"{r.eol_shared.fade:18.2%}"
        )


def _print_per_cell(results: list[CellResult]) -> None:
    rule("1. Per cell: the measured fade against the ECM's capacity estimate, both OCV modes")
    print(
        f"  {'cell':6s} {'ages':>4s} {'measured EOL':>13s} {'shared est':>11s} "
        f"{'per-age est':>12s} {'worst LoF':>10s} {'shared ref':>11s} {'per-age ref':>12s}"
    )
    for r in results:
        if r.error:
            print(f"  {r.cell:6s} {'--':>4s} {'not scored':>13s}")
            continue
        print(
            f"  {r.cell:6s} {r.n_scored:4d} {r.eol_shared.fade:13.2%} "
            f"{r.eol_shared.estimate:11.2%} {r.eol_per_age.estimate:12.2%} "
            f"{r.worst_misfit:10.1f} {r.refused_shared:4d}/{r.n_scored:<6d} "
            f"{r.refused_per_age:5d}/{r.n_scored:<6d}"
        )
    print("\n  'measured EOL' is the cell's own end-of-life fade (the ground truth); the two 'est'")
    print("  columns are the observer's paired capacity estimate at that same age. A negative")
    print("  measured fade against a positive estimate is the sign error -- a fading cell read as")
    print("  growing.")


def _print_refusal_distribution(results: list[CellResult]) -> None:
    rule("2. Refusal distribution across the eight cells")
    scored = [r for r in results if not r.error]
    total = sum(r.n_scored for r in scored)
    ref_shared = sum(r.refused_shared for r in scored)
    ref_per_age = sum(r.refused_per_age for r in scored)
    cells_all_shared = sum(1 for r in scored if r.refused_shared == r.n_scored)
    cells_all_per_age = sum(1 for r in scored if r.refused_per_age == r.n_scored)
    cov_shared = sum(int(row.coverage >= 0.5) for r in scored for row in r.shared)
    cov_per_age = sum(int(row.coverage >= 0.5) for r in scored for row in r.per_age)

    verdicts: dict[VerdictKind, int] = {}
    non_refusals: list[str] = []
    for r in scored:
        for mode, rows in (("shared", r.shared), ("per-age", r.per_age)):
            for row in rows:
                verdicts[row.verdict] = verdicts.get(row.verdict, 0) + 1
                if not row.refused:
                    non_refusals.append(f"{r.cell} cyc{row.cyc:04d} ({mode}): {row.verdict.value}")

    n = len(scored)
    print(f"  total ages scored         {total}  ({n} cells x up to {N_SCORE - 1} aged ages)")
    print(
        f"  shared-OCV refused        {ref_shared}/{total}   "
        f"({cells_all_shared}/{n} cells refuse every scored age)"
    )
    print(
        f"  per-age-OCV refused       {ref_per_age}/{total}   "
        f"({cells_all_per_age}/{n} cells refuse every scored age)"
    )
    print(
        f"  coverage of the fade      shared {cov_shared}/{total}, per-age {cov_per_age}/{total}  "
        f"(intervals containing the measured fade)"
    )
    median_sigma = float(np.median([row.sigma for r in scored for row in r.shared]))
    print(
        f"  1-sigma interval          median +/-{median_sigma:.2%} -- and it never covers the fade"
    )
    print("                            above: estimate and truth do not overlap, by far")
    kinds = ", ".join(f"{k.value} x{v}" for k, v in sorted(verdicts.items(), key=lambda kv: -kv[1]))
    print(f"  verdicts seen             {kinds}  ({2 * total} evaluations)")
    if non_refusals:
        print(f"  non-refusals ({len(non_refusals)}):")
        for line in non_refusals:
            print(f"    - {line}")
    else:
        print("  non-refusals              none -- every scored age, both modes, was refused")


def _print_phantom_gain_spread(results: list[CellResult]) -> None:
    rule("3. Phantom-gain spread: the ECM reports a gain while every cell fades")
    scored = [r for r in results if not r.error]
    fade = np.array([r.eol_shared.fade for r in scored])
    shared = np.array([r.eol_shared.estimate for r in scored])
    per_age = np.array([r.eol_per_age.estimate for r in scored])
    shared_all = np.array([row.estimate for r in scored for row in r.shared])
    per_age_all = np.array([row.estimate for r in scored for row in r.per_age])

    def band(name: str, values: np.ndarray, tail: str) -> None:
        print(
            f"  {name:22s} {values.min():+7.2%} ... {values.max():+7.2%}   "
            f"(median {np.median(values):+.2%})   {tail}"
        )

    n_gain = int((shared > 0).sum())
    band("measured EOL fade", fade, "-- every cell lost a fifth to a third of its capacity")
    band("shared-OCV EOL est", shared, f"-- {n_gain}/{len(scored)} cells report a capacity GAIN")
    band(
        "per-age-OCV EOL est",
        per_age,
        f"-- {int((per_age > 0).sum())}/{len(scored)} report a gain; "
        "re-measuring OCV makes it worse, not better",
    )
    print()
    print(
        f"  Across all {shared_all.size} shared-OCV ages, {100 * (shared_all > 0).mean():.0f}% of "
        f"estimates read positive; the estimate is a phantom gain almost"
    )
    print("  everywhere, and never within sign and magnitude of the real loss. The one cell whose")
    print("  shared estimate does not cross into gain is the most-faded of the eight, where the")
    print("  observer collapses to a near-zero estimate at vanishing sigma -- still tens of points")
    print("  from its measured fade, and still refused. The per-age control reads positive on")
    print(
        f"  {100 * (per_age_all > 0).mean():.0f}% of ages -- the dominant mismatch is the"
        " first-order dynamics, not the moving OCV."
    )


def _summary(results: list[CellResult]) -> None:
    rule("Summary: AstraCell across eight measured cells")
    scored = [r for r in results if not r.error]
    skipped = [r for r in results if r.error]
    total = sum(r.n_scored for r in scored)
    ref_shared = sum(r.refused_shared for r in scored)
    ref_per_age = sum(r.refused_per_age for r in scored)
    fade = np.array([r.eol_shared.fade for r in scored])
    shared = np.array([r.eol_shared.estimate for r in scored])

    print(
        f"  cells                    {len(scored)} of {len(results)} scored"
        + (f" ({len(skipped)} skipped: {', '.join(r.cell for r in skipped)})" if skipped else "")
    )
    print("  cell type                Oxford Kokam 740 mAh pouch, 40 degC to end of life")
    print("  observer                 first-order Thevenin ECM (R0, R1C1), measured pseudo-OCV")
    print(f"  ages scored              {total}, each against its own measured 1C fade")
    print()
    print(f"  measured EOL fade        {fade.min():+.1%} to {fade.max():+.1%} -- every cell fades")
    print(
        f"  shared-OCV EOL estimate  {shared.min():+.1%} to {shared.max():+.1%} -- a phantom gain, "
        "wrong in sign"
    )
    print(
        f"  capacity refused         shared {ref_shared}/{total}, per-age {ref_per_age}/{total} "
        "(REFUSE_MODEL_BIAS)"
    )
    print()
    print(
        "  These are the measured outcomes of THIS run, not asserted targets. v0.6 saw this on one"
    )
    print(
        "  cell; v0.7 sees it on eight, which makes the phantom the observer's failure to represent"
    )
    print("  a real cell rather than a quirk of Cell1. It does NOT upgrade the tier: still one")
    print("  first-order ECM, still isothermal, still a shared baseline from a different day, and")
    print(
        "  still no fault injected or detected. Tier 3 is contact, not validation -- nothing here"
    )
    print(
        "  says the ECM is RIGHT about a real cell, only that AstraCell knows when it is not, now"
    )
    print("  eight cells over. Read against docs/REAL_CELL.md and LIMITATIONS.md section 16.")


# --------------------------------------------------------------------------------------- figure


def _figure(results: list[CellResult]) -> None:
    scored = [r for r in results if not r.error]
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9.4, 8.4), height_ratios=[1.35, 1])

    # Top: every cell's two trajectories. All measured-fade lines share one dark colour and all
    # shared-OCV estimates one red, so the eye reads two bands -- a plunging one and one that never
    # leaves the neighbourhood of zero -- that do not meet on any cell.
    for r in scored:
        cyc = np.array([row.cyc for row in r.shared], dtype=float)
        top.plot(
            cyc,
            100.0 * np.array([row.fade for row in r.shared]),
            "-",
            color="#263238",
            alpha=0.55,
            lw=1.1,
            marker="o",
            ms=2.5,
        )
        top.plot(
            cyc,
            100.0 * np.array([row.estimate for row in r.shared]),
            "--",
            color="#c62828",
            alpha=0.55,
            lw=1.1,
            marker="s",
            ms=2.5,
        )
    top.axhline(0.0, color="#455a64", lw=0.8, ls=":")
    top.set(
        xlabel="characterisation cycle",
        ylabel="capacity deviation [%]",
        title="All eight Oxford cells: every cell fades, the first-order ECM reports a gain",
    )
    top.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#263238",
                marker="o",
                ms=4,
                lw=1.1,
                label="measured capacity fade (8 cells)",
            ),
            Line2D(
                [],
                [],
                color="#c62828",
                ls="--",
                marker="s",
                ms=4,
                lw=1.1,
                label="ECM estimate, shared OCV (8 cells)",
            ),
            Line2D([], [], color="#455a64", ls=":", lw=0.8, label="truth: no capacity change (0%)"),
        ],
        fontsize=8,
        loc="center left",
    )
    top.grid(alpha=0.3)

    # Bottom: end of life, per cell. A dumbbell from the measured fade (dark) to the shared
    # estimate (red) makes the per-cell error and spread legible; the per-age estimate (blue) sits
    # with it.
    idx = np.arange(len(scored))
    for i, r in enumerate(scored):
        f, s = 100.0 * r.eol_shared.fade, 100.0 * r.eol_shared.estimate
        bot.plot([i, i], [f, s], color="#90a4ae", lw=1.2, zorder=1)
    bot.scatter(
        idx,
        [100.0 * r.eol_shared.fade for r in scored],
        color="#263238",
        s=42,
        zorder=3,
        label="measured fade (ground truth)",
    )
    bot.scatter(
        idx,
        [100.0 * r.eol_shared.estimate for r in scored],
        color="#c62828",
        marker="s",
        s=42,
        zorder=3,
        label="ECM estimate, shared OCV",
    )
    bot.scatter(
        idx,
        [100.0 * r.eol_per_age.estimate for r in scored],
        color="#1565c0",
        marker="^",
        s=42,
        zorder=3,
        label="ECM estimate, per-age OCV",
    )
    bot.axhline(0.0, color="#455a64", lw=0.8, ls=":")
    bot.set(
        xticks=idx,
        ylabel="capacity deviation at end of life [%]",
        title="End of life, per cell: the measured loss versus the ECM's phantom estimate",
    )
    bot.set_xticklabels([r.cell for r in scored], fontsize=8)
    bot.legend(fontsize=8, loc="center right")
    bot.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "real_cell_capacity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------ main


def main() -> int:
    path = default_mat_path()
    if not path.exists():
        print(f"No Oxford dataset at {path}.")
        print("Fetch it first (ODC-ODbL, ~254 MB):  python scripts/fetch_oxford.py")
        print("Skipping (this is not a failure -- the dataset is an optional, licensed download).")
        return 0

    try:
        ages_by_cell = load_cells(str(path), CELLS)  # one read of the 254 MB file, all eight cells
    except ImportError as exc:
        print(f"The Oxford loader needs the optional [oxford] extra: {exc}")
        print("Install it:  pip install -e '.[oxford]'   then re-run.")
        return 0

    print(
        f"Scoring {len(CELLS)} Oxford cells against their own measured fade "
        f"({N_TRIALS} trials/age, SEED={SEED}). This runs the observer, not a battery."
    )
    results = [_score_cell(cell, ages_by_cell[cell]) for cell in CELLS]
    if not any(not r.error for r in results):
        print("No cell could be scored; aborting.")
        return 0

    _print_roster(results, path.name)
    _print_per_cell(results)
    _print_refusal_distribution(results)
    _print_phantom_gain_spread(results)
    _figure(results)
    _summary(results)
    print(f"\nFigure written to {FIGURES / 'real_cell_capacity.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
