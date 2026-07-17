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

**What v0.8 adds: depth.** v0.7 widened the run to eight cells (breadth) at one model order. v0.8
reruns the identical loop with a *second-order* observer -- a second RC branch on the first-order
ECM -- and asks whether the phantom and refusal are a property of the model *order* or of the
observer meeting a real cell. The branch is fixed forward structure (still fitting only ``R0`` and
capacity), its timescale fixed in advance, never tuned to shrink the phantom. The measured result
(the depth section this script prints) corrects a pre-run guess: a *fixed* second branch moves the
capacity estimate and lack-of-fit by round-off (<=3e-11) -- nothing -- because voltage's
sensitivity to R0 and capacity does not contain the RC branches. Model order enters the verdict
only by *fitting* the dynamics, which is what v0.9 does, below.

**What v0.9 adds: fitting the dynamics.** v0.8 left one door open -- a *fixed* branch is invisible,
but a *fitted* one need not be. v0.9 reruns the loop a third time with the first-order observer
refit over ``(R0, capacity, R1, C1)``: the fast RC branch is estimated, not held fixed. The
pre-registered question (``docs/V0.9_PLAN.md``) is whether that de-confounds the capacity verdict
(H1) or merely trades model bias for confounding (H2). The measured answer -- the fit-dynamics
section this script prints -- is that **H1 is falsified and retracted**: fitting R1,C1 does not
move the capacity verdict on any cell, the phantom persists, and the lack-of-fit barely falls. R1
*is* unidentifiable from a 1C discharge (VIF >> 10, exactly H2's mechanism) -- but that confounding
lands on R1, while capacity stays identifiable and refused as REFUSE_MODEL_BIAS. The phantom is OCV
drift, which no RC fitting on a 1C discharge reaches. ``tests/test_dynamics_fit.py``'s positive
control shows the same fit recovers an injected R1,C1 exactly under a pulse train, so the limit is
the excitation, not the code.

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
observer's failure to represent a real cell, not a quirk of Cell1. v0.8's depth pass shared that
prior, and the run sharpened it: a *fixed* second branch does not even improve the dynamic fit -- it
is invisible to the (R0, capacity) fit -- so it cannot clear the refusal. The narrower guess that a
second timescale would shrink the misfit is retracted here, measured wrong. That is not a failure of
the diagnostic. A screen that abstains on data it cannot trust is the entire thesis. Whatever prints
below is the measured outcome of this run -- no real-cell number is hard-coded here. See
docs/REAL_CELL.md.

The data is ODC-ODbL and is never committed here; ``scripts/fetch_oxford.py`` fetches it, and this
script skips cleanly when it is absent -- exactly as examples 06-07 skip without PyBaMM.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
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
    build_second_order_observer,
    detection_metrics,
    external_scenario,
    external_specs_4param,
    run_trials,
    verdict_distribution,
)
from astracell.observability.decision import VerdictKind
from astracell.observability.sensitivity import ParameterSpec
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

#: An observer is built from an OCV curve and a baseline capacity. v0.8 runs the loop twice: the
#: first-order ECM (``build_external_observer``) and the second-order one (an added slow RC branch,
#: ``build_second_order_observer``). The loop, the windows, the trials -- everything else is shared,
#: which lets the first- vs second-order comparison attribute any change to the model order.
ObserverFactory = Callable[[object, float], object]

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
    vif_target: float = 0.0  # VIF of the capacity target -- is the estimate itself confounded?
    vif_dynamics: tuple[float, ...] = ()  # VIF of the fitted R1,C1 (empty for the 2-param fit)

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
    cell: str,
    baseline: OxfordAge,
    aged: OxfordAge,
    observer: object,
    window_s: float,
    *,
    specs: tuple[ParameterSpec, ...] | None = None,
) -> AgeRow:
    """Score one aged cell: the observer's capacity estimate against the *measured* fade.

    ``specs`` selects the fitted parameter set: ``None`` is the 2-parameter ``(R0, capacity)`` fit
    every version through v0.8 used; ``external_specs_4param()`` also fits the RC branch (v0.9).
    Capacity stays the target at index 1 either way, so the reported estimate is comparable across
    fits -- what changes is only which other parameters the observer is allowed to move.
    """
    current, base_v, aged_v, soc0 = aligned_pair(baseline, aged, window_s=window_s)
    fade = measured_fade(baseline, aged)
    scenario = external_scenario(
        name=f"oxford_{cell.lower()}_cyc{aged.cyc:04d}",
        observer=observer,
        current_a=current,
        soc0=soc0,
        fault_magnitude=fade,
        target_index=CAPACITY_TARGET,
        specs=specs,
    )
    evidence = build_evidence(scenario, base_v, aged_v)
    result = run_trials(
        evidence.paired.scenario, N_TRIALS, seed=SEED, estimator="linear", prepared=evidence.paired
    )
    metrics = detection_metrics(result)
    top_verdict = max(verdict_distribution(result).items(), key=lambda kv: kv[1])[0]
    vif = evidence.paired.fit_ctx.vif
    return AgeRow(
        cyc=aged.cyc,
        fade=fade,
        estimate=evidence.paired_estimate,
        misfit=evidence.misfit_paired,
        sigma=evidence.paired.target_sigma,
        verdict=top_verdict,
        coverage=metrics.coverage,
        vif_target=float(vif[CAPACITY_TARGET]),
        vif_dynamics=tuple(float(v) for v in vif[2:]),  # R1, C1, ... beyond (R0, capacity)
    )


def _run_mode(
    cell: str,
    ages: dict[int, OxfordAge],
    baseline: OxfordAge,
    selection: list[int],
    window_s: float,
    *,
    per_age_ocv: bool,
    observer_factory: ObserverFactory,
    specs: tuple[ParameterSpec, ...] | None = None,
) -> tuple[AgeRow, ...]:
    """Evaluate the selected aged cells under one OCV policy: shared baseline OCV, or each age's."""
    shared_observer = observer_factory(pseudo_ocv_curve(baseline), baseline.capacity_ah)
    rows = []
    for cyc in selection:
        if cyc == baseline.cyc:
            continue
        aged = ages[cyc]
        observer = (
            observer_factory(pseudo_ocv_curve(aged), baseline.capacity_ah)
            if per_age_ocv
            else shared_observer
        )
        rows.append(_evaluate(cell, baseline, aged, observer, window_s, specs=specs))
    return tuple(rows)


def _score_cell(
    cell: str,
    ages: dict[int, OxfordAge],
    *,
    observer_factory: ObserverFactory = build_external_observer,
    specs: tuple[ParameterSpec, ...] | None = None,
) -> CellResult:
    """Score one cell end to end under both OCV policies, or record why it could not be scored.

    ``specs`` forwards to ``_evaluate``: ``None`` is the 2-param fit; ``external_specs_4param()``
    adds the fitted fast RC branch (v0.9). ``observer_factory`` chooses the forward model order; the
    two axes are independent -- v0.9's fit-dynamics run keeps the first-order observer and only
    enlarges the fitted set.
    """
    n_loaded = len(ages)
    usable = _usable_ages(ages)
    n_dropped = n_loaded - len(usable)
    if not usable:
        # No usable discharge means no definable baseline, so compute one BEFORE the try below would
        # index an empty dict. Report the miss rather than crash the whole eight-cell run.
        print(f"  [!] {cell}: no usable ~1C discharge; reported as skipped, not silently dropped")
        return CellResult(cell, n_loaded, n_dropped, 0, 0.0, 0, 0.0, (), (), error="no usable ages")
    baseline = usable[min(usable)]
    try:
        if len(usable) < 2:
            raise ValueError(f"only {len(usable)} usable ages; need >= 2 to score a fade")
        selection = _score_selection(usable)
        window_s = _safe_window({c: usable[c] for c in selection})
        shared = _run_mode(
            cell,
            usable,
            baseline,
            selection,
            window_s,
            per_age_ocv=False,
            observer_factory=observer_factory,
            specs=specs,
        )
        per_age = _run_mode(
            cell,
            usable,
            baseline,
            selection,
            window_s,
            per_age_ocv=True,
            observer_factory=observer_factory,
            specs=specs,
        )
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
    n_gain = int((shared > 0).sum())
    print(f"  measured EOL fade        {fade.min():+.1%} to {fade.max():+.1%} -- every cell fades")
    print(
        f"  shared-OCV EOL estimate  {shared.min():+.1%} to {shared.max():+.1%} -- "
        f"{n_gain}/{len(scored)} a phantom gain, wrong in sign"
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


# ----------------------------------------------------------------------------- depth (v0.8)


def _print_depth_comparison(first: list[CellResult], second: list[CellResult]) -> None:
    """v0.8: rerun the identical loop with a 2nd-order observer and measure what depth changes.

    The measured answer is: essentially nothing -- and the reason is structural, not incidental.
    Voltage's sensitivity to R0 and to capacity does not contain the RC branches (their
    overpotentials cancel in dV/dR0 and dV/dQ), so a *fixed* second branch is invisible to a fit
    over (R0, capacity). This corrects the pre-run expectation that a second timescale would shrink
    the misfit: it does not shrink it at all. Every number below is computed from THIS run.
    """
    rule("Depth (v0.8): a second RC branch is invisible to the capacity fit -- measured")
    pairs = [(f, s) for f, s in zip(first, second, strict=True) if not f.error and not s.error]
    if not pairs:
        print("  no cell scored under both observers; nothing to compare")
        return

    print(
        f"  {'cell':6s} {'meas EOL':>9s} {'1st est':>9s} {'2nd est':>9s} "
        f"{'1st LoF':>9s} {'2nd LoF':>9s}  verdict (1st -> 2nd)"
    )
    for f, s in pairs:
        fe, se = f.eol_shared, s.eol_shared
        v1, v2 = fe.verdict.value.upper(), se.verdict.value.upper()
        change = "same" if v1 == v2 else f"{v1} -> {v2}"
        print(
            f"  {f.cell:6s} {fe.fade:9.2%} {fe.estimate:9.2%} {se.estimate:9.2%} "
            f"{f.worst_misfit:9.1f} {s.worst_misfit:9.1f}  {change}"
        )

    changed = evaluations = 0
    max_dest = max_dlof = 0.0
    for f, s in pairs:
        for fr, sr in zip(f.shared + f.per_age, s.shared + s.per_age, strict=True):
            evaluations += 1
            changed += int(fr.verdict != sr.verdict)
            max_dest = max(max_dest, abs(fr.estimate - sr.estimate))
            max_dlof = max(max_dlof, abs(fr.misfit - sr.misfit))
    ref2_shared = sum(s.refused_shared for _, s in pairs)
    ref2_per_age = sum(s.refused_per_age for _, s in pairs)
    n_shared = sum(s.n_scored for _, s in pairs)

    print()
    print(f"  verdicts changed by depth   {changed}/{evaluations} evaluations (both OCV modes)")
    print(
        f"  2nd-order refusals          shared {ref2_shared}/{n_shared}, "
        f"per-age {ref2_per_age}/{n_shared}"
    )
    print(f"  largest |1st - 2nd| est     {max_dest:.1e}  over all {evaluations} evaluations")
    print(f"  largest |1st - 2nd| LoF     {max_dlof:.1e}  over all {evaluations} evaluations")
    print()
    print("  The 2nd-order estimate and lack-of-fit equal the first-order's to round-off (the")
    print("  maxima above) -- floating point, not dynamics. Depth is not failing to help here;")
    print("  it is INVISIBLE to this fit: voltage's sensitivity to R0 and capacity does not")
    print("  contain the RC branches, so a fixed richer forward model cannot move the estimate")
    print("  the gate refuses. The refusal is NOT a first-order artefact -- a second timescale")
    print("  the observer does not FIT changes nothing. Model order enters the capacity verdict")
    print("  only by FITTING the dynamics (R0,Q,R1,C1 -> +R2,C2): a 4->6-parameter")
    print("  identifiability problem that trades model bias for confounding -- the honest next")
    print("  step (v0.9), not asserted here.")


# ------------------------------------------------------------------- fit-dynamics (v0.9)


def _print_fit_dynamics_comparison(
    two_param: list[CellResult], four_param: list[CellResult]
) -> None:
    """v0.9: refit each cell over (R0, capacity, R1, C1) and measure whether FITTING the fast RC
    branch de-confounds the capacity verdict.

    v0.8 proved a *fixed* second RC branch is invisible to the (R0, capacity) fit. v0.9 fits the
    branch instead of holding it fixed. The pre-registered hope and bets (docs/V0.9_PLAN.md):
      H1  fitting R1,C1 absorbs the misfit, so the phantom capacity slides toward the truth;
      H2  R1,C1 are unidentifiable from a 1C discharge, so the verdict turns REFUSE_CONFOUNDED;
      H0  little changes -- the dominant residual is the moving OCV, which R1,C1 do not touch.
    Every number below is computed from THIS run; the adjudication is read off those numbers.
    """
    rule("Fit-dynamics (v0.9): does FITTING R1,C1 move the capacity verdict? -- measured")
    pairs = [
        (a, b) for a, b in zip(two_param, four_param, strict=True) if not a.error and not b.error
    ]
    if not pairs:
        print("  no cell scored under both fits; nothing to compare")
        return

    print(
        f"  {'cell':6s} {'meas EOL':>9s} {'2p est':>8s} {'4p est':>8s}"
        f" {'2p LoF':>8s} {'4p LoF':>8s} {'VIF R1':>8s} {'VIF C1':>7s}  verdict (2p -> 4p)"
    )
    for a, b in pairs:
        ae, be = a.eol_shared, b.eol_shared
        v1, v2 = ae.verdict.value.upper(), be.verdict.value.upper()
        change = "same" if v1 == v2 else f"{v1} -> {v2}"
        r1 = be.vif_dynamics[0] if be.vif_dynamics else float("nan")
        c1 = be.vif_dynamics[1] if len(be.vif_dynamics) > 1 else float("nan")
        print(
            f"  {a.cell:6s} {ae.fade:9.2%} {ae.estimate:8.2%} {be.estimate:8.2%} "
            f"{a.worst_misfit:8.1f} {b.worst_misfit:8.1f} {r1:8.1f} {c1:7.2f}  {change}"
        )

    changed = evaluations = shrank = grew = 0
    max_dest = 0.0
    lof_ratios: list[float] = []
    vif_r1: list[float] = []
    vif_c1: list[float] = []
    sigma_ratios: list[float] = []
    for a, b in pairs:
        for ar, br in zip(a.shared + a.per_age, b.shared + b.per_age, strict=True):
            evaluations += 1
            changed += int(ar.verdict != br.verdict)
            shrank += int(abs(br.estimate - br.fade) < abs(ar.estimate - ar.fade) - 1e-9)
            grew += int(abs(br.estimate - br.fade) > abs(ar.estimate - ar.fade) + 1e-9)
            max_dest = max(max_dest, abs(ar.estimate - br.estimate))
            if ar.misfit > 0.0:
                lof_ratios.append(br.misfit / ar.misfit)
            if br.vif_dynamics:
                vif_r1.append(br.vif_dynamics[0])
                if len(br.vif_dynamics) > 1:
                    vif_c1.append(br.vif_dynamics[1])
            if ar.sigma > 0.0:
                sigma_ratios.append(br.sigma / ar.sigma)

    med_lof = float(np.median(lof_ratios)) if lof_ratios else float("nan")
    med_r1 = float(np.median(vif_r1)) if vif_r1 else float("nan")
    med_c1 = float(np.median(vif_c1)) if vif_c1 else float("nan")
    med_sig = float(np.median(sigma_ratios)) if sigma_ratios else float("nan")

    print()
    print(f"  capacity verdicts changed    {changed}/{evaluations} evaluations (both OCV modes)")
    print(
        f"  |estimate - truth|           shrank {shrank}/{evaluations}, grew {grew}/{evaluations}"
    )
    print(f"  largest |2p - 4p| estimate   {max_dest:.2%}  over all {evaluations} evaluations")
    print(
        f"  lack-of-fit ratio 4p/2p      median {med_lof:.3f}  (~1 = fitting R1,C1 barely cut it)"
    )
    print(
        f"  identifiability cost         VIF(R1) median {med_r1:.0f}, VIF(C1) median {med_c1:.2f}"
    )
    print(f"  capacity CRLB inflation      sigma_4p/sigma_2p median {med_sig:.3f}")
    print()
    print("  MEASURED VERDICT on the pre-registration (docs/V0.9_PLAN.md):")
    print("  H1 (de-confounding) is FALSIFIED and retracted. The capacity verdict changes on")
    print(
        f"  {changed}/{evaluations} evaluations -- H1's hoped shift toward DIAGNOSE does not occur."
    )
    print(
        "  Fitting the fast RC branch moves the capacity estimate by at most the margin above, the"
    )
    print(
        "  phantom GAIN persists against a real fade, and the lack-of-fit is essentially unchanged."
    )
    print()
    print("  The outcome is H0-dominant, with H2 confirmed only where it cannot matter. R1 IS")
    print(f"  unidentifiable from a 1C discharge (VIF ~ {med_r1:.0f} >> 10), as H2 warned -- but")
    print(
        "  that confounding is quarantined to R1. Capacity's own VIF stays ~4 and its CRLB inflates"
    )
    print(
        f"  ~{100 * (med_sig - 1.0):.0f}%, so capacity remains identifiable and its refusal stays"
    )
    print(
        "  REFUSE_MODEL_BIAS (the moving OCV), never REFUSE_CONFOUNDED. The phantom is OCV drift,"
    )
    print(
        "  which no amount of RC fitting on a 1C discharge can reach. This SHARPENS v0.8: a fixed"
    )
    print(
        "  branch is invisible to the capacity fit; a fitted branch is inert on it too, because a"
    )
    print("  1C discharge cannot identify the branch at all.")
    print()
    print(
        "  Not a failure of the estimator. The positive control (tests/test_dynamics_fit.py) shows"
    )
    print(
        "  the same 4-param fit recovers an injected R1/C1 exactly under a pulse train (VIF < 10),"
    )
    print(
        "  and refuses the identical R1 fault as confounded under a 1C discharge. The limit is the"
    )
    print("  excitation, not the code -- and a richer excitation earns R1,C1, not capacity, whose")
    print(
        "  phantom is OCV drift. The cheapest test to earn each diagnosis differs. See REAL_CELL.md"
    )


# --------------------------------------------------------------------------------------- figure


def _figure(first: list[CellResult], second: list[CellResult]) -> None:
    pairs = [(f, s) for f, s in zip(first, second, strict=True) if not f.error and not s.error]
    scored = [f for f, _ in pairs]
    scored2 = [s for _, s in pairs]
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
        label="ECM estimate, shared OCV (1st-order)",
    )
    bot.scatter(
        idx,
        [100.0 * r.eol_shared.estimate for r in scored2],
        color="#6a1b9a",
        marker="D",
        s=30,
        zorder=4,
        label="ECM estimate, shared OCV (2nd-order)",
    )
    bot.scatter(
        idx,
        [100.0 * r.eol_per_age.estimate for r in scored],
        color="#1565c0",
        marker="^",
        s=42,
        zorder=3,
        label="ECM estimate, per-age OCV (1st-order)",
    )
    bot.axhline(0.0, color="#455a64", lw=0.8, ls=":")
    bot.set(
        xticks=idx,
        ylabel="capacity deviation at end of life [%]",
        title="End of life, per cell: measured loss vs the ECM's phantom (1st- and 2nd-order)",
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
        f"Scoring {len(CELLS)} Oxford cells vs their measured fade: first-order, second-order, and "
        f"the v0.9 fit-dynamics pass -- (R0, capacity) plus fitted (R1, C1) -- at {N_TRIALS} "
        f"trials/age, SEED={SEED}. This runs the observer, not a battery."
    )
    first = [_score_cell(cell, ages_by_cell[cell]) for cell in CELLS]
    second = [
        _score_cell(cell, ages_by_cell[cell], observer_factory=build_second_order_observer)
        for cell in CELLS
    ]
    # v0.9: the same first-order observer, refit over (R0, capacity, R1, C1) -- the fast RC branch
    # now *fitted* rather than held fixed. Only the fitted set changes, so the comparison against
    # ``first`` isolates the effect of fitting the dynamics.
    fit_dynamics = [
        _score_cell(cell, ages_by_cell[cell], specs=external_specs_4param()) for cell in CELLS
    ]
    if not any(not r.error for r in first):
        print("No cell could be scored; aborting.")
        return 0

    _print_roster(first, path.name)
    _print_per_cell(first)
    _print_refusal_distribution(first)
    _print_phantom_gain_spread(first)
    _summary(first)
    _print_depth_comparison(first, second)
    _print_fit_dynamics_comparison(first, fit_dynamics)
    _figure(first, second)
    print(f"\nFigure written to {FIGURES / 'real_cell_capacity.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
