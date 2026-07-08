"""Optimal experiment design: which measurement should I take next?

When AstraCell refuses a hypothesis, refusing is only half an answer. The other half
is *what would resolve it*. There are two levers:

1. **Instrument differently.** ``mask.recommend_temp_sensor`` -- a row mask over the
   existing sensitivity tensor, essentially free.
2. **Excite differently.** This module -- rank a library of candidate tests by the
   information each would add.

Fisher information from independent experiments **adds**, so ``FIM_after = FIM_before +
FIM(u)``. That additivity is a theorem, not a convenience, and it is asserted in
``tests/test_experiment.py``. It is what makes this cheap: simulate each candidate once,
then every ranking, re-ranking and multi-test plan is arithmetic on precomputed matrices.

Two scores, and the difference between them matters
---------------------------------------------------

**D-optimality** maximises ``det FIM`` -- the reciprocal volume of the joint confidence
ellipsoid over *every* parameter::

    EIG(u) = 0.5 * log( det FIM_after / det FIM_before )     [nats]

**Ds-optimality** maximises information about the *target* parameter alone, marginalising
the others as nuisance. For a single target that is exactly the entropy reduction of the
marginal posterior::

    EIG_target(u) = 0.5 * log( CRLB_before[target] / CRLB_after[target] )     [nats]

These disagree, and D-optimality is the wrong one here. A ten-second 2C pulse sharpens
``R0`` enormously for almost no time spent, which inflates ``det FIM`` -- while doing
almost nothing for a cooling fault. Ranked by D-optimality it looks like the best test
available. Ranked by what you actually asked about, it barely moves the answer.

You are diagnosing one thing. Optimise the axis you care about, not the volume.

Choosing a test, versus scoring one
-----------------------------------

``eig_target_per_minute`` is the right *greedy* score when you can keep running tests.
It is the wrong thing to maximise when you must cross a decision threshold once: then you
want the **cheapest test that gets you over the line**, which is what ``recommend_test``
returns. ``plan_tests`` does the greedy sequential version, accumulating information until
the target clears ``target_sigma`` or the budget runs out -- and reports honestly when no
sequence in the library can get there.

Assumptions, stated because they are load-bearing:

* Parameters are constant across experiments. Fine over minutes; false over months.
* Experiments are independent and each starts from the same operating point.
* The FIM is evaluated at the *nominal* (healthy) parameters, so this is **local**
  Ds-optimality, not a fully Bayesian expected information gain over a prior on theta.
  The ranking is more trustworthy than the absolute nats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astracell.duty.profiles import DutyProfile, constant_current, pulse_train, rest_then_pulse
from astracell.observability.fisher import (
    crlb,
    fisher_information,
    gaussian_entropy,
    information_gain,
    variance_inflation,
)
from astracell.observability.mask import DEFAULT_STRONG_SIGMA, detection_snr
from astracell.observability.sensitivity import ParameterSpec, sensitivities
from astracell.pack.params import PackParams
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import SensorTopology

FloatArray = NDArray[np.float64]

DEFAULT_MAX_VIF: float = 10.0


@dataclass(frozen=True)
class CandidateTest:
    """A diagnostic experiment we could choose to run."""

    name: str
    profile: DutyProfile
    rationale: str

    @property
    def cost_s(self) -> float:
        """Wall-clock cost of running it, in seconds."""
        return float(self.profile.time_s[-1] + self.profile.dt_s)

    @property
    def cost_minutes(self) -> float:
        return self.cost_s / 60.0


@dataclass(frozen=True)
class TestScore:
    """What one candidate test would buy, for one hypothesis."""

    test: CandidateTest
    eig_nats: float  # D-optimal: whole parameter set
    eig_target_nats: float  # Ds-optimal: the target's marginal. Use this one.
    eig_target_per_minute: float
    snr_after: float
    crlb_std_after: float
    vif_after: float

    @property
    def name(self) -> str:
        return self.test.name

    @property
    def cost_s(self) -> float:
        return self.test.cost_s

    def resolves(
        self, target_sigma: float = DEFAULT_STRONG_SIGMA, max_vif: float = DEFAULT_MAX_VIF
    ) -> bool:
        """Would this single test carry the hypothesis over the decision threshold?"""
        return self.snr_after >= target_sigma and self.vif_after <= max_vif


@dataclass(frozen=True)
class TestPlan:
    """A sequence of tests, and whether it actually gets you an answer."""

    steps: tuple[CandidateTest, ...]
    snr_trajectory: tuple[float, ...]  # SNR after each step; [0] is before any test
    resolved: bool
    target_sigma: float

    @property
    def total_cost_s(self) -> float:
        return sum(t.cost_s for t in self.steps)

    @property
    def final_snr(self) -> float:
        return self.snr_trajectory[-1]

    def render(self) -> str:
        lines = [f"  start: {self.snr_trajectory[0]:6.2f} sigma"]
        for i, test in enumerate(self.steps, start=1):
            lines.append(
                f"  {i}. run {test.name:<24s} ({test.cost_s:5.0f} s)  ->  "
                f"{self.snr_trajectory[i]:6.2f} sigma"
            )
        if self.resolved:
            lines.append(
                f"  RESOLVED after {self.total_cost_s / 60:.1f} min "
                f"({self.final_snr:.2f} >= {self.target_sigma:.0f} sigma)"
            )
        else:
            lines.append(
                f"  NOT RESOLVED. Best reachable in this library is {self.final_snr:.2f} sigma, "
                f"short of {self.target_sigma:.0f}. Instrument the pack instead."
            )
        return "\n".join(lines)


def default_test_library(capacity_ah: float = 60.0, dt_s: float = 1.0) -> tuple[CandidateTest, ...]:
    """A small library spanning diagnostic manoeuvres a BMS could actually command.

    Each is safe: no deep discharge, no overcharge, no thermal abuse. Every profile stays
    inside the simulator's valid SOC window from ``soc0 = 0.75``.

    ``rest_60s`` is a **negative control**, not a candidate. A scoring function that ranks
    it above zero is broken, and one that ranks it above a 2C pulse is broken loudly.
    """
    return (
        CandidateTest(
            "rest_60s",
            rest_then_pulse(
                60.0, dt_s, rest_s=60.0, pulse_c_rate=0.0, pulse_s=0.0, capacity_ah=capacity_ah
            ),
            "do nothing for a minute. The null test: a control, not a candidate.",
        ),
        CandidateTest(
            "pulse_0.5C_10s",
            rest_then_pulse(
                60.0, dt_s, rest_s=20.0, pulse_c_rate=0.5, pulse_s=10.0, capacity_ah=capacity_ah
            ),
            "a gentle current step. Cheap IR-drop excitation.",
        ),
        CandidateTest(
            "pulse_2C_10s",
            rest_then_pulse(
                60.0, dt_s, rest_s=20.0, pulse_c_rate=2.0, pulse_s=10.0, capacity_ah=capacity_ah
            ),
            "a hard current step. Maximal IR drop per second spent.",
        ),
        CandidateTest(
            "pulse_2C_180s_cooldown",
            rest_then_pulse(
                900.0, dt_s, rest_s=60.0, pulse_c_rate=2.0, pulse_s=180.0, capacity_ah=capacity_ah
            ),
            "heat the pack, then watch it cool. The only way to see a thermal time constant.",
        ),
        CandidateTest(
            "pulse_train_2.5C",
            pulse_train(600.0, dt_s, mean_c_rate=0.2, pulse_c_rate=2.5, capacity_ah=capacity_ah),
            "sustained hard pulsing. I^2 heating plus broadband current excitation.",
        ),
        CandidateTest(
            "slow_sweep_C/20",
            constant_current(1800.0, dt_s, c_rate=0.05, capacity_ah=capacity_ah),
            "a pseudo-OCV sweep. Traverses SOC slowly; the classic capacity measurement.",
        ),
    )


def _target_entropy_gain(variance_before: float, variance_after: float) -> float:
    """Entropy reduction [nats] of the target's *marginal* Gaussian posterior."""
    if not np.isfinite(variance_after) or variance_after <= 0.0:
        return 0.0
    if not np.isfinite(variance_before):
        return float("inf")
    return 0.5 * float(np.log(variance_before / variance_after))


def test_informations(
    params: PackParams,
    topology: SensorTopology,
    noise: NoiseModel,
    specs: tuple[ParameterSpec, ...],
    library: tuple[CandidateTest, ...],
    *,
    soc0: float = 0.75,
) -> list[FloatArray]:
    """Fisher information contributed by each candidate. **No priors** -- see ``rank_tests``.

    This is the only place simulations happen. Everything downstream is arithmetic on the
    matrices returned here, which is why ranking, re-ranking, and greedy planning are free.
    """
    out: list[FloatArray] = []
    for test in library:
        sens = sensitivities(params, test.profile.current_a, test.profile.dt_s, specs, soc0=soc0)
        out.append(fisher_information(sens, topology, noise))
    return out


def baseline_information(
    params: PackParams,
    topology: SensorTopology,
    noise: NoiseModel,
    specs: tuple[ParameterSpec, ...],
    baseline: DutyProfile | None,
    *,
    soc0: float = 0.75,
) -> FloatArray:
    """Information already in hand. Carries the nuisance priors -- exactly once."""
    n = len(specs)
    if baseline is None:
        return np.zeros((n, n))
    sens = sensitivities(params, baseline.current_a, baseline.dt_s, specs, soc0=soc0)
    return fisher_information(sens, topology, noise, specs=specs)


def _score(
    test: CandidateTest,
    fim_before: FloatArray,
    fim_test: FloatArray,
    index: int,
    magnitude: float,
) -> TestScore:
    fim_after = fim_before + fim_test
    var_before = crlb(fim_before)[index]
    var_after = crlb(fim_after)[index]
    eig_target = _target_entropy_gain(float(var_before), float(var_after))
    return TestScore(
        test=test,
        eig_nats=information_gain(fim_before, fim_after),
        eig_target_nats=eig_target,
        eig_target_per_minute=eig_target / max(test.cost_minutes, 1e-9),
        snr_after=float(detection_snr(np.array([var_after]), magnitude)[0]),
        crlb_std_after=float(np.sqrt(var_after)),
        vif_after=float(variance_inflation(fim_after)[index]),
    )


def rank_tests(
    params: PackParams,
    topology: SensorTopology,
    noise: NoiseModel,
    specs: tuple[ParameterSpec, ...],
    target: ParameterSpec,
    magnitude: float,
    *,
    library: tuple[CandidateTest, ...] | None = None,
    baseline: DutyProfile | None = None,
    soc0: float = 0.75,
) -> tuple[list[TestScore], FloatArray, list[FloatArray]]:
    """Score every candidate for one hypothesis.

    Returns ``(scores, fim_before, fim_per_test)``. Scores are sorted by
    ``eig_target_per_minute``, best first -- the Ds-optimal greedy score, not the
    D-optimal one. Hand ``fim_before`` and ``fim_per_test`` to ``plan_tests`` to avoid
    re-simulating.
    """
    if target not in specs:
        raise ValueError(f"{target.label()} is not among the differentiated parameters")
    index = specs.index(target)
    tests = library or default_test_library()

    fim_before = baseline_information(params, topology, noise, specs, baseline, soc0=soc0)
    fim_per_test = test_informations(params, topology, noise, specs, tests, soc0=soc0)

    scores = [
        _score(test, fim_before, fim_test, index, magnitude)
        for test, fim_test in zip(tests, fim_per_test, strict=True)
    ]
    scores.sort(key=lambda s: s.eig_target_per_minute, reverse=True)
    return scores, fim_before, fim_per_test


def recommend_test(
    scores: list[TestScore],
    *,
    target_sigma: float = DEFAULT_STRONG_SIGMA,
    max_vif: float = DEFAULT_MAX_VIF,
) -> TestScore | None:
    """The **cheapest** test that carries the hypothesis over the decision threshold.

    Not the most informative one, and not the best nats-per-minute one. If you have to
    cross a line once, you want the cheapest crossing. Returns ``None`` when no single
    test in the library gets there -- at which point ``plan_tests`` or a thermocouple.
    """
    clearing = [s for s in scores if s.resolves(target_sigma, max_vif)]
    if not clearing:
        return None
    return min(clearing, key=lambda s: s.cost_s)


def plan_tests(
    fim_before: FloatArray,
    fim_per_test: list[FloatArray],
    library: tuple[CandidateTest, ...],
    index: int,
    magnitude: float,
    *,
    target_sigma: float = DEFAULT_STRONG_SIGMA,
    max_tests: int = 12,
    allow_repeats: bool = True,
) -> TestPlan:
    """Greedy sequential design: keep adding the best test until the target clears.

    Greedy on **SNR gained per minute**, not on nats per minute. Those differ, and the
    difference matters: a test can be the most information-efficient thing available and
    still leave you short of the decision threshold. When you have to cross a line, you
    care about distance to the line.

    Information adds, so this needs no further simulation -- each step is a matrix sum and
    a diagonal of an inverse. Repeats are allowed because running the same experiment twice
    really does buy roughly sqrt(2) in the standard error, and pretending otherwise would
    understate what a technician can actually do.

    Reports ``resolved=False`` rather than silently returning the best of a bad lot.
    """
    fim = fim_before.copy()
    var = crlb(fim)[index]
    snr = [float(detection_snr(np.array([var]), magnitude)[0])]
    chosen: list[CandidateTest] = []
    remaining = list(range(len(library)))

    for _step in range(max_tests):
        if snr[-1] >= target_sigma:
            break

        best_i: int | None = None
        best_rate = -np.inf
        best_snr = snr[-1]
        best_fim: FloatArray | None = None

        for i in remaining:
            candidate = fim + fim_per_test[i]
            candidate_snr = float(detection_snr(np.array([crlb(candidate)[index]]), magnitude)[0])
            # Greedy on marginal SNR gained per minute of test time.
            gain_rate = (candidate_snr - snr[-1]) / max(library[i].cost_minutes, 1e-9)
            if gain_rate > best_rate:
                best_i, best_rate, best_snr, best_fim = i, gain_rate, candidate_snr, candidate

        if best_i is None or best_fim is None or best_snr <= snr[-1] + 1e-12:
            break  # nothing left that helps

        fim = best_fim
        chosen.append(library[best_i])
        snr.append(best_snr)
        if not allow_repeats:
            remaining.remove(best_i)

    return TestPlan(
        steps=tuple(chosen),
        snr_trajectory=tuple(snr),
        resolved=snr[-1] >= target_sigma,
        target_sigma=target_sigma,
    )


def render_ranking(scores: list[TestScore], fim_before: FloatArray, target: ParameterSpec) -> str:
    """A table a human can act on. Sorted by Ds-optimal nats per minute."""
    header = (
        f"  next-best-test for {target.label()}\n"
        f"  joint posterior entropy now: {gaussian_entropy(fim_before):+.2f} nats\n\n"
        f"  {'test':<24s} {'cost':>7s} {'EIG_D':>8s} {'EIG_target':>11s} "
        f"{'nats/min':>9s} {'SNR after':>10s} {'VIF':>6s}"
    )
    rows = [
        f"  {s.name:<24s} {s.cost_s:6.0f}s {s.eig_nats:8.2f} {s.eig_target_nats:11.2f} "
        f"{s.eig_target_per_minute:9.3f} {s.snr_after:9.2f}s {s.vif_after:6.2f}"
        for s in scores
    ]
    return "\n".join([header, *rows])
