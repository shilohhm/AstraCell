"""Calibrated abstention: are the diagnose/refuse decisions statistically meaningful?

Everything before v0.2 was a property of the design -- what *could* be resolved. This package
runs the estimator many times against a known injected truth and asks whether the resulting
intervals and verdicts hold up as *frequencies*: does a 90% interval cover 90% of the time,
does the bias gate turn overclaims into refusals, does more data buy precision without buying
accuracy.

It cannot prove AstraCell right about real batteries -- there are none here. It can prove
AstraCell *self-consistent*: honest about its own noise and its own model error, on its own
terms. ``docs/CALIBRATION.md`` draws that line precisely.

Pipeline::

    build_scenario  ->  prepare  ->  run_trials  ->  {coverage, abstention_metrics, ...}
"""

from astracell.calibration.metrics import (
    NOMINAL_LEVELS,
    AbstentionMetrics,
    SampleCountCurve,
    abstention_metrics,
    coverage,
    coverage_curve,
    covered,
    sample_count_curve,
    two_sided_z,
    verdict_distribution,
)
from astracell.calibration.montecarlo import (
    ScenarioContext,
    TrialResults,
    prepare,
    run_trials,
)
from astracell.calibration.scenario import (
    NOMINAL_FAULT,
    Scenario,
    build_scenario,
)

__all__ = [
    "NOMINAL_FAULT",
    "NOMINAL_LEVELS",
    "AbstentionMetrics",
    "Scenario",
    "ScenarioContext",
    "SampleCountCurve",
    "TrialResults",
    "abstention_metrics",
    "build_scenario",
    "coverage",
    "coverage_curve",
    "covered",
    "prepare",
    "run_trials",
    "sample_count_curve",
    "two_sided_z",
    "verdict_distribution",
]
