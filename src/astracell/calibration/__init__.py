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

v0.4 adds the positive control, because refusal metrics alone cannot distinguish a system that
declines wisely from one that declines always::

    build_evidence  ->  {prepare_raw, prepare_paired}  ->  run_trials  ->  detection_metrics
"""

from astracell.calibration.external import (
    C1_TARGET,
    CAPACITY_TARGET,
    R0_TARGET,
    R1_TARGET,
    build_external_observer,
    build_second_order_observer,
    constant_profile,
    external_scenario,
    external_specs_4param,
    observer_voltage,
    prepare_external,
    pulse_profile,
)
from astracell.calibration.metrics import (
    NOMINAL_LEVELS,
    AbstentionMetrics,
    DetectionMetrics,
    SampleCountCurve,
    abstention_metrics,
    coverage,
    coverage_curve,
    covered,
    detection_metrics,
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
from astracell.calibration.positive_control import (
    PairedEvidence,
    build_evidence,
    differential_voltage,
    paired_noise,
    paired_scenario,
    prepare_paired,
    prepare_raw,
)
from astracell.calibration.scenario import (
    NOMINAL_FAULT,
    Scenario,
    build_scenario,
)

__all__ = [
    "C1_TARGET",
    "CAPACITY_TARGET",
    "NOMINAL_FAULT",
    "NOMINAL_LEVELS",
    "R0_TARGET",
    "R1_TARGET",
    "AbstentionMetrics",
    "DetectionMetrics",
    "PairedEvidence",
    "Scenario",
    "ScenarioContext",
    "SampleCountCurve",
    "TrialResults",
    "abstention_metrics",
    "build_evidence",
    "build_external_observer",
    "build_scenario",
    "build_second_order_observer",
    "constant_profile",
    "coverage",
    "coverage_curve",
    "covered",
    "detection_metrics",
    "differential_voltage",
    "external_scenario",
    "external_specs_4param",
    "observer_voltage",
    "paired_noise",
    "paired_scenario",
    "prepare",
    "prepare_external",
    "prepare_paired",
    "prepare_raw",
    "pulse_profile",
    "run_trials",
    "sample_count_curve",
    "two_sided_z",
    "verdict_distribution",
]
