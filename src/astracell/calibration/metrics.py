"""What the trials add up to: coverage, abstention quality, and the variance-vs-bias curve.

These are pure functions over ``TrialResults`` arrays. The separation matters: the harness
decides *what happened*, and this module decides *what to make of it*, so the definitions of
"coverage" and "harmful overclaim" are here, in one place, arguable and testable, rather than
smuggled into a plotting script.

Three families of metric, kept distinct on purpose:

* **Interval calibration** -- does an interval claimed to cover the truth ``L`` of the time do
  so? This is about the *estimate*, independent of any verdict. ``coverage``.
* **Abstention quality** -- when the observer commits (DIAGNOSE) or declines (REFUSE_*), is it
  right to? This is about the *decision*. ``abstention_metrics``.
* **Detection quality** -- when a fault really is there, is it found, and found *correctly*?
  This needs a positive control to be meaningful at all. ``detection_metrics`` (v0.4).

The one number that ties v0.2 to the rest of the repository is the harmful-overclaim rate: a
trial that DIAGNOSES while its confident interval misses the truth. Under model mismatch the
variance-only observer does this constantly; the point of the bias gate is to convert those
into refusals, which is honest even though it lowers the diagnosis rate.

It is also, alone, insufficient. A gate that refuses every input drives harmful overclaim to
zero and diagnoses nothing -- and v0.3's external gate did exactly that without noticing, because
nothing in this module could tell the two apart. ``detection_metrics`` is what tells them apart:
a true positive requires the DIAGNOSE *and* an interval that covers the injected truth *and* the
right sign. Overclaim measures the harm avoided; true positives measure the price paid for it.
Report both or neither.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astracell.calibration.montecarlo import TrialResults
from astracell.observability.decision import VerdictKind

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: The nominal confidence levels the coverage plot reports against.
NOMINAL_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 0.99)

_REFUSALS = (
    VerdictKind.REFUSE_UNOBSERVABLE,
    VerdictKind.REFUSE_CONFOUNDED,
    VerdictKind.REFUSE_MODEL_BIAS,
)


def _normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, ~1e-9 accurate).

    Dependency-free on purpose: the repository is numpy-only, and pulling in scipy for one
    quantile would be the sort of scope creep this project exists to resist.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def two_sided_z(level: float) -> float:
    """The half-width multiplier for a two-sided interval at confidence ``level``."""
    return _normal_quantile(0.5 * (1.0 + level))


def _half_width(results: TrialResults, level: float, *, bias_aware: bool) -> float:
    """Interval half-width: ``z*sigma`` (variance-only) or ``z*sqrt(sigma^2+b^2)`` (bias-aware).

    ``inf`` when the parameter is unidentifiable (``sigma`` infinite) -- an unbounded interval
    trivially covers, which is the correct, useless answer for something you cannot see.
    """
    spread = results.crlb_std
    if bias_aware:
        spread = float(np.hypot(results.crlb_std, results.bias))
    return two_sided_z(level) * spread


def covered(results: TrialResults, level: float, *, bias_aware: bool = False) -> BoolArray:
    """Boolean mask: is the true fault inside each trial's interval at this level?"""
    half = _half_width(results, level, bias_aware=bias_aware)
    return np.abs(results.delta_hat - results.delta_true) <= half


def coverage(results: TrialResults, level: float, *, bias_aware: bool = False) -> float:
    """Empirical fraction of trials whose interval covers the truth. Should approach ``level``."""
    return float(np.mean(covered(results, level, bias_aware=bias_aware)))


def coverage_curve(
    results: TrialResults,
    levels: tuple[float, ...] = NOMINAL_LEVELS,
    *,
    bias_aware: bool = False,
) -> FloatArray:
    """Empirical coverage at each nominal level -- the y-axis of the coverage plot."""
    return np.array([coverage(results, level, bias_aware=bias_aware) for level in levels])


def verdict_distribution(results: TrialResults) -> dict[VerdictKind, float]:
    """Fraction of trials landing on each verdict kind (zeros included, so keys are stable)."""
    counts = {kind: 0 for kind in VerdictKind}
    for verdict in results.verdicts:
        counts[verdict] += 1
    return {kind: n / results.n_trials for kind, n in counts.items()}


@dataclass(frozen=True)
class AbstentionMetrics:
    """Decision-quality summary of one scenario's trials.

    ``harmful_overclaim_rate`` is the load-bearing number: a DIAGNOSE whose confident interval
    misses the truth. ``useful_refusal_rate`` is its counterweight: a model-bias refusal that
    prevented exactly such an overclaim.
    """

    n_trials: int
    coverage_level: float
    diagnosis_rate: float
    weak_rate: float
    refusal_rate: float
    refuse_unobservable_rate: float
    refuse_confounded_rate: float
    refuse_model_bias_rate: float
    harmful_overclaim_rate: float
    useful_refusal_rate: float


def abstention_metrics(
    results: TrialResults,
    *,
    coverage_level: float = 0.95,
) -> AbstentionMetrics:
    """Diagnosis / refusal rates and the two rates that judge them.

    A trial is a *harmful overclaim* if it DIAGNOSES while the truth lies outside its
    variance-only interval at ``coverage_level``. A *useful refusal* is a ``REFUSE_MODEL_BIAS``
    on a trial whose variance-only interval would have missed -- the gate earning its keep.
    """
    verdicts = np.array([str(v) for v in results.verdicts])
    n = results.n_trials
    is_diagnose = verdicts == str(VerdictKind.DIAGNOSE)
    is_model_bias = verdicts == str(VerdictKind.REFUSE_MODEL_BIAS)
    missed = ~covered(results, coverage_level, bias_aware=False)

    def rate(kind: VerdictKind) -> float:
        return float(np.mean(verdicts == str(kind)))

    return AbstentionMetrics(
        n_trials=n,
        coverage_level=coverage_level,
        diagnosis_rate=float(np.mean(is_diagnose)),
        weak_rate=rate(VerdictKind.WEAK_EVIDENCE),
        refusal_rate=float(np.mean(np.isin(verdicts, [str(k) for k in _REFUSALS]))),
        refuse_unobservable_rate=rate(VerdictKind.REFUSE_UNOBSERVABLE),
        refuse_confounded_rate=rate(VerdictKind.REFUSE_CONFOUNDED),
        refuse_model_bias_rate=rate(VerdictKind.REFUSE_MODEL_BIAS),
        harmful_overclaim_rate=float(np.mean(is_diagnose & missed)),
        useful_refusal_rate=float(np.mean(is_model_bias & missed)),
    )


@dataclass(frozen=True)
class DetectionMetrics:
    """Positive-control scoring: did the observer find the fault, and was it the right fault?

    ``diagnosis_rate`` counts DIAGNOSE verdicts. ``true_positive_rate`` counts only the ones that
    *earned* it -- the estimate has the right sign and its confident interval covers the truth.
    The gap between them is the rate of being **right for the wrong reason**, and it is not small
    where the phantom is large: v0.4's raw path diagnoses a doubled series resistance at 10.7 sigma
    while reporting 91.4% against a truth of 100%, an eight-point miss with a 0.12% interval. A
    diagnostic that names the fault and mis-sizes it by eight sigma has not detected anything.

    Null scenarios (``delta_true == 0``) have no true positives by definition, so every DIAGNOSE
    is a ``false_positive_rate``. On a null the false-positive and harmful-overclaim rates are
    *identically equal*: DIAGNOSE demands ``|z| >= 5`` and coverage demands ``|z| <= 1.96``, so no
    trial can be both. Asserted, not assumed, in ``tests/test_positive_control.py``.
    """

    n_trials: int
    delta_true: float
    coverage_level: float
    diagnosis_rate: float
    true_positive_rate: float
    false_positive_rate: float
    weak_rate: float
    refusal_rate: float
    harmful_overclaim_rate: float
    coverage: float

    @property
    def is_null(self) -> bool:
        return self.delta_true == 0.0


def detection_metrics(
    results: TrialResults,
    *,
    coverage_level: float = 0.95,
) -> DetectionMetrics:
    """Score one scenario's trials as a detector, against the truth that was injected."""
    verdicts = np.array([str(v) for v in results.verdicts])
    is_diagnose = verdicts == str(VerdictKind.DIAGNOSE)
    is_covered = covered(results, coverage_level, bias_aware=False)
    truth = results.delta_true
    null = truth == 0.0

    right_sign = np.sign(results.delta_hat) == np.sign(truth)
    earned = is_diagnose & is_covered & right_sign

    return DetectionMetrics(
        n_trials=results.n_trials,
        delta_true=truth,
        coverage_level=coverage_level,
        diagnosis_rate=float(np.mean(is_diagnose)),
        true_positive_rate=0.0 if null else float(np.mean(earned)),
        false_positive_rate=float(np.mean(is_diagnose)) if null else 0.0,
        weak_rate=float(np.mean(verdicts == str(VerdictKind.WEAK_EVIDENCE))),
        refusal_rate=float(np.mean(np.isin(verdicts, [str(k) for k in _REFUSALS]))),
        harmful_overclaim_rate=float(np.mean(is_diagnose & ~is_covered)),
        coverage=float(np.mean(is_covered)),
    )


@dataclass(frozen=True)
class SampleCountCurve:
    """The canonical variance-vs-bias curve: more data buys precision, never accuracy."""

    sample_counts: FloatArray
    snr_var: FloatArray  # sqrt(k) * |m| / sigma -- grows without bound
    snr_bias: FloatArray  # |m| / sqrt(sigma^2/k + b^2) -- saturates at the ceiling
    ceiling: float  # |m| / |b|
    center: float  # where the estimate cloud sits: delta_true + b, empirically
    band_lo: FloatArray  # a percentile band on the estimate, tightening as 1/sqrt(k)
    band_hi: FloatArray


def sample_count_curve(
    results: TrialResults,
    sample_counts: FloatArray,
    *,
    band_level: float = 0.95,
) -> SampleCountCurve:
    """Project the single-experiment trials onto an axis of repeated experiments.

    For the linear estimator the estimate is exactly linear in the noise, so averaging ``k``
    independent repeats is *exactly* the ``k=1`` scatter shrunk by ``1/sqrt(k)`` about the same
    (biased) centre. So the band here is not a fresh Monte Carlo but the exact rescaling of the
    one we ran -- which is the honest thing to draw, and makes the point sharper: the cloud
    tightens on a centre that never was the truth.

    The signal is the *true* fault magnitude ``|m|`` (a null scenario has none, so pass a
    faulted one). ``snr_var`` uses only the variance; ``snr_bias`` includes the frozen bias.
    """
    counts = np.asarray(sample_counts, dtype=float)
    if np.any(counts < 1.0):
        raise ValueError("sample counts must be >= 1")

    signal = abs(results.delta_true)
    sigma, bias = results.crlb_std, abs(results.bias)
    snr_var = np.sqrt(counts) * signal / sigma if sigma > 0.0 else np.full(counts.shape, np.inf)
    snr_bias = signal / np.sqrt(sigma**2 / counts + bias**2)
    ceiling = signal / bias if bias > 0.0 else math.inf

    center = float(np.mean(results.delta_hat))
    z = two_sided_z(band_level)
    scatter = z * results.crlb_std / np.sqrt(counts)
    return SampleCountCurve(
        sample_counts=counts,
        snr_var=np.asarray(snr_var, dtype=float),
        snr_bias=snr_bias,
        ceiling=ceiling,
        center=center,
        band_lo=center - scatter,
        band_hi=center + scatter,
    )
