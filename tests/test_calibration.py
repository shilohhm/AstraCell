"""Calibration is a claim about frequencies, so its tests are claims about frequencies.

The load-bearing ones are pinned to specific seeded scenarios rather than asserted in the
abstract, for the same reason the rest of the suite is: a calibration harness that cannot
reproduce its own numbers has no business certifying anyone else's. Every Monte Carlo here
is seeded, every tolerance is generous enough to survive the seed but tight enough to fail if
the underlying claim breaks.

Four things get proved, in rising order of how much they would hurt to be wrong:

1. The harness is deterministic and the noise it draws has the covariance the FIM assumes.
2. A correctly-specified estimator *attains* the Cramer-Rao bound -- the thing §10 of
   LIMITATIONS said had never been shown.
3. Under model mismatch the variance-only interval is overconfident, and the bias gate turns
   the resulting overclaims into refusals.
4. More data shrinks the variance and not the bias, on the nose.
"""

from __future__ import annotations

import numpy as np
import pytest

from astracell.calibration import (
    NOMINAL_LEVELS,
    abstention_metrics,
    build_scenario,
    coverage,
    coverage_curve,
    prepare,
    run_trials,
    sample_count_curve,
    two_sided_z,
    verdict_distribution,
)
from astracell.duty import constant_current, pulse_train
from astracell.observability.estimator import fit_gauss_newton
from astracell.observability.fisher import whiten_ar1
from astracell.observability.sensitivity import ParamKind
from astracell.plant import REALISTIC_MISMATCH
from astracell.sensors.noise import NoiseModel
from astracell.sensors.sampling import sample_measurement_noise
from astracell.sensors.topology import SensorTopology

# A short window keeps the Monte Carlo tests quick; the example uses a longer one.
SHORT = pulse_train(300.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0).current_a


def _cap(**kwargs: object) -> object:
    """A capacity-fault scenario on the short window, the workhorse of these tests."""
    return build_scenario(
        name="cap",
        fault_kind=ParamKind.CAPACITY,
        target_cell=1,
        current_a=SHORT,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- determinism


def test_run_trials_is_bit_for_bit_deterministic_under_a_seed() -> None:
    scenario = _cap()
    first = run_trials(scenario, 50, seed=7)
    again = run_trials(scenario, 50, seed=7)
    other = run_trials(scenario, 50, seed=8)

    np.testing.assert_array_equal(first.delta_hat, again.delta_hat)
    assert first.verdicts == again.verdicts
    # A different seed must actually move the estimates; otherwise the seed does nothing.
    assert not np.array_equal(first.delta_hat, other.delta_hat)


def test_the_sampled_noise_whitens_to_white() -> None:
    # The whole calibration edifice rests on Sigma_sample == Sigma_FIM. The sharpest statement
    # of that identity: whitening the sampled noise with the FIM's own whitener must return an
    # uncorrelated, unit-scaled process. If it does, the empirical estimator covariance the
    # coverage tests measure is comparable to the CRLB and not to some other matrix.
    rho = 0.6
    noise = NoiseModel(voltage_rho=rho)
    topology = SensorTopology(n_cells=1, voltage_cells=(0,), temp_cells=())
    n_time = 12
    rng = np.random.default_rng(0)
    whitened = np.stack(
        [
            whiten_ar1(sample_measurement_noise(topology, noise, n_time, rng)[:, 0, :1], rho)[:, 0]
            for _ in range(12000)
        ]
    )

    cov = np.cov(whitened, rowvar=False)
    variance = noise.voltage_variance
    np.testing.assert_allclose(np.diag(cov), variance, rtol=0.05)
    off_diagonal = cov - np.diag(np.diag(cov))
    assert np.abs(off_diagonal).max() < 0.1 * variance


# --------------------------------------------------------------------------- attainability


def test_the_estimator_recovers_an_injected_fault_from_noiseless_data() -> None:
    # Anti-tautology: fit against the plant with no noise added. A matched observer must recover
    # the exact fault, or every coverage number downstream is measuring a broken estimator.
    scenario = _cap()
    prepared = prepare(scenario)
    result = fit_gauss_newton(prepared.fit_ctx, prepared.plant_output)

    assert result.converged
    recovered = result.delta[scenario.target_index]
    assert recovered == pytest.approx(scenario.fault_magnitude, abs=2e-3)


def test_a_correctly_specified_estimator_attains_the_cramer_rao_bound() -> None:
    # The claim LIMITATIONS §10 could not previously make: the bound is not just a bound, the
    # maximum-likelihood estimator reaches it. Scatter matches CRLB; the estimate is unbiased.
    scenario = _cap()
    result = run_trials(
        scenario,
        150,
        seed=0,
        estimator="gauss_newton",
        estimator_options={"max_iter": 12, "step_tol": 1e-5},
    )

    assert result.delta_hat.std() == pytest.approx(result.crlb_std, rel=0.25)
    # Recognisably the injected fault: not zero, not doubled, centred near the truth.
    assert -0.065 < result.delta_hat.mean() < -0.035
    assert coverage(result, 0.90) > 0.75


def test_the_linear_estimator_covers_exactly_at_the_null() -> None:
    # At the null there is no fault and no mismatch, so the linear fit is exactly N(0, CRLB)
    # and its coverage is nominal to sampling error. This is the razor for noise/whitening bugs.
    scenario = _cap(fault_magnitude=0.0)
    result = run_trials(scenario, 3000, seed=0, estimator="linear")

    empirical = coverage_curve(result)
    for level, got in zip(NOMINAL_LEVELS, empirical, strict=True):
        assert abs(got - level) < 0.03


# --------------------------------------------------------------------------- mismatch honesty


def test_model_mismatch_makes_the_variance_only_interval_overconfident() -> None:
    scenario = _cap(mismatch=REALISTIC_MISMATCH)
    result = run_trials(scenario, 300, seed=0, estimator="linear")

    assert result.bias < -0.05  # a real structural bias, not rounding
    variance_only = coverage(result, 0.95, bias_aware=False)
    bias_aware = coverage(result, 0.95, bias_aware=True)
    assert variance_only < 0.10  # the 95% interval almost never covers -- badly overconfident
    assert bias_aware > variance_only  # admitting the bias widens the interval back over truth


def test_the_bias_gate_converts_overclaims_into_refusals() -> None:
    gated = abstention_metrics(run_trials(_cap(mismatch=REALISTIC_MISMATCH), 300, seed=0))
    ungated = abstention_metrics(
        run_trials(_cap(mismatch=REALISTIC_MISMATCH, use_bias_gate=False), 300, seed=0)
    )

    assert ungated.harmful_overclaim_rate > 0.5  # variance-only diagnoses a fault that is bias
    assert gated.harmful_overclaim_rate < ungated.harmful_overclaim_rate
    assert gated.harmful_overclaim_rate < 0.05  # the gate all but eliminates it
    assert gated.refuse_model_bias_rate > 0.5  # by refusing, which is the honest move
    assert gated.useful_refusal_rate > 0.5  # and those refusals were the ones that mattered


# --------------------------------------------------------------------------- reachability


def test_every_verdict_kind_is_reachable_and_counted() -> None:
    const = constant_current(300.0, 1.0, c_rate=0.5).current_a
    reach = {
        "diagnose": build_scenario(
            name="d", fault_kind=ParamKind.R0, target_cell=1, current_a=SHORT
        ),
        "weak_evidence": build_scenario(
            name="w", fault_kind=ParamKind.HA, target_cell=3, current_a=SHORT
        ),
        "refuse_unobservable": build_scenario(
            name="u", fault_kind=ParamKind.HA, target_cell=0, current_a=SHORT
        ),
        "refuse_model_bias": _cap(mismatch=REALISTIC_MISMATCH),
        "refuse_confounded": build_scenario(
            name="c",
            fault_kind=ParamKind.R0,
            target_cell=1,
            current_a=const,
            use_all_specs=True,
            include_current_bias=True,
        ),
    }
    for expected, scenario in reach.items():
        distribution = verdict_distribution(run_trials(scenario, 120, seed=0))
        hit = next(kind for kind in distribution if str(kind) == expected)
        assert distribution[hit] > 0.0, f"{expected} was never reached ({scenario.name})"


# --------------------------------------------------------------------------- variance vs bias


def test_more_samples_shrink_the_variance_but_not_the_bias() -> None:
    result = run_trials(_cap(mismatch=REALISTIC_MISMATCH), 200, seed=0, estimator="linear")
    counts = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
    curve = sample_count_curve(result, counts)

    # Variance-only confidence grows as sqrt(k): a 10^4 increase in samples is a 100x SNR.
    assert curve.snr_var[-1] / curve.snr_var[0] == pytest.approx(100.0, rel=1e-6)
    # Bias-aware SNR saturates at the ceiling and stops dead.
    assert curve.snr_bias[-1] == pytest.approx(curve.ceiling, rel=0.02)
    assert curve.snr_bias[-1] < 0.01 * curve.snr_var[-1]
    # The estimate cloud tightens by 100x...
    width = curve.band_hi - curve.band_lo
    assert width[-1] == pytest.approx(width[0] / 100.0, rel=1e-6)
    # ...onto a centre that was never the truth.
    assert abs(curve.center - result.delta_true) > 0.05


def test_two_sided_z_matches_the_standard_normal_quantiles() -> None:
    assert two_sided_z(0.95) == pytest.approx(1.959964, abs=1e-4)
    assert two_sided_z(0.99) == pytest.approx(2.575829, abs=1e-4)
    assert two_sided_z(0.6826895) == pytest.approx(1.0, abs=1e-4)
