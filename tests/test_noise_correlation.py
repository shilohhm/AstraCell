"""AR(1) noise correlation, and the current-bias nuisance parameter.

The claims tested here are exact. ``whiten_ar1`` is not an approximation to
``x^T R^-1 x``; it *is* ``x^T R^-1 x``, and the first test proves it against an explicitly
constructed correlation matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from astracell.duty import constant_current, pulse_train
from astracell.observability.fisher import (
    crlb,
    design_matrix,
    fisher_information,
    prior_information,
    variance_inflation,
    whiten_ar1,
)
from astracell.observability.sensitivity import (
    CURRENT_BIAS_SPEC,
    ParameterSpec,
    ParamKind,
    local_specs,
    perturb,
    sensitivities,
    with_current_bias,
)
from astracell.pack import PackTopology, nominal_pack
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import SensorTopology, realistic_topology


@pytest.fixture(scope="module")
def setup():
    pack = PackTopology(n_modules=2, cells_per_module=4)
    params = nominal_pack(pack, seed=0)
    topology = realistic_topology(pack, n_temp_sensors=2)
    duty = pulse_train(300.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0)
    return pack, params, topology, duty


# ---------------------------------------------------------------------------
# whiten_ar1 is exact
# ---------------------------------------------------------------------------
def test_whitening_with_zero_rho_is_the_identity() -> None:
    x = np.random.default_rng(0).standard_normal((50, 3, 2))
    assert whiten_ar1(x, 0.0) is x


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.9, -0.4])
def test_whitening_reproduces_the_inverse_correlation_quadratic_form(rho: float) -> None:
    """``whiten(x)^T whiten(x) == x^T R^-1 x`` for the AR(1) correlation matrix R.

    Built explicitly here and inverted with a dense solve, so the test knows nothing
    about the tridiagonal closed form the implementation exploits.
    """
    n = 40
    rng = np.random.default_rng(1)
    x = rng.standard_normal((n, 1, 3))

    lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    corr = rho ** lags.astype(float)

    flat = x[:, 0, :]
    expected = flat.T @ np.linalg.solve(corr, flat)

    w = whiten_ar1(x, rho)[:, 0, :]
    np.testing.assert_allclose(w.T @ w, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.9])
def test_whitening_suppresses_a_constant_and_amplifies_an_alternating_signature(rho: float) -> None:
    """The mechanism behind the whole robustness story, in six lines.

    AR(1) whitening is a scaled first difference. Its effect on information is an exact,
    symmetric pair: a **constant** sensitivity keeps a fraction ``(1-rho)/(1+rho)``; an
    **alternating** one *gains* the reciprocal, ``(1+rho)/(1-rho)``. That is why pulsed
    excitation is immune to 1/f noise and a slow SOC drift is not.
    """
    n = 4000
    dc = np.ones((n, 1, 1))
    alternating = ((-1.0) ** np.arange(n)).reshape(n, 1, 1)

    dc_ratio = float((whiten_ar1(dc, rho) ** 2).sum() / (dc**2).sum())
    alt_ratio = float((whiten_ar1(alternating, rho) ** 2).sum() / (alternating**2).sum())

    assert dc_ratio == pytest.approx((1.0 - rho) / (1.0 + rho), rel=0.02)
    assert alt_ratio == pytest.approx((1.0 + rho) / (1.0 - rho), rel=0.02)
    assert dc_ratio < 1.0 < alt_ratio
    assert dc_ratio * alt_ratio == pytest.approx(1.0, rel=0.05), "the pair is reciprocal"


def test_dc_standard_error_degrades_as_the_effective_sample_size_predicts() -> None:
    """A DC signature's standard error grows as ``sqrt((1+rho)/(1-rho))``.

    Equivalently, ``N_eff = N (1-rho)/(1+rho)``. This is the textbook effective-sample-size
    result, and it drops straight out of the exact GLS information ``1^T R^-1 1``.
    """
    n = 4000
    dc = np.ones((n, 1, 1))
    for rho in (0.5, 0.9, 0.95):
        info = float((whiten_ar1(dc, rho) ** 2).sum())
        assert np.sqrt(n / info) == pytest.approx(np.sqrt((1 + rho) / (1 - rho)), rel=0.02)


# ---------------------------------------------------------------------------
# Backwards compatibility: rho = 0 must reproduce the white-noise FIM exactly
# ---------------------------------------------------------------------------
def test_white_noise_fim_matches_the_flat_design_matrix_computation(setup) -> None:
    """``fisher_information`` now whitens per channel. With rho=0 it must agree, exactly,
    with the flattened ``S^T diag(1/sigma^2) S`` that ``design_matrix`` supports."""
    _, params, topology, duty = setup
    noise = NoiseModel()
    specs = local_specs(1)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)

    rows, kinds = design_matrix(sens, topology)
    variances = np.where(kinds == 0, noise.voltage_variance, noise.temp_variance)
    weighted = rows / np.sqrt(variances)[:, None]
    expected = weighted.T @ weighted

    np.testing.assert_allclose(fisher_information(sens, topology, noise), expected, rtol=1e-12)


def test_information_is_additive_across_independent_channels(setup) -> None:
    """The additivity the experiment planner relies on. Holds under correlated noise too,
    because the AR(1) structure is *within* a channel's time series, not across channels."""
    _, params, topology, duty = setup
    specs = local_specs(1)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)

    for rho in (0.0, 0.9):
        noise = NoiseModel(voltage_rho=rho, temp_rho=rho)
        voltage_only = topology.without_temp_sensors()
        temp_only = SensorTopology(
            topology.n_cells, voltage_cells=(), temp_cells=topology.temp_cells
        )

        total = fisher_information(sens, topology, noise)
        parts = fisher_information(sens, voltage_only, noise) + fisher_information(
            sens, temp_only, noise
        )
        np.testing.assert_allclose(total, parts, rtol=1e-10, atol=1e-12)


def test_correlated_noise_costs_capacity_far_more_than_resistance(setup) -> None:
    """The headline of the robustness pass, as a property rather than a number.

    R0's signature rides on the current pulses; capacity's is a near-DC SOC ramp. Under
    AR(1) noise the DC signature is annihilated and the pulsed one is not.
    """
    _, params, topology, duty = setup
    specs = with_current_bias(local_specs(1))

    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    white = np.sqrt(crlb(fisher_information(sens, topology, NoiseModel(), specs=specs)))
    corr_noise = NoiseModel(voltage_rho=0.9, temp_rho=0.9)
    corr = np.sqrt(crlb(fisher_information(sens, topology, corr_noise, specs=specs)))

    r0_cost = corr[0] / white[0]
    capacity_cost = corr[1] / white[1]
    assert capacity_cost > 2.0 * r0_cost, (
        f"expected capacity to suffer far more: R0 x{r0_cost:.2f}, capacity x{capacity_cost:.2f}"
    )


def test_capacity_degradation_is_monotone_in_rho_but_resistance_is_not(setup) -> None:
    """Correlated noise does not uniformly hurt. It reallocates.

    Capacity's signature is a slow SOC ramp -- near-DC -- so it pays the full
    ``sqrt((1+rho)/(1-rho))`` and pays it monotonically. R0's rides the pulse edges, which
    whitening *amplifies*, so R0 is non-monotone in rho.
    """
    _, params, topology, duty = setup
    specs = with_current_bias(local_specs(1))
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)

    def std(rho: float):
        noise = NoiseModel(voltage_rho=rho, temp_rho=rho)
        return np.sqrt(crlb(fisher_information(sens, topology, noise, specs=specs)))

    rhos = (0.0, 0.5, 0.9, 0.95, 0.99)
    capacity = [std(r)[1] for r in rhos]
    resistance = [std(r)[0] for r in rhos]

    assert capacity == sorted(capacity), f"capacity must degrade monotonically: {capacity}"
    assert resistance != sorted(resistance), "R0 is not monotone in rho"
    assert capacity[-1] / capacity[0] > 5.0, "capacity pays heavily"


def test_a_pulsed_signature_beats_white_noise_under_strong_correlation(setup) -> None:
    """The counterintuitive one, and the reason the naive 'correlated noise is bad'
    intuition fails: at rho = 0.99 the resistance bound is *tighter* than under white
    noise. Differencing destroys the noise faster than it destroys a pulsed signal."""
    _, params, topology, duty = setup
    specs = with_current_bias(local_specs(1))
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)

    def std(rho: float):
        noise = NoiseModel(voltage_rho=rho, temp_rho=rho)
        return np.sqrt(crlb(fisher_information(sens, topology, noise, specs=specs)))

    assert std(0.99)[0] < std(0.0)[0], "a pulsed R0 signature gains from 1/f noise"


def test_a_thermocouple_stops_carrying_cooling_information_under_strong_correlation(setup) -> None:
    """Why the instrumented/uninstrumented ordering can invert at extreme rho.

    A cooling fault's thermal signature evolves on the pack time constant (~200 s) and is
    essentially DC against a 1 Hz sampler, so whitening annihilates it. What survives is
    ``R0(T)``'s leak into the *voltage* channel, which rides the current pulses. Past some
    rho the thermocouple contributes nothing and cell position decides instead.
    """
    _, params, topology, duty = setup
    sensed = topology.temp_cells[0]
    specs = with_current_bias(local_specs(sensed))
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    temp_only = SensorTopology(topology.n_cells, voltage_cells=(), temp_cells=topology.temp_cells)
    ha = 2

    def temp_share(rho: float) -> float:
        noise = NoiseModel(voltage_rho=rho, temp_rho=rho)
        total = fisher_information(sens, topology, noise)[ha, ha]
        thermal = fisher_information(sens, temp_only, noise)[ha, ha]
        return float(thermal / total)

    assert temp_share(0.0) > 0.5, "with white noise the thermocouple dominates hA"
    assert temp_share(0.99) < 0.1, "with strong 1/f noise it contributes almost nothing"
    assert temp_share(0.99) < temp_share(0.9) < temp_share(0.0)


def test_rho_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="voltage_rho"):
        NoiseModel(voltage_rho=1.0)
    with pytest.raises(ValueError, match="temp_rho"):
        NoiseModel(temp_rho=-1.5)


# ---------------------------------------------------------------------------
# The current-bias nuisance parameter
# ---------------------------------------------------------------------------
def test_current_bias_spec_is_global_and_has_no_cell() -> None:
    assert CURRENT_BIAS_SPEC.is_global
    assert CURRENT_BIAS_SPEC.cell is None
    assert CURRENT_BIAS_SPEC.label() == "I_bias"
    assert CURRENT_BIAS_SPEC.unit() == "A"


def test_a_global_parameter_may_not_carry_a_cell_index() -> None:
    with pytest.raises(ValueError, match="pack-global"):
        ParameterSpec(3, ParamKind.CURRENT_BIAS)


def test_a_cell_parameter_must_carry_a_cell_index() -> None:
    with pytest.raises(ValueError, match="per-cell"):
        ParameterSpec(None, ParamKind.R0)


def test_perturb_refuses_a_global_parameter() -> None:
    params = nominal_pack(PackTopology(2, 4), seed=0)
    with pytest.raises(ValueError, match="global parameter"):
        perturb(params, CURRENT_BIAS_SPEC, 0.1)


def test_with_current_bias_is_idempotent() -> None:
    specs = local_specs(0)
    once = with_current_bias(specs)
    assert with_current_bias(once) is once
    assert len(once) == len(specs) + 1


def test_prior_information_only_regularises_the_nuisance_parameter() -> None:
    """We are not willing to assume a cell is healthy in order to conclude it is healthy."""
    specs = with_current_bias(local_specs(0))
    prior = prior_information(specs, NoiseModel(current_bias_sigma_a=2.0))
    assert prior.shape == (4, 4)
    np.testing.assert_allclose(np.diag(prior)[:3], 0.0)
    assert prior[3, 3] == pytest.approx(0.25)  # 1 / 2^2
    assert np.count_nonzero(prior) == 1


def test_current_bias_sensitivity_is_in_volts_per_amp(setup) -> None:
    """A 1 A current offset shifts each cell's voltage by roughly -(R0 + R1)."""
    _, params, _, duty = setup
    sens = sensitivities(params, duty.current_a, duty.dt_s, (CURRENT_BIAS_SPEC,))
    # At t=0 the RC branch is empty, so dV/db is exactly -R0.
    for cell in range(params.n_cells):
        assert sens[0, cell, 0, 0] == pytest.approx(-params.r0_ohm[cell], rel=1e-6)


def test_current_bias_is_cheap_under_pulsed_excitation(setup) -> None:
    """Prediction, verified: 32 voltage channels pin down a common-mode offset, so
    carrying it as unknown costs per-cell R0 essentially nothing."""
    _, params, topology, duty = setup
    noise = NoiseModel()
    plain = local_specs(1)
    nuisance = with_current_bias(plain)

    s0 = sensitivities(params, duty.current_a, duty.dt_s, plain)
    s1 = sensitivities(params, duty.current_a, duty.dt_s, nuisance)
    std0 = np.sqrt(crlb(fisher_information(s0, topology, noise, specs=plain)))
    std1 = np.sqrt(crlb(fisher_information(s1, topology, noise, specs=nuisance)))

    assert std1[0] / std0[0] < 1.02, "R0 should barely notice an unknown common-mode current"
    assert std1[3] < 0.1, "and the data should determine the bias far better than the 2 A prior"


def test_constant_current_makes_capacity_unisolatable_from_the_current_bias(setup) -> None:
    """The finding: a constant current offset and a capacity drift alias.

    Under pulsed excitation they separate. Under constant current the variance inflation
    factor crosses the conventional VIF>10 multicollinearity threshold, and the decision
    layer must refuse to *isolate* capacity even though something is plainly wrong.
    """
    _, params, topology, duty = setup
    noise = NoiseModel()
    specs = with_current_bias(local_specs(1))
    flat = constant_current(300.0, 1.0, c_rate=0.2)

    def capacity_vif(profile):
        sens = sensitivities(params, profile.current_a, profile.dt_s, specs)
        return float(variance_inflation(fisher_information(sens, topology, noise, specs=specs))[1])

    assert capacity_vif(duty) < 10.0, "pulsed excitation separates capacity from current bias"
    assert capacity_vif(flat) > 10.0, "constant current confounds them past the VIF threshold"
