"""Physics sanity checks, mostly property-based.

These are the highest-leverage tests in the package. A sensitivity is a difference
of two simulations; if the simulator quietly violates conservation of charge, the
Cramer-Rao bound built on top of it is a confident number about nothing.

Each test states a physical law and lets hypothesis try to break it.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from astracell.cell.ecm import r0_at_temperature
from astracell.cell.ocv import LFP_LIKE, NMC_LIKE, T_REF_K
from astracell.cell.thermal import irreversible_heat, reversible_heat
from astracell.duty import constant_current, pulse_train
from astracell.pack import PackTopology, SimulationError, nominal_pack, simulate

SLOW = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(scope="module")
def pack() -> PackTopology:
    return PackTopology(n_modules=2, cells_per_module=4)


# ---------------------------------------------------------------------------
# OCV
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("curve", [NMC_LIKE, LFP_LIKE], ids=lambda c: c.name)
def test_ocv_is_strictly_increasing_in_soc(curve) -> None:
    # Inside the clipping bounds: OcvCurve clips to [1e-3, 1-1e-3], so the curve is
    # flat outside by construction and diff() would legitimately be zero there.
    soc = np.linspace(1.5e-3, 1.0 - 1.5e-3, 2001)
    ocv = curve.ocv(soc)
    assert np.all(np.diff(ocv) > 0.0), "OCV must increase with state of charge"
    assert np.all(curve.docv_dsoc(soc) > 0.0), "analytic dOCV/dSOC must be positive"


@pytest.mark.parametrize("curve", [NMC_LIKE, LFP_LIKE], ids=lambda c: c.name)
def test_ocv_saturates_outside_the_clipping_bounds(curve) -> None:
    """Clipping is a guardrail, not a modelling choice: the simulator raises instead."""
    assert curve.ocv(-0.5) == pytest.approx(curve.ocv(0.0))
    assert curve.ocv(1.5) == pytest.approx(curve.ocv(1.0))


@pytest.mark.parametrize("curve", [NMC_LIKE, LFP_LIKE], ids=lambda c: c.name)
def test_analytic_docv_dsoc_matches_finite_difference(curve) -> None:
    soc = np.linspace(0.05, 0.95, 200)
    h = 1e-6
    numeric = (curve.ocv(soc + h) - curve.ocv(soc - h)) / (2 * h)
    np.testing.assert_allclose(curve.docv_dsoc(soc), numeric, rtol=1e-5)


def test_lfp_plateau_is_flatter_than_nmc() -> None:
    """The whole reason LFP is hard: a capacity fault barely moves the voltage.

    Restricted to the plateau (0.3-0.7 SOC). Above ~0.75 the synthetic LFP curve
    enters its upper knee and the comparison stops being about the plateau.
    """
    soc = np.linspace(0.3, 0.7, 100)
    assert LFP_LIKE.docv_dsoc(soc).max() < NMC_LIKE.docv_dsoc(soc).min()
    ratio = NMC_LIKE.docv_dsoc(0.5) / LFP_LIKE.docv_dsoc(0.5)
    assert ratio > 3.0, f"expected the plateau to be several times flatter, got {ratio:.1f}x"


@pytest.mark.parametrize("curve", [NMC_LIKE, LFP_LIKE], ids=lambda c: c.name)
def test_temperature_shifts_ocv_by_the_entropic_coefficient(curve) -> None:
    soc = np.linspace(0.1, 0.9, 50)
    delta_t = 10.0
    # A ~4 V OCV minus a ~4 V OCV to recover a ~0.2 mV shift: the absolute tolerance
    # is set by float64 cancellation at 4 V, not by anything physical.
    shifted = curve.ocv(soc, T_REF_K + delta_t) - curve.ocv(soc, T_REF_K)
    np.testing.assert_allclose(shifted, delta_t * curve.docv_dtemp(soc), rtol=1e-8, atol=1e-14)


# ---------------------------------------------------------------------------
# Resistance and heat
# ---------------------------------------------------------------------------
@given(temp_k=st.floats(253.15, 333.15))
def test_resistance_decreases_with_temperature(temp_k: float) -> None:
    r0 = np.array([1.5e-3])
    hot = r0_at_temperature(r0, np.array([temp_k + 5.0]))
    cold = r0_at_temperature(r0, np.array([temp_k]))
    assert hot[0] < cold[0], "Arrhenius: a warmer cell has lower ohmic resistance"


@given(current=st.floats(-200.0, 200.0), v_rc_scale=st.floats(0.0, 1.0))
def test_irreversible_heat_is_never_negative(current: float, v_rc_scale: float) -> None:
    """I^2*R0 + I*v1 >= 0 because v1 relaxes toward R1*I, sharing I's sign."""
    r0, r1 = np.array([1.5e-3]), np.array([0.8e-3])
    v_rc = v_rc_scale * r1 * current  # any point on the physical relaxation path
    assert irreversible_heat(current, r0, v_rc)[0] >= -1e-18


def test_reversible_heat_flips_sign_between_charge_and_discharge() -> None:
    """The entropic term is the reason charge and discharge heat differently."""
    temp = np.array([298.15])
    dudt = NMC_LIKE.docv_dtemp(np.array([0.5]))
    assert reversible_heat(+50.0, temp, dudt)[0] == pytest.approx(
        -reversible_heat(-50.0, temp, dudt)[0]
    )


# ---------------------------------------------------------------------------
# Simulator invariants
# ---------------------------------------------------------------------------
@SLOW
@given(c_rate=st.floats(0.05, 0.6), duration=st.floats(60.0, 900.0))
def test_coulomb_conservation(c_rate: float, duration: float) -> None:
    """dSOC must equal integrated charge divided by capacity.

    The SOC update is exact per step, so the only error is the float64 accumulation
    of ``n`` sequential subtractions against a single summation. That grows like
    ``n * eps``, hence rtol 1e-9 rather than 1e-15.
    """
    pack = PackTopology(2, 4)
    params = nominal_pack(pack, seed=None)
    duty = constant_current(duration, 1.0, c_rate)
    sim = simulate(params, duty.current_a, duty.dt_s, soc0=0.8)

    # The final recorded sample is the state *before* the last step is taken.
    charge_c = duty.current_a[:-1].sum() * duty.dt_s
    expected = charge_c / (params.capacity_ah * 3600.0)
    observed = sim.soc[0] - sim.soc[-1]
    np.testing.assert_allclose(observed, expected, rtol=1e-9)


def test_energy_balance_identity_holds_exactly() -> None:
    """I*(OCV - V) == I^2*R0 + I*v1, to machine precision, at every sample."""
    pack = PackTopology(2, 4)
    params = nominal_pack(pack, seed=0)
    duty = pulse_train(600.0, 1.0, mean_c_rate=0.3, pulse_c_rate=1.5)
    sim = simulate(params, duty.current_a, duty.dt_s)

    r0 = r0_at_temperature(params.r0_ohm, sim.temp_k, params.ea_over_r_k)
    ocv = params.curve.ocv(sim.soc, sim.temp_k)
    current = sim.current_a[:, None]

    overpotential = current * (ocv - sim.voltage_v)
    v_rc = ocv - sim.voltage_v - current * r0
    dissipation = current**2 * r0 + current * v_rc
    np.testing.assert_allclose(overpotential, dissipation, atol=1e-14)


def test_at_rest_the_pack_relaxes_monotonically_to_the_coolant_temperature() -> None:
    """Second law, in the form we can check: no current, no heat, only cooling."""
    pack = PackTopology(2, 4)
    params = nominal_pack(pack, seed=0)
    n = 900
    current = np.zeros(n)
    hot = params.coolant_temp_k + 12.0
    sim = simulate(params, current, 1.0, temp0_k=hot)

    cooling = np.diff(sim.temp_k, axis=0)
    assert np.all(cooling <= 1e-12), "a resting pack above coolant temperature must only cool"
    assert np.all(sim.temp_k[-1] < sim.temp_k[0])
    assert np.all(sim.temp_k[-1] > params.coolant_temp_k), "it approaches, never crosses"


def test_at_rest_voltage_equals_ocv() -> None:
    params = nominal_pack(PackTopology(2, 4), seed=0)
    sim = simulate(params, np.zeros(120), 1.0)
    np.testing.assert_allclose(
        sim.voltage_v[-1], params.curve.ocv(sim.soc[-1], sim.temp_k[-1]), atol=1e-12
    )


@SLOW
@given(c_rate=st.floats(0.1, 1.0))
def test_discharge_voltage_never_exceeds_ocv(c_rate: float) -> None:
    params = nominal_pack(PackTopology(2, 4), seed=0)
    duty = constant_current(600.0, 1.0, c_rate)
    sim = simulate(params, duty.current_a, duty.dt_s)
    ocv = params.curve.ocv(sim.soc, sim.temp_k)
    assert np.all(sim.voltage_v <= ocv + 1e-12), "discharge must sit below OCV"


def test_discharging_pack_warms_up() -> None:
    params = nominal_pack(PackTopology(2, 4), seed=0)
    duty = constant_current(900.0, 1.0, c_rate=1.0)
    sim = simulate(params, duty.current_a, duty.dt_s)
    assert np.all(sim.temp_k[-1] > sim.temp_k[0])


def test_simulator_raises_rather_than_clipping_out_of_range_soc() -> None:
    """Silently clipping SOC would put a kink in the sensitivities."""
    params = nominal_pack(PackTopology(2, 4), seed=0)
    duty = constant_current(3600.0, 1.0, c_rate=1.0)  # 1C for an hour from 75%
    with pytest.raises(SimulationError, match="SOC left"):
        simulate(params, duty.current_a, duty.dt_s, soc0=0.75)


# ---------------------------------------------------------------------------
# Conduction network
# ---------------------------------------------------------------------------
def test_conductance_matrix_is_a_laplacian(pack: PackTopology) -> None:
    lap = pack.conductance_matrix(0.6, 0.15)
    np.testing.assert_allclose(lap, lap.T, atol=1e-15)
    np.testing.assert_allclose(lap.sum(axis=1), 0.0, atol=1e-12)
    eigvals = np.linalg.eigvalsh(lap)
    assert eigvals.min() > -1e-12, "conduction Laplacian must be positive semi-definite"


def test_an_isothermal_pack_conducts_no_heat_internally(pack: PackTopology) -> None:
    lap = pack.conductance_matrix(0.6, 0.15)
    isothermal = np.full(pack.n_cells, 305.0)
    np.testing.assert_allclose(lap @ isothermal, 0.0, atol=1e-11)
