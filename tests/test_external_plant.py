"""The external-plant gate: does calibrated abstention survive a plant AstraCell did not write?

Split deliberately into two halves:

* Tests that need no PyBaMM -- the OCV-table builder and the *self-consistency control*. The
  control is the load-bearing one: it feeds the external-plant pipeline an ECM-generated trace,
  where the model mismatch is exactly zero, and asserts coverage is nominal. If that holds, any
  undercoverage in the PyBaMM half is model mismatch and not a bug in the plumbing. It runs
  everywhere, so the guarantee does not depend on an optional dependency being installed.

* Tests that need PyBaMM -- ``pytest.importorskip`` skips them cleanly when it is absent, which is
  the contract for an optional dependency. They pin the actual v0.3 result: a healthy cell whose
  variance-only interval overclaims, and a bias gate that refuses instead.

PyBaMM's solver emits RuntimeWarnings that the repo's ``filterwarnings = error`` would otherwise
promote to failures; ``pytestmark`` scopes an ignore to this module only.
"""

from __future__ import annotations

import numpy as np
import pytest

from astracell.calibration import (
    abstention_metrics,
    build_external_observer,
    coverage,
    coverage_curve,
    external_scenario,
    observer_voltage,
    prepare_external,
    pulse_profile,
    run_trials,
)
from astracell.calibration.metrics import NOMINAL_LEVELS
from astracell.cell.ocv import NMC_LIKE, ocv_from_table
from astracell.pack.params import PackParams
from astracell.plant import pybamm_pseudo_ocv, simulate_pybamm_cell

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

N_TIME = 300
CAPACITY_AH = 5.0
SOC0 = 0.9


def _money_current() -> np.ndarray:
    return pulse_profile(CAPACITY_AH, mean_c_rate=0.5, pulse_c_rate=1.0, n_time=N_TIME)


# --------------------------------------------------------------------- no PyBaMM: the OCV builder


def test_ocv_from_table_interpolates_and_differentiates_a_known_curve() -> None:
    # A known smooth curve sampled densely: the table's value and slope must match the analytic
    # ones. This is what lets an external pseudo-OCV stand in for the analytic curves elsewhere.
    soc = np.linspace(0.05, 0.98, 400)
    truth = 3.5 + 0.6 * soc - 0.2 * soc**2
    curve = ocv_from_table(soc, truth, name="t", chemistry="synthetic")

    probe = np.array([0.2, 0.5, 0.85])
    np.testing.assert_allclose(curve.ocv(probe), 3.5 + 0.6 * probe - 0.2 * probe**2, atol=1e-4)
    np.testing.assert_allclose(curve.docv_dsoc(probe), 0.6 - 0.4 * probe, atol=2e-3)


def test_ocv_from_table_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        ocv_from_table(np.array([0.1, 0.2]), np.array([3.5, 3.6]), name="t", chemistry="c")
    with pytest.raises(ValueError):
        # Non-monotone SOC after sorting (a duplicate) is ambiguous and must be refused.
        soc = np.array([0.1, 0.2, 0.2, 0.4, 0.5])
        ocv_from_table(soc, np.linspace(3.5, 4.0, 5), name="t", chemistry="c")


# --------------------------------------------------- no PyBaMM: the self-consistency control


def test_the_external_harness_is_self_consistent_on_an_ecm_plant() -> None:
    # The anti-harness-bug control. Plant == observer (an ECM feeding itself), so the structural
    # bias is identically zero and the linear fit is exactly N(0, CRLB). Coverage must be nominal;
    # if it is not, the external-plant path itself is broken and the PyBaMM numbers are worthless.
    observer = build_external_observer(NMC_LIKE, CAPACITY_AH)
    scenario = external_scenario(
        name="control", observer=observer, current_a=_money_current(), soc0=SOC0
    )
    prepared = prepare_external(scenario, observer_voltage(scenario))

    assert abs(prepared.target_bias) < 1e-9  # an ECM cannot mismatch itself
    result = run_trials(scenario, 3000, seed=0, estimator="linear", prepared=prepared)
    for level, got in zip(NOMINAL_LEVELS, coverage_curve(result), strict=True):
        assert abs(got - level) < 0.03


# --------------------------------------------------------------------- PyBaMM: the real thing


@pytest.fixture(scope="module")
def observer() -> PackParams:
    """PyBaMM's pseudo-OCV and the one-cell ECM that shares it, built once for the module."""
    pytest.importorskip("pybamm")
    soc, ocv = pybamm_pseudo_ocv(model="SPMe", n_points=150)
    curve = ocv_from_table(soc, ocv, name="pybamm", chemistry="SPMe/Chen2020")
    return build_external_observer(curve, CAPACITY_AH)


@pytest.mark.pybamm
def test_pybamm_pseudo_ocv_is_monotone_and_in_range() -> None:
    pytest.importorskip("pybamm")
    soc, ocv = pybamm_pseudo_ocv(model="SPMe", n_points=100)
    assert soc.shape == ocv.shape == (100,)
    assert np.all(np.diff(soc) > 0.0)
    assert np.all(np.diff(ocv) > 0.0)  # OCV rises with SOC, monotonically
    assert ocv.min() > 2.5 and ocv.max() < 4.3  # a sane Li-ion window


@pytest.mark.pybamm
def test_pybamm_cell_produces_voltage_of_expected_shape_and_units(observer: PackParams) -> None:
    plant = simulate_pybamm_cell(_money_current(), 1.0, model="SPMe", soc0=SOC0)
    assert plant.voltage_v.shape == (N_TIME,)
    assert np.all(np.isfinite(plant.voltage_v))
    assert np.all((plant.voltage_v > 2.5) & (plant.voltage_v < 4.3))  # volts, not millivolts
    assert plant.capacity_ah == pytest.approx(CAPACITY_AH, rel=0.01)
    assert plant.dt_s == pytest.approx(1.0)


@pytest.mark.pybamm
def test_the_ecm_fits_the_external_plant_and_the_bias_is_real(observer: PackParams) -> None:
    plant = simulate_pybamm_cell(_money_current(), 1.0, model="SPMe", soc0=SOC0)
    scenario = external_scenario(
        name="pybamm", observer=observer, current_a=_money_current(), soc0=SOC0
    )
    prepared = prepare_external(scenario, plant.voltage_v)
    result = run_trials(scenario, 100, seed=0, estimator="linear", prepared=prepared)

    assert np.all(np.isfinite(result.delta_hat))
    assert np.isfinite(result.crlb_std) and result.crlb_std > 0.0
    # A healthy cell, yet the ECM reports a large capacity deviation: the diffusion it cannot model.
    assert abs(result.bias) > 0.10


@pytest.mark.pybamm
def test_the_external_plant_undercovers_the_variance_only_interval(observer: PackParams) -> None:
    plant = simulate_pybamm_cell(_money_current(), 1.0, model="SPMe", soc0=SOC0)
    scenario = external_scenario(
        name="pybamm", observer=observer, current_a=_money_current(), soc0=SOC0
    )
    prepared = prepare_external(scenario, plant.voltage_v)
    result = run_trials(scenario, 200, seed=0, estimator="linear", prepared=prepared)

    variance_only = coverage(result, 0.95, bias_aware=False)
    bias_aware = coverage(result, 0.95, bias_aware=True)
    assert variance_only < 0.10  # the 95% interval almost never covers a healthy cell's zero
    assert bias_aware > variance_only  # admitting the bias widens it back over the truth


@pytest.mark.pybamm
def test_the_bias_gate_reduces_overclaim_against_the_external_plant(observer: PackParams) -> None:
    plant = simulate_pybamm_cell(_money_current(), 1.0, model="SPMe", soc0=SOC0)
    gated_scn = external_scenario(
        name="g", observer=observer, current_a=_money_current(), soc0=SOC0
    )
    gated_prep = prepare_external(gated_scn, plant.voltage_v)
    ungated_scn = external_scenario(
        name="u", observer=observer, current_a=_money_current(), soc0=SOC0, use_bias_gate=False
    )
    ungated_prep = prepare_external(ungated_scn, plant.voltage_v)

    gated = abstention_metrics(run_trials(gated_scn, 200, seed=0, prepared=gated_prep))
    ungated = abstention_metrics(run_trials(ungated_scn, 200, seed=0, prepared=ungated_prep))

    assert ungated.harmful_overclaim_rate > 0.5  # variance-only diagnoses a fault that is not there
    assert gated.harmful_overclaim_rate < ungated.harmful_overclaim_rate
    assert gated.harmful_overclaim_rate < 0.05  # the gate all but eliminates it
    assert gated.refuse_model_bias_rate > 0.5  # by refusing, which is the honest move
