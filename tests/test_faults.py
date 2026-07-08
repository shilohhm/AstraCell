"""Fault injection: correctness, and above all immutability.

The ground-truth ``PackParams`` is what every identifiability claim is measured
against. If an injector could mutate it in place, a stale reference would silently
become a wrong answer, and the failure would look like a scientific result rather
than a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from astracell.faults import (
    PhysicalFault,
    PhysicalFaultKind,
    apply_physical_faults,
    apply_sensor_faults,
    cooling_weakness,
    high_internal_resistance,
    reduced_capacity,
    temp_sensor_bias,
    voltage_sensor_bias,
)
from astracell.pack import PackTopology, nominal_pack, simulate
from astracell.sensors import NoiseModel, measure
from astracell.sensors.topology import realistic_topology


@pytest.fixture
def params():
    return nominal_pack(PackTopology(2, 4), seed=0)


# ---------------------------------------------------------------------------
# Magnitude conventions
# ---------------------------------------------------------------------------
def test_high_resistance_raises_r0_and_touches_nothing_else(params) -> None:
    faulted = apply_physical_faults(params, [high_internal_resistance(3, 0.20)])
    assert faulted.r0_ohm[3] == pytest.approx(params.r0_ohm[3] * 1.20)
    np.testing.assert_array_equal(np.delete(faulted.r0_ohm, 3), np.delete(params.r0_ohm, 3))
    np.testing.assert_array_equal(faulted.capacity_ah, params.capacity_ah)
    np.testing.assert_array_equal(faulted.ha_w_per_k, params.ha_w_per_k)


def test_reduced_capacity_lowers_capacity(params) -> None:
    faulted = apply_physical_faults(params, [reduced_capacity(1, 0.05)])
    assert faulted.capacity_ah[1] == pytest.approx(params.capacity_ah[1] * 0.95)


def test_cooling_weakness_lowers_ha(params) -> None:
    faulted = apply_physical_faults(params, [cooling_weakness(2, 0.40)])
    assert faulted.ha_w_per_k[2] == pytest.approx(params.ha_w_per_k[2] * 0.60)


def test_faults_on_the_same_cell_compose_multiplicatively(params) -> None:
    faulted = apply_physical_faults(
        params, [high_internal_resistance(0, 0.20), high_internal_resistance(0, 0.10)]
    )
    assert faulted.r0_ohm[0] == pytest.approx(params.r0_ohm[0] * 1.20 * 1.10)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------
def test_injection_never_mutates_the_original(params) -> None:
    snapshot = {
        field: np.array(getattr(params, field), copy=True)
        for field in ("r0_ohm", "capacity_ah", "ha_w_per_k")
    }
    apply_physical_faults(
        params,
        [high_internal_resistance(0, 0.5), reduced_capacity(1, 0.3), cooling_weakness(2, 0.6)],
    )
    for field, before in snapshot.items():
        np.testing.assert_array_equal(getattr(params, field), before)


def test_injection_returns_a_distinct_object(params) -> None:
    faulted = apply_physical_faults(params, [high_internal_resistance(0, 0.2)])
    assert faulted is not params
    assert faulted.r0_ohm is not params.r0_ohm


def test_applying_no_faults_is_the_identity(params) -> None:
    assert apply_physical_faults(params, []) is params


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("magnitude", [-0.1, 1.0, 1.5])
def test_magnitude_must_lie_in_the_unit_interval(magnitude: float) -> None:
    with pytest.raises(ValueError, match="magnitude"):
        PhysicalFault(PhysicalFaultKind.HIGH_R, 0, magnitude)


def test_a_fault_outside_the_pack_raises(params) -> None:
    with pytest.raises(IndexError, match="pack has"):
        apply_physical_faults(params, [high_internal_resistance(99, 0.2)])


def test_a_sensor_bias_outside_the_channel_count_raises(params) -> None:
    pack = PackTopology(2, 4)
    topology = realistic_topology(pack, n_temp_sensors=2)
    meas = measure(
        simulate(params, np.zeros(20), 1.0), topology, NoiseModel(), np.random.default_rng(0)
    )
    with pytest.raises(IndexError, match="temperature channels"):
        apply_sensor_faults(meas, [temp_sensor_bias(channel=7, bias_k=2.0)])


# ---------------------------------------------------------------------------
# Sensor faults: the pack is healthy, the number is wrong
# ---------------------------------------------------------------------------
def test_sensor_bias_shifts_exactly_one_channel(params) -> None:
    pack = PackTopology(2, 4)
    topology = realistic_topology(pack, n_temp_sensors=2)
    sim = simulate(params, np.full(50, 20.0), 1.0)
    clean = measure(sim, topology, NoiseModel(), np.random.default_rng(0))

    biased = apply_sensor_faults(clean, [voltage_sensor_bias(2, 5e-3), temp_sensor_bias(1, 2.0)])

    np.testing.assert_allclose(biased.voltage_v[:, 2] - clean.voltage_v[:, 2], 5e-3)
    np.testing.assert_allclose(biased.temp_k[:, 1] - clean.temp_k[:, 1], 2.0)
    for channel in range(topology.n_voltage):
        if channel != 2:
            np.testing.assert_array_equal(biased.voltage_v[:, channel], clean.voltage_v[:, channel])
    np.testing.assert_array_equal(biased.temp_k[:, 0], clean.temp_k[:, 0])


def test_sensor_faults_do_not_mutate_the_measurements(params) -> None:
    pack = PackTopology(2, 4)
    topology = realistic_topology(pack, n_temp_sensors=2)
    clean = measure(
        simulate(params, np.full(30, 10.0), 1.0), topology, NoiseModel(), np.random.default_rng(0)
    )
    snapshot = np.array(clean.voltage_v, copy=True)
    apply_sensor_faults(clean, [voltage_sensor_bias(0, 0.1)])
    np.testing.assert_array_equal(clean.voltage_v, snapshot)


def test_a_sensor_bias_leaves_the_physics_untouched(params) -> None:
    """The distinction that makes diagnosis hard: same reading, different world."""
    pack = PackTopology(2, 4)
    topology = realistic_topology(pack, n_temp_sensors=2)
    sim = simulate(params, np.full(60, 30.0), 1.0)
    rng = np.random.default_rng(0)

    biased = apply_sensor_faults(
        measure(sim, topology, NoiseModel(), rng), [temp_sensor_bias(0, 3.0)]
    )
    hot_cell = apply_physical_faults(params, [cooling_weakness(topology.temp_cells[0], 0.4)])
    hot_sim = simulate(hot_cell, np.full(60, 30.0), 1.0)

    # The biased sensor reports a hotter cell that is not hot; the real fault makes
    # a cell that *is* hot. Only the second changes the true temperature field.
    assert (
        biased.temp_k[-1, 0]
        > measure(sim, topology, NoiseModel(), np.random.default_rng(0)).temp_k[-1, 0]
    )
    assert hot_sim.temp_k[-1, topology.temp_cells[0]] > sim.temp_k[-1, topology.temp_cells[0]]
    np.testing.assert_array_equal(sim.temp_k, simulate(params, np.full(60, 30.0), 1.0).temp_k)
