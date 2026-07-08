"""Sensor topology and noise."""

from __future__ import annotations

import numpy as np
import pytest

from astracell.pack import PackTopology, nominal_pack, simulate
from astracell.sensors import NoiseModel, measure
from astracell.sensors.topology import (
    SensorTopology,
    evenly_spaced_temp_sensors,
    realistic_topology,
)


@pytest.fixture
def pack() -> PackTopology:
    return PackTopology(n_modules=4, cells_per_module=8)


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
def test_realistic_topology_has_one_voltage_channel_per_cell_and_few_thermocouples(pack) -> None:
    topology = realistic_topology(pack, n_temp_sensors=4)
    assert topology.n_voltage == pack.n_cells
    assert topology.n_temp == 4
    assert topology.n_temp < topology.n_voltage / 4, "the whole premise of the package"


@pytest.mark.parametrize("n_sensors", [1, 2, 3, 4, 8, 16])
def test_evenly_spaced_sensors_are_distinct_and_in_range(pack, n_sensors: int) -> None:
    cells = evenly_spaced_temp_sensors(pack, n_sensors)
    assert len(cells) == len(set(cells))
    assert all(0 <= c < pack.n_cells for c in cells)


def test_sensors_spread_across_modules_before_doubling_up(pack) -> None:
    """Modules are the thermally weak axis; each needs its own thermocouple first."""
    cells = evenly_spaced_temp_sensors(pack, pack.n_modules)
    modules = {pack.coords(c)[0] for c in cells}
    assert modules == set(range(pack.n_modules))


def test_zero_sensors_is_allowed(pack) -> None:
    assert evenly_spaced_temp_sensors(pack, 0) == ()


def test_too_many_sensors_raises(pack) -> None:
    with pytest.raises(ValueError, match="cannot place"):
        evenly_spaced_temp_sensors(pack, pack.n_cells + 1)


def test_duplicate_channels_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        SensorTopology(8, voltage_cells=(0, 0, 1), temp_cells=())


def test_out_of_range_channels_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside the pack"):
        SensorTopology(8, voltage_cells=(0, 99), temp_cells=())


def test_adding_a_sensor_is_idempotent_and_non_mutating(pack) -> None:
    topology = realistic_topology(pack, n_temp_sensors=4)
    existing = topology.temp_cells[0]
    assert topology.with_temp_sensor_at(existing) is topology

    grown = topology.with_temp_sensor_at(1)
    assert grown.n_temp == topology.n_temp + 1
    assert topology.n_temp == 4, "the original topology must be unchanged"
    assert 1 in grown.temp_cells


def test_without_temp_sensors_keeps_voltage_channels(pack) -> None:
    blind = realistic_topology(pack, n_temp_sensors=4).without_temp_sensors()
    assert blind.n_temp == 0
    assert blind.n_voltage == pack.n_cells


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
def test_total_variance_is_gaussian_plus_quantisation() -> None:
    noise = NoiseModel(voltage_sigma_v=1e-3, voltage_lsb_v=100e-6)
    assert noise.voltage_variance == pytest.approx(1e-6 + (100e-6) ** 2 / 12.0)


def test_channel_variances_are_ordered_voltages_then_temperatures(pack) -> None:
    topology = realistic_topology(pack, n_temp_sensors=4)
    noise = NoiseModel()
    variances = noise.channel_variances(topology)
    assert variances.shape == (topology.n_channels,)
    np.testing.assert_allclose(variances[: topology.n_voltage], noise.voltage_variance)
    np.testing.assert_allclose(variances[topology.n_voltage :], noise.temp_variance)


def test_measurement_noise_matches_the_specified_standard_deviation(pack) -> None:
    """The sensor model must actually deliver the sigma it advertises."""
    params = nominal_pack(pack, seed=0)
    topology = realistic_topology(pack, n_temp_sensors=4)
    # No quantisation, so the residual is pure Gaussian and the check is sharp.
    noise = NoiseModel(voltage_sigma_v=2e-3, voltage_lsb_v=0.0, temp_sigma_k=0.5, temp_lsb_k=0.0)

    sim = simulate(params, np.zeros(4000), 1.0)  # rest: the true signal is constant
    meas = measure(sim, topology, noise, np.random.default_rng(0))

    v_residual = meas.voltage_v - sim.voltage_v[:, topology.voltage_index]
    t_residual = meas.temp_k - sim.temp_k[:, topology.temp_index]

    assert v_residual.std() == pytest.approx(noise.voltage_sigma_v, rel=0.05)
    assert t_residual.std() == pytest.approx(noise.temp_sigma_k, rel=0.05)
    assert abs(v_residual.mean()) < 0.1 * noise.voltage_sigma_v


def test_quantisation_snaps_to_the_lsb_grid(pack) -> None:
    params = nominal_pack(pack, seed=0)
    topology = realistic_topology(pack, n_temp_sensors=2)
    noise = NoiseModel(voltage_sigma_v=0.0, voltage_lsb_v=1e-3, temp_sigma_k=0.0, temp_lsb_k=0.0)
    meas = measure(simulate(params, np.zeros(10), 1.0), topology, noise, np.random.default_rng(0))
    remainder = np.abs(meas.voltage_v / 1e-3 - np.round(meas.voltage_v / 1e-3))
    assert remainder.max() < 1e-9


def test_measure_rejects_a_topology_of_the_wrong_size(pack) -> None:
    params = nominal_pack(pack, seed=0)
    wrong = realistic_topology(PackTopology(2, 4), n_temp_sensors=2)
    with pytest.raises(ValueError, match="cells"):
        measure(simulate(params, np.zeros(10), 1.0), wrong, NoiseModel(), np.random.default_rng(0))


def test_measurements_carry_the_right_shapes(pack) -> None:
    params = nominal_pack(pack, seed=0)
    topology = realistic_topology(pack, n_temp_sensors=4)
    meas = measure(
        simulate(params, np.zeros(37), 1.0), topology, NoiseModel(), np.random.default_rng(0)
    )
    assert meas.voltage_v.shape == (37, pack.n_cells)
    assert meas.temp_k.shape == (37, 4)
    assert meas.as_vector().size == 37 * topology.n_channels
