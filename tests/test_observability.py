"""Tests for the identifiability machinery.

These assert **theorems**, not observed numbers. A test that hard-codes "the SNR is
149" is a test of the OCV curve, not of the code, and it will break the moment
someone replaces the stand-in curves with real ones -- which is the whole plan.

The two exceptions are marked ``regression`` and say so out loud.
"""

from __future__ import annotations

import numpy as np
import pytest

from astracell.cell.ecm import r0_at_temperature
from astracell.duty import constant_current, pulse_train
from astracell.observability.detectability import detectability_heatmap
from astracell.observability.fisher import (
    condition_number,
    crlb,
    design_matrix,
    fisher_information,
    information_gain,
    variance_inflation,
)
from astracell.observability.mask import (
    Observability,
    classify,
    detection_snr,
    grey_cell_map,
    recommend_temp_sensor,
)
from astracell.observability.sensitivity import (
    TEMPERATURE_CHANNEL,
    VOLTAGE_CHANNEL,
    ParameterSpec,
    ParamKind,
    all_specs,
    local_specs,
    perturb,
    sensitivities,
)
from astracell.pack import PackTopology, nominal_pack
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import SensorTopology, realistic_topology


@pytest.fixture(scope="module")
def setup():
    pack = PackTopology(n_modules=2, cells_per_module=4)
    params = nominal_pack(pack, seed=0)
    duty = pulse_train(300.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0)
    topology = realistic_topology(pack, n_temp_sensors=2)
    return pack, params, duty, topology, NoiseModel()


@pytest.fixture(scope="module")
def sens_all(setup):
    pack, params, duty, _, _ = setup
    specs = all_specs(pack.n_cells)
    return specs, sensitivities(params, duty.current_a, duty.dt_s, specs)


@pytest.fixture(scope="module")
def demo_setup():
    """A pack with enough thermal information for the headline results to exist.

    The small ``setup`` fixture is deliberately information-starved -- 300 samples,
    two thermocouples -- so that even an instrumented cell sits near 1 sigma. That is
    fine for testing structural theorems (which hold at any information level) but it
    cannot exhibit the headline behaviour, because at that excitation *nothing* is
    observable. This fixture matches ``examples/01_first_demo.py``.

    Costs ~6 s of simulation, computed once for the whole module.
    """
    pack = PackTopology(n_modules=4, cells_per_module=8)
    params = nominal_pack(pack, seed=0)
    duty = pulse_train(1200.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0)
    topology = realistic_topology(pack, n_temp_sensors=4)
    noise = NoiseModel()
    specs = all_specs(pack.n_cells)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    return pack, params, duty, topology, noise, specs, sens


# ---------------------------------------------------------------------------
# Sensitivity: check the finite differences against closed form where one exists
# ---------------------------------------------------------------------------
def test_initial_voltage_sensitivity_to_r0_is_exactly_minus_i_times_r0(setup) -> None:
    """At t=0 the RC branch is empty and T equals the reference, so V = OCV - I*R0.

    The derivative wrt a *relative* R0 perturbation is therefore exactly -I*R0.
    Voltage is linear in that perturbation, so a central difference recovers it to
    machine precision. If this test fails, the sensitivity tensor is wired wrong.
    """
    pack, params, duty, _, _ = setup
    specs = tuple(ParameterSpec(c, ParamKind.R0) for c in range(pack.n_cells))
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)

    i0 = duty.current_a[0]
    for cell in range(pack.n_cells):
        observed = sens[0, cell, VOLTAGE_CHANNEL, cell]
        expected = -i0 * params.r0_ohm[cell]
        assert observed == pytest.approx(expected, rel=1e-9)


def test_cell_voltages_are_independent_at_the_first_sample(setup) -> None:
    """A series string shares current, so at t=0 cell i's R0 cannot move cell j's V."""
    pack, params, duty, _, _ = setup
    specs = tuple(ParameterSpec(c, ParamKind.R0) for c in range(pack.n_cells))
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    block = sens[0, :, VOLTAGE_CHANNEL, :]
    off_diagonal = block - np.diag(np.diag(block))
    assert np.abs(off_diagonal).max() < 1e-15


def test_temperature_at_the_first_sample_is_the_initial_condition(setup) -> None:
    """Nothing can perturb a state before the first step is taken."""
    _, params, duty, _, _ = setup
    specs = local_specs(1)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    assert np.abs(sens[0, :, TEMPERATURE_CHANNEL, :]).max() == 0.0


def test_capacity_does_not_move_voltage_at_the_first_sample(setup) -> None:
    _, params, duty, _, _ = setup
    specs = (ParameterSpec(0, ParamKind.CAPACITY),)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    assert np.abs(sens[0, :, VOLTAGE_CHANNEL, 0]).max() < 1e-15


def test_perturb_does_not_mutate_the_original(setup) -> None:
    _, params, _, _, _ = setup
    before = np.array(params.r0_ohm, copy=True)
    perturbed = perturb(params, ParameterSpec(2, ParamKind.R0), 0.5)
    np.testing.assert_array_equal(params.r0_ohm, before)
    assert perturbed.r0_ohm[2] == pytest.approx(before[2] * 1.5)


def test_pack_params_arrays_are_read_only(setup) -> None:
    _, params, _, _, _ = setup
    with pytest.raises(ValueError):
        params.r0_ohm[0] = 1.0


# ---------------------------------------------------------------------------
# Fisher information: structural properties
# ---------------------------------------------------------------------------
def test_fisher_information_is_symmetric_and_positive_semidefinite(setup, sens_all) -> None:
    _, _, _, topology, noise = setup
    _, sens = sens_all
    fim = fisher_information(sens, topology, noise)
    np.testing.assert_allclose(fim, fim.T, atol=0.0)
    eigvals = np.linalg.eigvalsh(fim)
    assert eigvals.min() > -1e-6 * max(eigvals.max(), 1.0)


def test_design_matrix_has_one_row_per_sample_per_channel(setup, sens_all) -> None:
    _, _, duty, topology, _ = setup
    specs, sens = sens_all
    rows, kinds = design_matrix(sens, topology)
    assert rows.shape == (duty.n_samples * topology.n_channels, len(specs))
    assert np.count_nonzero(kinds == 0) == duty.n_samples * topology.n_voltage
    assert np.count_nonzero(kinds == 1) == duty.n_samples * topology.n_temp


def test_adding_a_sensor_never_decreases_information(setup, sens_all) -> None:
    """Loewner monotonicity: FIM(more sensors) - FIM(fewer) is positive semi-definite.

    This is a theorem, not an empirical observation. A violation means the design
    matrix is being assembled wrong.
    """
    _, _, _, topology, noise = setup
    _, sens = sens_all
    before = fisher_information(sens, topology, noise)
    after = fisher_information(sens, topology.with_temp_sensor_at(1), noise)
    eigvals = np.linalg.eigvalsh(after - before)
    scale = max(np.abs(before).max(), 1.0)
    assert eigvals.min() > -1e-9 * scale


def test_adding_a_sensor_never_increases_the_cramer_rao_bound(setup, sens_all) -> None:
    """The corollary that actually matters: more data cannot make you less certain."""
    _, _, _, topology, noise = setup
    _, sens = sens_all
    before = crlb(fisher_information(sens, topology, noise))
    after = crlb(fisher_information(sens, topology.with_temp_sensor_at(1), noise))

    assert not np.any(np.isfinite(before) & ~np.isfinite(after)), "a bound became infinite"
    both = np.isfinite(before) & np.isfinite(after)
    assert np.all(after[both] <= before[both] * (1.0 + 1e-6))


def test_information_gain_from_adding_a_sensor_is_non_negative(setup, sens_all) -> None:
    _, _, _, topology, noise = setup
    _, sens = sens_all
    before = fisher_information(sens, topology, noise)
    after = fisher_information(sens, topology.with_temp_sensor_at(1), noise)
    assert information_gain(before, after) >= -1e-9


def test_crlb_reports_infinity_for_a_parameter_with_no_information() -> None:
    """A pinv would return a finite, min-norm variance here. That would be a lie."""
    fim = np.diag([4.0, 0.0])
    variance = crlb(fim)
    assert variance[0] == pytest.approx(0.25)
    assert np.isinf(variance[1])


def test_crlb_reports_infinity_for_perfectly_collinear_parameters() -> None:
    """Two parameters with identical sensitivity vectors are individually unidentified."""
    s = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    fim = s.T @ s
    assert np.all(np.isinf(crlb(fim)))


def test_crlb_of_a_diagonal_fim_is_its_reciprocal() -> None:
    fim = np.diag([2.0, 8.0, 50.0])
    np.testing.assert_allclose(crlb(fim), [0.5, 0.125, 0.02], rtol=1e-12)


def test_variance_inflation_is_one_for_an_orthogonal_design() -> None:
    """VIF == 1 exactly when a parameter is uncorrelated with every other."""
    np.testing.assert_allclose(variance_inflation(np.diag([3.0, 7.0])), [1.0, 1.0], rtol=1e-12)


def test_variance_inflation_is_never_below_one(setup, sens_all) -> None:
    """Confounding can only inflate a variance, never deflate it."""
    _, _, _, topology, noise = setup
    _, sens = sens_all
    vif = variance_inflation(fisher_information(sens, topology, noise))
    assert np.all(vif >= 1.0 - 1e-9)


def test_variance_inflation_grows_with_correlation() -> None:
    weak = np.array([[1.0, 0.1], [0.1, 1.0]])
    strong = np.array([[1.0, 0.99], [0.99, 1.0]])
    assert variance_inflation(strong)[0] > variance_inflation(weak)[0] > 1.0


def test_condition_number_of_the_identity_is_one() -> None:
    assert condition_number(np.eye(4)) == pytest.approx(1.0)


def test_empty_sensor_topology_yields_zero_information(setup, sens_all) -> None:
    pack, _, _, _, noise = setup
    specs, sens = sens_all
    blind = SensorTopology(pack.n_cells, voltage_cells=(), temp_cells=())
    fim = fisher_information(sens, blind, noise)
    assert fim.shape == (len(specs), len(specs))
    assert np.all(fim == 0.0)
    assert np.all(np.isinf(crlb(fim)))


# ---------------------------------------------------------------------------
# SNR and classification
# ---------------------------------------------------------------------------
def test_detection_snr_is_linear_in_fault_magnitude() -> None:
    variance = np.array([0.01, 0.04])
    np.testing.assert_allclose(detection_snr(variance, 0.2), 2 * detection_snr(variance, 0.1))


def test_detection_snr_of_an_unidentified_parameter_is_zero() -> None:
    assert detection_snr(np.array([np.inf]), 0.5)[0] == 0.0


def test_classify_respects_the_thresholds() -> None:
    levels = classify(np.array([0.0, 1.9, 2.0, 4.9, 5.0, 100.0]))
    expected = [
        Observability.UNOBSERVABLE,
        Observability.UNOBSERVABLE,
        Observability.WEAK,
        Observability.WEAK,
        Observability.OBSERVABLE,
        Observability.OBSERVABLE,
    ]
    assert list(levels) == [int(e) for e in expected]


def test_classify_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError):
        classify(np.array([1.0]), weak_sigma=5.0, strong_sigma=2.0)


# ---------------------------------------------------------------------------
# Excitation buys information
# ---------------------------------------------------------------------------
def test_more_current_excitation_tightens_the_resistance_bound(setup) -> None:
    """The identifiability thesis, in its simplest form: excitation is information."""
    _, params, _, topology, noise = setup
    specs = local_specs(1)
    bounds = []
    for amplitude in (0.1, 2.0):
        duty = pulse_train(300.0, 1.0, mean_c_rate=0.2, pulse_c_rate=amplitude)
        sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
        bounds.append(crlb(fisher_information(sens, topology, noise))[0])
    assert bounds[1] < bounds[0], "a larger current swing must sharpen the R0 estimate"


def test_constant_current_confounds_resistance_and_capacity_more_than_pulses(setup) -> None:
    """Constant current is the worst excitation: an IR offset and an OCV drift alias."""
    _, params, _, topology, noise = setup
    specs = local_specs(1, (ParamKind.R0, ParamKind.CAPACITY))

    flat = constant_current(300.0, 1.0, c_rate=0.2)
    pulsed = pulse_train(300.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0)

    vifs = []
    for duty in (flat, pulsed):
        sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
        vifs.append(variance_inflation(fisher_information(sens, topology, noise))[0])
    assert vifs[0] > vifs[1], "constant current must confound R0 with capacity more than pulses do"


def test_heatmap_snr_equals_magnitude_over_crlb_std(setup) -> None:
    """The closed form the heatmap relies on, checked against the definition."""
    _, params, _, topology, noise = setup
    result = detectability_heatmap(
        params,
        topology,
        noise,
        cell=1,
        kind=ParamKind.R0,
        excitation_c_rate=np.array([0.5, 1.5]),
        magnitude=np.array([0.05, 0.20]),
    )
    for i, m in enumerate(result.magnitude):
        for j, std in enumerate(result.crlb_std):
            assert result.snr[i, j] == pytest.approx(m / std, rel=1e-9)


def test_min_detectable_magnitude_is_five_sigma_of_the_crlb(setup) -> None:
    _, params, _, topology, noise = setup
    result = detectability_heatmap(
        params,
        topology,
        noise,
        cell=1,
        excitation_c_rate=np.array([1.0]),
        magnitude=np.array([0.1]),
    )
    np.testing.assert_allclose(result.min_detectable_magnitude(5.0), 5.0 * result.crlb_std)


# ---------------------------------------------------------------------------
# The headline behaviour
# ---------------------------------------------------------------------------
def test_an_instrumented_cell_is_always_more_identifiable_than_an_uninstrumented_one(setup) -> None:
    """The ordering property. Holds at any excitation, however information-starved.

    A cooling fault perturbs hA, which moves temperature. Every cell's hA leaks into
    the voltage channel through R0(T), so no cell is *exactly* unidentifiable -- but a
    thermocouple beats that leak, always.
    """
    pack, params, duty, topology, noise = setup
    grey = grey_cell_map(
        params, duty.current_a, duty.dt_s, topology, noise, kind=ParamKind.HA, magnitude=0.40
    )

    sensed = np.array(topology.temp_cells)
    unsensed = np.array([c for c in range(pack.n_cells) if c not in topology.temp_cells])
    assert grey.snr[sensed].min() > grey.snr[unsensed].max(), (
        "an instrumented cell must be strictly more identifiable than any uninstrumented one"
    )


def test_cooling_faults_are_identifiable_exactly_where_the_thermocouples_are(demo_setup) -> None:
    """The refusal, as a property. This is the headline result.

    At the demo's excitation, the four cells carrying a thermocouple clear 5 sigma and
    the other twenty-eight sit below 2 sigma. AstraCell abstains on those, whether or
    not a fault is present -- which is the point.
    """
    pack, params, duty, topology, noise, _, _ = demo_setup
    grey = grey_cell_map(
        params, duty.current_a, duty.dt_s, topology, noise, kind=ParamKind.HA, magnitude=0.40
    )

    assert set(grey.unobservable_cells()).isdisjoint(topology.temp_cells)
    observable = {c for c in range(pack.n_cells) if grey.level[c] == int(Observability.OBSERVABLE)}
    assert observable == set(topology.temp_cells)


def test_excitation_can_substitute_for_a_thermocouple(setup) -> None:
    """The other half of 'what should I measure next?'.

    Heat generation scales as I^2, so a harder current pulse makes the
    voltage-as-thermometer pathway work. An uninstrumented cell's cooling fault can be
    bought into view by exciting harder instead of by adding a sensor. Both options
    fall out of the same Fisher information; the code does not have to choose.
    """
    _, params, _, topology, noise = setup
    specs = local_specs(1, (ParamKind.HA,))
    bounds = []
    for amplitude in (0.5, 2.5):
        duty = pulse_train(600.0, 1.0, mean_c_rate=0.2, pulse_c_rate=amplitude)
        sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
        bounds.append(crlb(fisher_information(sens, topology.without_temp_sensors(), noise))[0])
    assert bounds[1] < bounds[0] / 4.0, "I^2 heating should sharpen the hA bound superlinearly"


def test_resistance_faults_are_identifiable_everywhere(setup) -> None:
    """Every cell reports its own voltage, so every cell's R0 is visible."""
    pack, params, duty, topology, noise = setup
    grey = grey_cell_map(
        params, duty.current_a, duty.dt_s, topology, noise, kind=ParamKind.R0, magnitude=0.20
    )
    assert grey.counts()["observable"] == pack.n_cells
    assert grey.unobservable_cells() == ()


def test_the_recommended_thermocouple_is_the_one_on_the_faulty_cell(setup, sens_all) -> None:
    """'What should I measure next?' has a defensible answer, and it is not a guess."""
    _, _, _, topology, noise = setup
    specs, sens = sens_all
    target = ParameterSpec(1, ParamKind.HA)
    ranked = recommend_temp_sensor(sens, topology, noise, specs, target, 0.40)
    best_cell, best_snr = ranked[0]
    assert best_cell == target.cell
    assert best_snr > ranked[1][1], "the best placement must strictly beat the runner-up"


def test_adding_the_recommended_sensor_flips_the_verdict(demo_setup) -> None:
    """REFUSE -> DIAGNOSE, driven entirely by a counterfactual row mask."""
    from astracell.observability.decision import VerdictKind, assess

    _, _, _, topology, noise, specs, sens = demo_setup
    target = ParameterSpec(10, ParamKind.HA)  # cell 10 carries no thermocouple
    assert target.cell not in topology.temp_cells

    before = assess(sens, topology, noise, specs, target, 0.40)
    after = assess(sens, topology.with_temp_sensor_at(target.cell), noise, specs, target, 0.40)

    assert before.kind is VerdictKind.REFUSE_UNOBSERVABLE
    assert after.will_diagnose
    assert after.crlb_std < before.crlb_std


@pytest.mark.regression
def test_a_thermocouple_informs_about_its_own_cell_and_essentially_nothing_else(
    demo_setup,
) -> None:
    """A regression test on one specific configuration, not a universal law.

    At this noise level, conduction carries almost no hA information from a cell to a
    thermocouple on its neighbour. So the cell *adjacent* to a thermocouple is no better
    determined than a cell four hops away -- in fact it is worse, because the far cell
    here is a pack corner, and a corner has fewer conduction paths, so it warms more for
    the same hA change and reads out more strongly through its own voltage.

    This is the finding that motivates using the Fisher information rather than a hop
    count. A distance heuristic gets it exactly backwards.

    If a parameter change breaks this test, the finding has changed and the README must
    change with it. That is the point of pinning it.
    """
    pack, params, duty, topology, noise, _, _ = demo_setup
    grey = grey_cell_map(
        params, duty.current_a, duty.dt_s, topology, noise, kind=ParamKind.HA, magnitude=0.40
    )

    def distance(cell: int) -> int:
        return min(pack.grid_distance(cell, s) for s in topology.temp_cells)

    sensor = topology.temp_cells[0]
    neighbour = sensor - 1  # distance 1
    corner = 0  # distance 4, and a pack corner

    assert distance(neighbour) == 1
    assert distance(corner) > distance(neighbour)

    # The thermocouple dominates its own cell by a wide margin...
    assert grey.snr[sensor] > 3.0 * grey.snr[neighbour]
    # ...and does essentially nothing for the cell next door, which a distant corner beats.
    assert grey.snr[corner] > grey.snr[neighbour], (
        f"corner cell {corner} (distance {distance(corner)}, SNR {grey.snr[corner]:.3f}) "
        f"should beat cell {neighbour} (distance 1, SNR {grey.snr[neighbour]:.3f})"
    )


def test_r0_temperature_dependence_is_the_thermal_leak_into_the_voltage_channel(setup) -> None:
    """Without Arrhenius R0(T), a cooling fault would be invisible to voltage entirely.

    This is why cell voltage works as a (bad) thermometer, and why hA is only
    *nearly* unidentifiable off-sensor rather than exactly so.
    """
    _, params, duty, topology, noise = setup
    specs = local_specs(1, (ParamKind.HA,))

    with_arrhenius = crlb(
        fisher_information(
            sensitivities(params, duty.current_a, duty.dt_s, specs),
            topology.without_temp_sensors(),
            noise,
        )
    )[0]
    isothermal_r0 = params.evolve(ea_over_r_k=0.0)
    without_arrhenius = crlb(
        fisher_information(
            sensitivities(isothermal_r0, duty.current_a, duty.dt_s, specs),
            topology.without_temp_sensors(),
            noise,
        )
    )[0]

    assert with_arrhenius < without_arrhenius, (
        "R0(T) is the pathway by which hA becomes visible in voltage"
    )


def test_no_arrhenius_and_no_thermocouples_leaves_only_the_entropic_pathway(setup) -> None:
    """Sanity: some thermal information survives via dOCV/dT even with R0(T) frozen."""
    _, params, duty, topology, noise = setup
    r0_at_temperature(params.r0_ohm, np.array([300.0]), 0.0)  # ea=0 -> no temperature dependence
    specs = local_specs(1, (ParamKind.HA,))
    isothermal_r0 = params.evolve(ea_over_r_k=0.0)
    fim = fisher_information(
        sensitivities(isothermal_r0, duty.current_a, duty.dt_s, specs),
        topology.without_temp_sensors(),
        noise,
    )
    assert fim[0, 0] > 0.0, "the entropic term still couples temperature into voltage"
    assert detection_snr(crlb(fim), 0.40)[0] < 2.0, "but nowhere near enough to diagnose"
