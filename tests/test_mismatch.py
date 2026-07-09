"""Model mismatch, and whether the bias we compute means anything.

A bias number is easy to produce and hard to trust. The obvious failure mode is a harness
that reports a large "structural bias" which is really a bug, a noise artifact, or the
projection machinery rubber-stamping whatever it is handed. Four independent lines of
evidence against that, and none of them relies on the plant being right:

1. **Algebra.** A residual orthogonal to the sensitivity span produces exactly zero bias;
   a residual equal to ``S @ delta`` produces exactly bias ``delta``. The projection is
   doing linear algebra, not agreeing with us.
2. **Invariance.** Multiply every noise sigma by ten, or replicate the experiment a hundred
   times, and the bias does not move by one bit while the CRLB moves by a factor of ten.
   Bias is structural; variance is statistical; the code knows the difference.
3. **Convergence.** The linearised bias predicts, to first order, where an actual damped
   Gauss-Newton fit of the observer to the plant comes to rest.
4. **The off switch.** Zero mismatch produces a bitwise-zero residual, and therefore zero
   bias, through the ordinary code path. This is the weakest of the four -- the plant shares
   the observer's cell primitives -- but it catches stray terms and recording-order slips.
"""

from __future__ import annotations

import numpy as np
import pytest

from astracell.duty import pulse_train
from astracell.observability.bias import (
    BiasConvergenceError,
    bias_aware_snr,
    bias_ceiling,
    parameter_bias,
    pseudo_true_bias,
    residual_score,
    solve_bias,
    structural_residual,
)
from astracell.observability.decision import VerdictKind, assess_under_mismatch, decide
from astracell.observability.fisher import channel_slices, crlb, fisher_information
from astracell.observability.mask import detection_snr
from astracell.observability.sensitivity import (
    ParameterSpec,
    ParamKind,
    all_specs,
    local_specs,
    sensitivities,
    with_current_bias,
)
from astracell.pack import PackTopology, nominal_pack
from astracell.pack.simulate import simulate
from astracell.plant import (
    NO_MISMATCH,
    REALISTIC_MISMATCH,
    MismatchModel,
    PlantStabilityError,
    simulate_plant,
)
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import realistic_topology

QUIET = NoiseModel()
LOUD = NoiseModel(voltage_sigma_v=1e-2, voltage_lsb_v=1e-3, temp_sigma_k=5.0, temp_lsb_k=0.625)


@pytest.fixture(scope="module")
def setup():
    pack = PackTopology(n_modules=2, cells_per_module=4)
    params = nominal_pack(pack, seed=0)
    topology = realistic_topology(pack, n_temp_sensors=2)
    duty = pulse_train(600.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0)
    return pack, params, topology, duty


@pytest.fixture(scope="module")
def sensed_cell(setup):
    """A cell with a thermocouple, so all three of its parameters are identifiable."""
    _, _, topology, _ = setup
    return topology.temp_cells[0]


@pytest.fixture(scope="module")
def residual(setup):
    _, params, _, duty = setup
    return structural_residual(params, REALISTIC_MISMATCH, duty.current_a, duty.dt_s)


# ---------------------------------------------------------------------------
# 1. Algebra: the projection is not a rubber stamp
# ---------------------------------------------------------------------------
def test_a_residual_orthogonal_to_the_sensitivity_span_produces_no_bias(setup, sensed_cell) -> None:
    """The part of the model error nobody can attribute to a parameter costs nothing.

    Built by projecting a random residual *out* of the span of the whitened sensitivity
    columns. It inflates the fit residual and biases nothing, which is what "orthogonal"
    has to mean if the formula is right.
    """
    _, params, topology, duty = setup
    specs = local_specs(sensed_cell)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)

    rng = np.random.default_rng(0)
    arbitrary = rng.standard_normal(sens.shape[:3])

    # Project out the span, channel by channel, in the whitened metric that residual_score
    # uses. Whitening with rho = 0 is the identity, so plain least squares suffices here.
    columns = np.concatenate(
        [
            sens[:, index, channel, :].reshape(-1, sens.shape[3]) / sigma
            for index, channel, sigma, _ in channel_slices(topology, QUIET)
        ]
    )
    flat = np.concatenate(
        [
            arbitrary[:, index, channel].reshape(-1) / sigma
            for index, channel, sigma, _ in channel_slices(topology, QUIET)
        ]
    )
    coefficients, *_ = np.linalg.lstsq(columns, flat, rcond=None)
    orthogonal_flat = flat - columns @ coefficients

    # Unflatten back into a residual array, undoing the 1/sigma scaling.
    orthogonal = np.zeros_like(arbitrary)
    offset = 0
    for index, channel, sigma, _ in channel_slices(topology, QUIET):
        size = arbitrary[:, index, channel].size
        shape = arbitrary[:, index, channel].shape
        orthogonal[:, index, channel] = (
            orthogonal_flat[offset : offset + size].reshape(shape) * sigma
        )
        offset += size

    bias = parameter_bias(sens, orthogonal, topology, QUIET)
    assert np.abs(bias).max() < 1e-8, f"orthogonal residual leaked into the parameters: {bias}"


def test_a_residual_that_is_exactly_a_parameter_change_is_recovered_exactly(
    setup, sensed_cell
) -> None:
    """``r = S @ delta`` must give back ``b == delta``.

    If the plant differs from the observer by exactly a parameter perturbation, the fit
    should attribute it to exactly that perturbation. Anything else means the projection
    is wrong. This pins the formula independently of any battery physics.
    """
    _, params, topology, duty = setup
    specs = local_specs(sensed_cell)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)

    delta = np.array([0.031, -0.017, 0.240])
    synthetic = sens @ delta  # (n_time, n_cells, 2)

    bias = parameter_bias(sens, synthetic, topology, QUIET)
    np.testing.assert_allclose(bias, delta, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("rho", [0.0, 0.7])
def test_the_bias_projection_uses_the_same_whitening_as_the_fisher_matrix(setup, rho) -> None:
    """``S @ delta`` is recovered exactly under correlated noise too.

    ``fisher_information`` and ``residual_score`` index through the same ``channel_slices``,
    so a residual can never be whitened differently from the sensitivities it is projected
    onto. If they ever diverge, this test breaks.
    """
    _, params, topology, duty = setup
    specs = local_specs(1)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    noise = NoiseModel(voltage_rho=rho, temp_rho=rho)

    delta = np.array([0.02, -0.05, 0.10])
    bias = parameter_bias(sens, sens @ delta, topology, noise)
    np.testing.assert_allclose(bias, delta, rtol=1e-7, atol=1e-11)


def test_solve_bias_refuses_to_invent_a_value_for_an_unidentifiable_direction() -> None:
    """No ``pinv``. A parameter the data says nothing about has no bias either."""
    fim = np.diag([4.0, 0.0])
    bias = solve_bias(fim, np.array([2.0, 7.0]))
    assert bias[0] == pytest.approx(0.5)
    assert np.isinf(bias[1])


def test_a_residual_on_an_uninstrumented_channel_cannot_bias_anything(setup) -> None:
    """Nobody looked, so nothing was inferred."""
    _, params, topology, duty = setup
    specs = local_specs(1)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)

    blind = [c for c in range(params.n_cells) if c not in topology.temp_cells]
    residual = np.zeros(sens.shape[:3])
    residual[:, blind, 1] = 5.0  # a 5 K modelling error on every unmeasured thermocouple

    assert np.abs(parameter_bias(sens, residual, topology, QUIET)).max() < 1e-9


# ---------------------------------------------------------------------------
# 2. Invariance: bias is structural, variance is statistical
# ---------------------------------------------------------------------------
def test_rescaling_every_noise_sigma_moves_the_crlb_and_not_the_bias(
    setup, sensed_cell, residual
) -> None:
    """The single sharpest distinction between the two gates, and it is exact.

    ``b = FIM^-1 S^T Sigma^-1 r``: scale ``Sigma`` by ``c`` and the ``FIM^-1`` contributes
    ``c`` while the score contributes ``1/c``. Nothing is left. Meanwhile the CRLB scales by
    ``c`` outright. Better sensors buy precision; they buy no correctness at all.
    """
    _, params, topology, duty = setup
    specs = local_specs(sensed_cell)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)

    quiet = parameter_bias(sens, residual, topology, QUIET)
    loud = parameter_bias(sens, residual, topology, LOUD)
    np.testing.assert_allclose(loud, quiet, rtol=1e-12)

    ratio = np.sqrt(crlb(fisher_information(sens, topology, LOUD))) / np.sqrt(
        crlb(fisher_information(sens, topology, QUIET))
    )
    assert np.allclose(ratio, 10.0, rtol=1e-6), "the CRLB must scale with the noise"


def test_replicating_the_experiment_shrinks_the_variance_and_not_the_bias(
    setup, sensed_cell, residual
) -> None:
    """Information adds, so ``k`` replicas give ``k*FIM`` and ``k*score``. The ``k`` cancels."""
    _, params, topology, duty = setup
    specs = local_specs(sensed_cell)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    fim = fisher_information(sens, topology, QUIET)
    score = residual_score(sens, residual, topology, QUIET)

    once = solve_bias(fim, score)
    for k in (4.0, 100.0):
        np.testing.assert_allclose(solve_bias(k * fim, k * score), once, rtol=1e-12)
        np.testing.assert_allclose(
            np.sqrt(crlb(k * fim)), np.sqrt(crlb(fim)) / np.sqrt(k), rtol=1e-10
        )


def test_the_bias_ceiling_is_the_limit_of_the_total_snr(setup, sensed_cell, residual) -> None:
    """``m / sqrt(CRLB + b^2) -> m / |b|`` as the variance is driven to zero."""
    _, params, topology, duty = setup
    specs = local_specs(sensed_cell)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    fim = fisher_information(sens, topology, QUIET)
    bias = parameter_bias(sens, residual, topology, QUIET)
    magnitude = 0.20

    ceiling = bias_ceiling(bias, magnitude)
    approached = bias_aware_snr(crlb(1e8 * fim), bias, magnitude)
    np.testing.assert_allclose(approached, ceiling, rtol=1e-4)

    variance_only = detection_snr(crlb(1e8 * fim), magnitude)
    assert np.all(variance_only > 100.0 * ceiling), "the CRLB alone would grow without bound"


def test_total_snr_never_exceeds_the_variance_only_snr(setup, sensed_cell, residual) -> None:
    _, params, topology, duty = setup
    specs = local_specs(sensed_cell)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    variance = crlb(fisher_information(sens, topology, QUIET))
    bias = parameter_bias(sens, residual, topology, QUIET)

    assert np.all(bias_aware_snr(variance, bias, 0.2) <= detection_snr(variance, 0.2) + 1e-12)
    np.testing.assert_allclose(
        bias_aware_snr(variance, np.zeros_like(bias), 0.2), detection_snr(variance, 0.2)
    )


# ---------------------------------------------------------------------------
# 3. Convergence: the linearisation predicts an actual fit
# ---------------------------------------------------------------------------
def test_the_linearised_bias_is_first_order_accurate(setup, sensed_cell) -> None:
    """Relative error against a real Gauss-Newton fit must vanish with the mismatch.

    This is the test that makes the whole module credible. ``parameter_bias`` is one Newton
    step; ``pseudo_true_bias`` iterates to the fixed point. If they agree only by accident,
    the error will not fall as the mismatch shrinks. It falls linearly.

    Note where they *disagree*: at full ``REALISTIC_MISMATCH`` the linearisation is off by
    tens of percent. It is a scale estimate, not a correction, and the example quotes the
    iterated fit.
    """
    _, params, topology, duty = setup
    specs = local_specs(sensed_cell)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)

    errors = []
    for scale in (0.3, 0.1, 0.03):
        mismatch = REALISTIC_MISMATCH.scaled(scale)
        linear = parameter_bias(
            sens, structural_residual(params, mismatch, duty.current_a, duty.dt_s), topology, QUIET
        )
        exact = pseudo_true_bias(
            params, mismatch, duty.current_a, duty.dt_s, topology, QUIET, specs
        )
        errors.append(float(np.abs(linear - exact).max() / np.abs(exact).max()))

    assert errors[0] > errors[1] > errors[2], f"error must fall with mismatch: {errors}"
    assert errors[2] < 0.05, f"at 3% mismatch the linearisation should be tight, got {errors[2]}"
    # A tenfold smaller mismatch should give roughly a tenfold smaller error: O(scale).
    assert errors[0] / errors[2] > 4.0, "the error is not first order in the mismatch"


def test_the_bias_is_first_order_in_the_mismatch_strength(setup, sensed_cell) -> None:
    """Halving every mismatch knob halves the bias. ``b / scale`` converges to a constant.

    ``atol`` is not laziness. ``R0``'s first-order coefficient is nearly zero here -- the
    contributions of the four knobs very nearly cancel, as ``examples/04`` shows by walking
    the excitation through the crossing -- so its *relative* convergence is meaningless while
    its absolute bias is 0.07%. Asserting a relative tolerance on a quantity that passes
    through zero would be a test of arithmetic noise.
    """
    _, params, topology, duty = setup
    specs = local_specs(sensed_cell)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)

    def slope(scale: float) -> np.ndarray:
        mismatch = REALISTIC_MISMATCH.scaled(scale)
        residual = structural_residual(params, mismatch, duty.current_a, duty.dt_s)
        return parameter_bias(sens, residual, topology, QUIET) / scale

    np.testing.assert_allclose(slope(0.05), slope(0.1), rtol=0.05, atol=2e-3)


def test_bias_is_additive_across_independent_mismatch_mechanisms(setup, sensed_cell) -> None:
    """To first order the knobs do not interact, which is what "linear" buys you."""
    _, params, topology, duty = setup
    specs = local_specs(sensed_cell)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    scale = 0.2

    def bias_of(mismatch: MismatchModel) -> np.ndarray:
        return parameter_bias(
            sens, structural_residual(params, mismatch, duty.current_a, duty.dt_s), topology, QUIET
        )

    pieces = [
        MismatchModel(r0_soc_slope=REALISTIC_MISMATCH.r0_soc_slope * scale),
        MismatchModel(second_rc_r_ohm=REALISTIC_MISMATCH.second_rc_r_ohm * scale),
        MismatchModel(
            core_surface_resistance_k_per_w=REALISTIC_MISMATCH.core_surface_resistance_k_per_w
            / scale
        ),
        MismatchModel(temp_sensor_tau_s=REALISTIC_MISMATCH.temp_sensor_tau_s * scale),
    ]
    separate = sum(bias_of(piece) for piece in pieces)

    together = bias_of(
        MismatchModel(
            r0_soc_slope=pieces[0].r0_soc_slope,
            second_rc_r_ohm=pieces[1].second_rc_r_ohm,
            core_surface_resistance_k_per_w=pieces[2].core_surface_resistance_k_per_w,
            temp_sensor_tau_s=pieces[3].temp_sensor_tau_s,
        )
    )
    np.testing.assert_allclose(together, separate, rtol=0.10)


def test_pseudo_true_bias_refuses_a_global_nuisance_parameter(setup, sensed_cell) -> None:
    _, params, topology, duty = setup
    specs = with_current_bias(local_specs(sensed_cell))
    with pytest.raises(ValueError, match="per-cell parameters only"):
        pseudo_true_bias(
            params, REALISTIC_MISMATCH, duty.current_a, duty.dt_s, topology, QUIET, specs
        )


def test_pseudo_true_bias_refuses_a_parameter_with_no_information_at_all(setup) -> None:
    """No thermocouples *and* no ``R0(T)`` coupling: ``hA`` then touches nothing measured.

    The FIM is exactly singular, so the pseudo-true value does not exist -- there is a whole
    ray of parameters fitting the data equally well. Raise, rather than hand back the
    minimum-norm point on that ray with a straight face.
    """
    _, params, topology, duty = setup
    isothermal = params.evolve(ea_over_r_k=0.0)
    blind = next(c for c in range(params.n_cells) if c not in topology.temp_cells)
    specs = (ParameterSpec(blind, ParamKind.R0), ParameterSpec(blind, ParamKind.HA))

    with pytest.raises(BiasConvergenceError, match="unidentifiable"):
        pseudo_true_bias(
            isothermal,
            REALISTIC_MISMATCH,
            duty.current_a,
            duty.dt_s,
            topology.without_temp_sensors(),
            QUIET,
            specs,
        )


def test_pseudo_true_bias_raises_rather_than_returning_a_half_converged_fit(setup) -> None:
    """A barely-identifiable direction makes damped Gauss-Newton crawl. Say so.

    Cell 1 carries no thermocouple, so its cooling coefficient is visible only through the
    faint ``R0(T)`` leak. The fit is not singular, merely hopeless: the step cap keeps it
    stable and it never settles. Returning ``theta`` after ``max_iter`` would be a number
    with no meaning attached.
    """
    _, params, topology, duty = setup
    blind = next(c for c in range(params.n_cells) if c not in topology.temp_cells)
    specs = (ParameterSpec(blind, ParamKind.HA),)

    with pytest.raises(BiasConvergenceError, match="did not converge"):
        pseudo_true_bias(
            params,
            REALISTIC_MISMATCH,
            duty.current_a,
            duty.dt_s,
            topology.without_temp_sensors(),
            QUIET,
            specs,
        )


# ---------------------------------------------------------------------------
# 4. The off switch, and the plant itself
# ---------------------------------------------------------------------------
def test_zero_mismatch_reproduces_the_observer_exactly(setup) -> None:
    """Bitwise, not approximately. The knobs are wired to nothing when they are zero."""
    _, params, _, duty = setup
    observer = simulate(params, duty.current_a, duty.dt_s)
    plant = simulate_plant(params, duty.current_a, duty.dt_s, NO_MISMATCH)

    assert np.array_equal(plant.voltage_v, observer.voltage_v)
    assert np.array_equal(plant.soc, observer.soc)
    assert np.array_equal(plant.temp_measured_k, observer.temp_k)
    assert np.array_equal(plant.temp_core_k, plant.temp_surface_k)
    assert np.array_equal(plant.core_surface_gradient_k, np.zeros_like(plant.temp_core_k))


def test_zero_mismatch_gives_exactly_zero_residual_and_zero_bias(setup, sensed_cell) -> None:
    _, params, topology, duty = setup
    residual = structural_residual(params, NO_MISMATCH, duty.current_a, duty.dt_s)
    assert np.count_nonzero(residual) == 0

    specs = local_specs(sensed_cell)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    assert np.count_nonzero(parameter_bias(sens, residual, topology, QUIET)) == 0
    assert np.all(np.isinf(bias_ceiling(np.zeros(3), 0.2)))


def test_scaled_to_zero_is_exact_and_scaling_is_monotone() -> None:
    assert REALISTIC_MISMATCH.scaled(0.0).is_exact
    assert not REALISTIC_MISMATCH.scaled(1e-3).is_exact
    assert NO_MISMATCH.is_exact
    half = REALISTIC_MISMATCH.scaled(0.5)
    assert half.second_rc_r_ohm == REALISTIC_MISMATCH.second_rc_r_ohm * 0.5
    assert half.second_rc_tau_s == REALISTIC_MISMATCH.second_rc_tau_s  # shape, not strength
    assert half.core_heat_fraction == REALISTIC_MISMATCH.core_heat_fraction


def test_the_structural_residual_is_exactly_zero_at_the_initial_sample(setup, residual) -> None:
    """Which is what makes ``theta*`` well defined. See ``plant.mismatch``."""
    assert np.count_nonzero(residual[0]) == 0


def test_the_realistic_residual_dwarfs_the_noise_floor(setup, residual) -> None:
    """If it did not, none of this would matter."""
    voltage_rms = float(np.sqrt((residual[:, :, 0] ** 2).mean()))
    assert voltage_rms > 3.0 * QUIET.voltage_sigma_v


def test_a_stiff_core_surface_mode_is_refused_not_silently_substepped(setup) -> None:
    _, params, _, duty = setup
    stiff = MismatchModel(core_surface_resistance_k_per_w=1e-4)
    with pytest.raises(PlantStabilityError, match="forward Euler"):
        simulate_plant(params, duty.current_a, duty.dt_s, stiff)


def test_mismatch_validation() -> None:
    with pytest.raises(ValueError, match="second_rc_r_ohm"):
        MismatchModel(second_rc_r_ohm=-1.0)
    with pytest.raises(ValueError, match="core_heat_fraction"):
        MismatchModel(core_heat_fraction=1.0)
    with pytest.raises(ValueError, match="second_rc_tau_s"):
        MismatchModel(second_rc_tau_s=0.0)
    with pytest.raises(ValueError, match="factor"):
        REALISTIC_MISMATCH.scaled(-0.5)


def test_the_plant_actually_develops_a_core_surface_gradient(setup) -> None:
    _, params, _, duty = setup
    plant = simulate_plant(params, duty.current_a, duty.dt_s, REALISTIC_MISMATCH)
    assert np.abs(plant.core_surface_gradient_k).max() > 0.1
    assert np.all(plant.temp_core_k >= plant.temp_surface_k - 1e-9), "the core is the heat source"


# ---------------------------------------------------------------------------
# The decision layer
# ---------------------------------------------------------------------------
def test_omitting_the_bias_reproduces_the_old_verdict_exactly() -> None:
    """The mismatch gate is opt-in. Without it, ``decide`` is what it always was."""
    target = ParameterSpec(0, ParamKind.R0)
    without = decide(target, 0.2, snr=50.0, crlb_std=0.004, vif=1.0)
    assert without.kind is VerdictKind.DIAGNOSE
    assert without.bias is None
    assert without.snr_total == without.snr
    assert np.isinf(without.bias_ceiling)


def test_a_fault_that_clears_the_noise_but_not_the_model_is_refused() -> None:
    """The acceptance criterion, in miniature: 50 sigma of variance, 0.4 sigma of truth."""
    target = ParameterSpec(0, ParamKind.CAPACITY)
    verdict = decide(target, 0.05, snr=50.0, crlb_std=0.001, vif=1.0, bias=-0.12)

    assert verdict.kind is VerdictKind.REFUSE_MODEL_BIAS
    assert verdict.refused
    assert not verdict.will_diagnose
    assert verdict.snr == pytest.approx(50.0)
    assert verdict.snr_total < 0.5
    assert verdict.bias_ceiling == pytest.approx(0.05 / 0.12)
    assert "false positive" in verdict.reason
    assert verdict.recommendation is not None
    assert "measure differently" in verdict.recommendation
    assert "more data" not in verdict.recommendation


def test_a_diagnosis_squeezed_below_five_sigma_is_weakened_not_refused() -> None:
    target = ParameterSpec(0, ParamKind.R0)
    verdict = decide(target, 0.2, snr=40.0, crlb_std=0.005, vif=1.0, bias=0.06)
    assert verdict.kind is VerdictKind.WEAK_EVIDENCE
    assert 2.0 <= verdict.snr_total < 5.0
    assert "do not act on it" in verdict.reason


def test_a_small_bias_leaves_a_strong_diagnosis_standing() -> None:
    target = ParameterSpec(0, ParamKind.R0)
    verdict = decide(target, 0.2, snr=40.0, crlb_std=0.005, vif=1.0, bias=0.001)
    assert verdict.kind is VerdictKind.DIAGNOSE
    assert "survives a structural bias" in verdict.reason


def test_confounding_is_still_checked_before_bias() -> None:
    target = ParameterSpec(0, ParamKind.R0)
    verdict = decide(target, 0.2, snr=50.0, crlb_std=0.004, vif=99.0, bias=-0.5)
    assert verdict.kind is VerdictKind.REFUSE_CONFOUNDED


def test_an_unobservable_parameter_is_refused_for_variance_not_bias() -> None:
    """Both gates would refuse. The reason must name the one that fired first, because the
    remedies differ: one is a sensor, the other is a model."""
    target = ParameterSpec(0, ParamKind.HA)
    verdict = decide(target, 0.4, snr=1.1, crlb_std=0.36, vif=2.0, bias=5.0)
    assert verdict.kind is VerdictKind.REFUSE_UNOBSERVABLE


@pytest.mark.regression
def test_capacity_survives_the_noise_and_dies_of_model_bias(setup) -> None:
    """The headline, pinned on one configuration.

    With the ammeter believed, a 5% capacity fault sits at tens of sigma above the
    Cramer-Rao floor and is nevertheless not credible: the observer's own missing diffusion
    branch manufactures an apparent capacity loss several times larger than the fault.
    """
    _, params, topology, duty = setup
    target = ParameterSpec(1, ParamKind.CAPACITY)
    specs = local_specs(1)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    residual = structural_residual(params, REALISTIC_MISMATCH, duty.current_a, duty.dt_s)

    variance_only = assess_under_mismatch(
        sens, topology, QUIET, specs, target, 0.05, np.zeros_like(residual)
    )
    with_bias = assess_under_mismatch(sens, topology, QUIET, specs, target, 0.05, residual)

    assert variance_only.kind is VerdictKind.DIAGNOSE
    assert variance_only.snr > 10.0
    assert with_bias.kind is VerdictKind.REFUSE_MODEL_BIAS
    assert with_bias.snr == pytest.approx(variance_only.snr)  # same data, same variance
    assert abs(with_bias.bias or 0.0) > 0.05  # bias exceeds the fault we are hunting
    assert with_bias.bias_ceiling < 2.0


@pytest.mark.regression
def test_a_free_current_bias_soaks_up_common_mode_model_error(setup) -> None:
    """Nuisance parameters are where structural error goes to hide.

    Freeing the pack-global current offset collapses the per-cell capacity bias, because a
    common-mode voltage residual looks like a mis-calibrated shunt. The capacity estimate
    really does improve. The "shunt calibration" you get back is pure model error, and
    nothing in the CRLB flags it as such.
    """
    _, params, topology, duty = setup
    residual = structural_residual(params, REALISTIC_MISMATCH, duty.current_a, duty.dt_s)

    plain = local_specs(1)
    nuisance = with_current_bias(plain)
    fixed = parameter_bias(
        sensitivities(params, duty.current_a, duty.dt_s, plain), residual, topology, QUIET
    )
    freed = parameter_bias(
        sensitivities(params, duty.current_a, duty.dt_s, nuisance),
        residual,
        topology,
        QUIET,
        nuisance,
    )

    capacity = 1  # index of CAPACITY within local_specs
    assert abs(fixed[capacity]) > 10.0 * abs(freed[capacity]), (
        f"the nuisance parameter should absorb the common-mode residual: "
        f"{100 * fixed[capacity]:.2f}% -> {100 * freed[capacity]:.2f}%"
    )
    assert abs(freed[-1]) > 0.1, "and it should show up as a fictitious current offset"


@pytest.mark.regression
def test_harder_excitation_moves_the_bias_rather_than_removing_it(setup) -> None:
    """Variance can be reduced. Bias can only be moved.

    Raising the pulse amplitude pins ``R0`` down with the IR drop, so the slow polarisation
    residual has nowhere left to go but capacity. ``R0``'s credibility improves and
    capacity's collapses -- while both CRLBs improve. The Ds-optimal test planner of
    ``examples/03_next_best_test.py`` optimises the variance and is blind to this.
    """
    _, params, topology, _ = setup
    cell = 1
    specs = local_specs(cell)

    def measure(pulse_c_rate: float) -> tuple[np.ndarray, np.ndarray]:
        duty = pulse_train(600.0, 1.0, mean_c_rate=0.2, pulse_c_rate=pulse_c_rate)
        sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
        residual = structural_residual(params, REALISTIC_MISMATCH, duty.current_a, duty.dt_s)
        std = np.sqrt(crlb(fisher_information(sens, topology, QUIET)))
        return parameter_bias(sens, residual, topology, QUIET), std

    gentle_bias, gentle_std = measure(0.25)
    hard_bias, hard_std = measure(2.5)

    assert np.all(hard_std[:2] < gentle_std[:2]), "harder excitation must reduce the variance"
    assert abs(hard_bias[0]) < abs(gentle_bias[0]), "R0's bias should shrink"
    assert abs(hard_bias[1]) > abs(gentle_bias[1]), "capacity's bias should grow"


def test_assess_under_mismatch_rejects_a_target_outside_the_spec_set(setup, residual) -> None:
    pack, params, topology, duty = setup
    specs = local_specs(1)
    sens = sensitivities(params, duty.current_a, duty.dt_s, specs)
    with pytest.raises(ValueError, match="not among"):
        assess_under_mismatch(
            sens, topology, QUIET, specs, ParameterSpec(0, ParamKind.R0), 0.2, residual
        )


def test_residual_score_rejects_a_shape_mismatch(setup) -> None:
    _, params, topology, duty = setup
    sens = sensitivities(params, duty.current_a, duty.dt_s, all_specs(params.n_cells))
    with pytest.raises(ValueError, match="does not match"):
        residual_score(sens, np.zeros((3, 3, 2)), topology, QUIET)
