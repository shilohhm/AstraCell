"""Optimal experiment design.

The load-bearing claim is that Fisher information from independent experiments **adds**,
which is what lets ``plan_tests`` search over sums of precomputed matrices instead of
re-simulating. Everything else follows from it.
"""

from __future__ import annotations

import numpy as np
import pytest

from astracell.duty import pulse_train
from astracell.observability.experiment import (
    CandidateTest,
    default_test_library,
    plan_tests,
    rank_tests,
    recommend_test,
    render_ranking,
)
from astracell.observability.fisher import crlb, gaussian_entropy, information_gain
from astracell.observability.sensitivity import (
    ParameterSpec,
    ParamKind,
    local_specs,
    with_current_bias,
)
from astracell.pack import PackTopology, nominal_pack
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import realistic_topology

TARGET_CELL = 1


@pytest.fixture(scope="module")
def setup():
    pack = PackTopology(n_modules=2, cells_per_module=4)
    params = nominal_pack(pack, seed=0)
    topology = realistic_topology(pack, n_temp_sensors=2)
    baseline = pulse_train(300.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0)
    # A small library: the full default takes 6 x 2 x 4 simulations of up to 1800 steps.
    keep = {"rest_60s", "pulse_2C_10s", "pulse_train_2.5C"}
    library = tuple(t for t in default_test_library() if t.name in keep)
    return pack, params, topology, baseline, library


@pytest.fixture(scope="module")
def ranked(setup):
    _, params, topology, baseline, library = setup
    specs = with_current_bias(local_specs(TARGET_CELL))
    target = ParameterSpec(TARGET_CELL, ParamKind.HA)
    scores, fim_before, fim_per_test = rank_tests(
        params, topology, NoiseModel(), specs, target, 0.40, library=library, baseline=baseline
    )
    return specs, target, scores, fim_before, fim_per_test, library


# ---------------------------------------------------------------------------
# The theorem everything rests on
# ---------------------------------------------------------------------------
def test_information_gain_equals_the_entropy_difference() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4))
    before = a @ a.T + 4 * np.eye(4)
    b = rng.standard_normal((4, 4))
    after = before + b @ b.T

    assert information_gain(before, after) == pytest.approx(
        gaussian_entropy(before) - gaussian_entropy(after)
    )


def test_information_gain_is_non_negative_when_information_is_added() -> None:
    rng = np.random.default_rng(1)
    before = np.eye(3) * 2.0
    extra = rng.standard_normal((3, 3))
    assert information_gain(before, before + extra @ extra.T) >= -1e-12


def test_information_gain_is_infinite_when_a_singular_prior_becomes_identifiable() -> None:
    singular = np.diag([1.0, 0.0])
    assert np.isinf(information_gain(singular, np.eye(2)))
    assert information_gain(np.eye(2), singular) == 0.0


def test_gaussian_entropy_of_a_singular_fim_is_infinite() -> None:
    assert np.isinf(gaussian_entropy(np.diag([1.0, 0.0])))


def test_gaussian_entropy_of_the_identity_is_the_standard_value() -> None:
    k = 3
    assert gaussian_entropy(np.eye(k)) == pytest.approx(0.5 * k * np.log(2 * np.pi * np.e))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_doing_nothing_gains_nothing_about_the_target(ranked) -> None:
    """``rest_60s`` is a negative control for the *target*. If it scores there, the score
    is broken."""
    _, _, scores, _, _, _ = ranked
    control = next(s for s in scores if s.name == "rest_60s")
    assert control.eig_target_nats < 0.01
    assert not control.resolves()


def test_doing_nothing_still_informs_the_current_bias(ranked) -> None:
    """And a third demonstration that D-optimality answers a question nobody asked.

    Resting is not information-free. With zero commanded current, any voltage offset that
    scales with each cell's R0 reveals a current-sensor bias -- so ``det FIM`` grows. It
    reveals nothing whatever about a cooling fault. ``eig_nats`` sees the former;
    ``eig_target_nats`` correctly sees the latter.
    """
    _, _, scores, _, _, _ = ranked
    control = next(s for s in scores if s.name == "rest_60s")
    assert control.eig_nats > control.eig_target_nats
    assert control.eig_nats > 1e-3, "a rest period does constrain the current bias"


def test_the_control_never_outranks_a_hard_pulse(ranked) -> None:
    _, _, scores, _, _, _ = ranked
    control = next(s for s in scores if s.name == "rest_60s")
    pulse = next(s for s in scores if s.name == "pulse_2C_10s")
    assert pulse.eig_target_per_minute > control.eig_target_per_minute


def test_every_test_gains_non_negative_information(ranked) -> None:
    _, _, scores, _, _, _ = ranked
    for s in scores:
        assert s.eig_target_nats >= -1e-9, f"{s.name} lost information"
        assert s.eig_nats >= -1e-9


def test_target_eig_is_the_marginal_entropy_reduction(ranked) -> None:
    """``eig_target_nats == 0.5 * log(var_before / var_after)``, by definition."""
    specs, target, scores, fim_before, fim_per_test, library = ranked
    index = specs.index(target)
    var_before = crlb(fim_before)[index]
    by_name = {t.name: f for t, f in zip(library, fim_per_test, strict=True)}
    for s in scores:
        var_after = crlb(fim_before + by_name[s.name])[index]
        expected = 0.5 * np.log(var_before / var_after)
        assert s.eig_target_nats == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_scores_are_sorted_by_target_nats_per_minute(ranked) -> None:
    _, _, scores, _, _, _ = ranked
    rates = [s.eig_target_per_minute for s in scores]
    assert rates == sorted(rates, reverse=True)


def test_ranking_a_parameter_outside_the_spec_set_raises(setup) -> None:
    _, params, topology, baseline, library = setup
    specs = local_specs(TARGET_CELL)
    with pytest.raises(ValueError, match="not among"):
        rank_tests(
            params,
            topology,
            NoiseModel(),
            specs,
            ParameterSpec(0, ParamKind.R0),
            0.2,
            library=library,
            baseline=baseline,
        )


def test_render_ranking_produces_a_row_per_test(ranked) -> None:
    _, target, scores, fim_before, _, _ = ranked
    text = render_ranking(scores, fim_before, target)
    assert target.label() in text
    for s in scores:
        assert s.name in text


# ---------------------------------------------------------------------------
# Recommending vs scoring: they are different questions
# ---------------------------------------------------------------------------
def test_recommend_test_returns_the_cheapest_crossing_not_the_most_efficient(ranked) -> None:
    _, _, scores, _, _, _ = ranked
    best = recommend_test(scores, target_sigma=5.0)
    clearing = [s for s in scores if s.resolves(5.0)]
    if clearing:
        assert best is not None
        assert best.cost_s == min(s.cost_s for s in clearing)
    else:
        assert best is None


def test_recommend_test_returns_none_when_nothing_clears(ranked) -> None:
    _, _, scores, _, _, _ = ranked
    assert recommend_test(scores, target_sigma=1e9) is None


def test_recommend_test_respects_the_vif_gate(ranked) -> None:
    _, _, scores, _, _, _ = ranked
    assert recommend_test(scores, target_sigma=0.0, max_vif=0.0) is None


# ---------------------------------------------------------------------------
# Greedy planning
# ---------------------------------------------------------------------------
def test_plan_snr_trajectory_is_monotone_non_decreasing(ranked) -> None:
    specs, target, _, fim_before, fim_per_test, library = ranked
    plan = plan_tests(
        fim_before, fim_per_test, library, specs.index(target), 0.40, target_sigma=6.0
    )
    traj = plan.snr_trajectory
    pairs = zip(traj, traj[1:], strict=False)
    assert all(b >= a - 1e-9 for a, b in pairs), "information cannot hurt"


def test_a_resolved_plan_actually_crosses_the_threshold(ranked) -> None:
    specs, target, _, fim_before, fim_per_test, library = ranked
    plan = plan_tests(
        fim_before, fim_per_test, library, specs.index(target), 0.40, target_sigma=5.0
    )
    if plan.resolved:
        assert plan.final_snr >= 5.0
        assert plan.total_cost_s == sum(t.cost_s for t in plan.steps)
    else:
        assert plan.final_snr < 5.0
        assert "NOT RESOLVED" in plan.render()


def test_an_unreachable_threshold_is_reported_not_hidden(ranked) -> None:
    specs, target, _, fim_before, fim_per_test, library = ranked
    plan = plan_tests(
        fim_before, fim_per_test, library, specs.index(target), 0.40, target_sigma=1e6, max_tests=3
    )
    assert not plan.resolved
    assert "NOT RESOLVED" in plan.render()
    assert len(plan.steps) <= 3


def test_a_library_of_null_tests_resolves_nothing(ranked) -> None:
    specs, target, _, fim_before, fim_per_test, library = ranked
    i = next(k for k, t in enumerate(library) if t.name == "rest_60s")
    plan = plan_tests(
        fim_before, [fim_per_test[i]], (library[i],), specs.index(target), 0.40, target_sigma=5.0
    )
    assert not plan.resolved


def test_repeats_are_allowed_and_help(ranked) -> None:
    """Running the same experiment twice really does buy ~sqrt(2) in the standard error."""
    specs, target, _, fim_before, fim_per_test, library = ranked
    index = specs.index(target)
    i = next(k for k, t in enumerate(library) if t.name == "pulse_train_2.5C")

    once = crlb(fim_before + fim_per_test[i])[index]
    twice = crlb(fim_before + 2 * fim_per_test[i])[index]
    assert twice < once


def test_candidate_cost_is_the_profile_duration() -> None:
    lib = default_test_library()
    pulse = next(t for t in lib if t.name == "pulse_2C_10s")
    assert pulse.cost_s == pytest.approx(60.0)
    assert pulse.cost_minutes == pytest.approx(1.0)
    assert isinstance(pulse, CandidateTest)


# ---------------------------------------------------------------------------
# The finding: D-optimality optimises the wrong axis
# ---------------------------------------------------------------------------
@pytest.mark.regression
def test_d_optimality_and_ds_optimality_disagree_and_ds_is_right() -> None:
    """Pinned on one configuration, and it is the reason ``eig_target_nats`` exists.

    Ranked by total information gained, D-optimality (det FIM over every parameter) crowns
    a long cooldown, because 900 seconds sharpen R0 and capacity. Ds-optimality -- the
    target's marginal -- crowns sustained hard pulsing. The realised SNR settles it.
    """
    pack = PackTopology(4, 8)
    params = nominal_pack(pack, seed=0)
    topology = realistic_topology(pack, 4)
    baseline = pulse_train(1200.0, 1.0, mean_c_rate=0.2, pulse_c_rate=1.0)
    keep = {"pulse_2C_180s_cooldown", "pulse_train_2.5C"}
    library = tuple(t for t in default_test_library() if t.name in keep)
    target = ParameterSpec(10, ParamKind.HA)  # no thermocouple on cell 10
    specs = with_current_bias(local_specs(10))

    scores, _, _ = rank_tests(
        params, topology, NoiseModel(), specs, target, 0.40, library=library, baseline=baseline
    )
    by_d = max(scores, key=lambda s: s.eig_nats)
    by_ds = max(scores, key=lambda s: s.eig_target_nats)

    assert by_d.name == "pulse_2C_180s_cooldown"
    assert by_ds.name == "pulse_train_2.5C"
    assert by_ds.snr_after > by_d.snr_after, (
        "Ds-optimality must pick the test that actually resolves the hypothesis: "
        f"{by_ds.name} {by_ds.snr_after:.2f}s vs {by_d.name} {by_d.snr_after:.2f}s"
    )
