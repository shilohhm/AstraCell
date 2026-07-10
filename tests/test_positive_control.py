"""The positive control: inject a real fault into the external plant, and see whether it comes back.

v0.3's external tests could only ever show that AstraCell refuses. This module exists because a
system that refuses *everything* passes every one of them. Split, like ``test_external_plant.py``,
into a half that needs no PyBaMM and a half that does:

* **No PyBaMM.** The algebra and the traps. That baseline subtraction and paired comparison are
  the same estimator; that the paired harness covers at nominal when the two traces agree; that
  the lack-of-fit bias does not improve when you buy better sensors; and -- load-bearing -- that
  v0.3's ``prepare_external`` gate refuses *any* residual whatsoever, which is why v0.4 could not
  simply reuse it. These run everywhere.

* **PyBaMM.** The physics. A contact resistance that is exactly an ohmic drop, a cathode
  diffusivity that is exactly nothing an ECM can see, and the four verdicts they earn.

PyBaMM's solver emits RuntimeWarnings that ``filterwarnings = error`` would promote to failures;
``pytestmark`` scopes an ignore to this module only.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from astracell.calibration import (
    CAPACITY_TARGET,
    R0_TARGET,
    build_evidence,
    build_external_observer,
    coverage_curve,
    detection_metrics,
    external_scenario,
    observer_voltage,
    paired_noise,
    prepare_external,
    pulse_profile,
    run_trials,
)
from astracell.calibration.metrics import NOMINAL_LEVELS
from astracell.cell.ocv import NMC_LIKE, ocv_from_table
from astracell.observability.bias import lack_of_fit_bias, lack_of_fit_norm
from astracell.observability.decision import VerdictKind
from astracell.observability.estimator import build_context, output_tensor
from astracell.observability.sensitivity import with_current_bias
from astracell.pack.params import PackParams
from astracell.plant import (
    HEALTHY,
    contact_resistance,
    pybamm_pseudo_ocv,
    simulate_pybamm_cell,
    slow_cathode,
)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

N_TIME = 300
CAPACITY_AH = 5.0
SOC0 = 0.9
DT_S = 1.0
#: The observer's nominal series resistance, from ``external.DEFAULT_OBSERVER_R0_OHM``. Faults are
#: injected in ohms and converted here, because the plant has no parameter called "R0".
OBSERVER_R0_OHM = 0.025


def _current() -> np.ndarray:
    return pulse_profile(CAPACITY_AH, mean_c_rate=0.5, pulse_c_rate=1.0, n_time=N_TIME)


def _residual_tensor(voltage_residual: np.ndarray) -> np.ndarray:
    """A one-cell ``(t, cell, channel)`` residual: voltage only, no temperature signal."""
    return output_tensor(voltage_residual[:, None], np.zeros((voltage_residual.size, 1)))


def _ecm_observer() -> PackParams:
    return build_external_observer(NMC_LIKE, CAPACITY_AH)


# ------------------------------------------------------- no PyBaMM: the trap v0.3 walked into


def test_the_v03_external_gates_credibility_statistic_is_pure_noise() -> None:
    # `prepare_external` without a baseline projects the very trace it is about to fit, so the
    # "structural bias" it hands the gate IS the expected estimate: b = E[delta_hat]. Then
    #
    #     SNR_total = |b + eps| / sqrt(sigma^2 + b^2),   eps ~ N(0, sigma^2)
    #     E[SNR_total^2] = (b^2 + sigma^2) / (sigma^2 + b^2) = 1     exactly, for any b
    #
    # The statistic is a pure noise statistic. It carries no information about the residual at
    # all -- an exact zero and a 1 V ramp (a 1718% capacity deviation!) give the same RMS of 1.
    # Whenever |b| >> sigma, which is the only regime in which a bias gate has a job, it
    # concentrates on 1 and REFUSE_MODEL_BIAS is certain.
    #
    # This is the whole reason v0.4 exists, and it is *correct behaviour* for v0.3, whose plant
    # was always healthy. Pinned here so nobody reuses that call on a plant that might be broken.
    observer = _ecm_observer()
    scenario = external_scenario(
        name="degenerate", observer=observer, current_a=_current(), soc0=SOC0
    )
    nominal = observer_voltage(scenario)
    t = np.arange(N_TIME)

    for residual in (
        0.02 * np.sin(t / 17.0),  # a sine wave, not a battery
        np.linspace(0.0, 1.0, N_TIME),  # an absurd 1 V ramp
        1e-6 * np.cos(t / 3.0),  # a whisper: b -> 0, and the statistic degenerates to |eps|/sigma
    ):
        prepared = prepare_external(scenario, nominal + residual)
        result = run_trials(scenario, 2000, seed=0, estimator="linear", prepared=prepared)

        # The bias *is* the expected estimate, not an independent measurement of it.
        assert prepared.target_bias == pytest.approx(
            float(result.delta_hat.mean()), abs=4.0 * result.crlb_std / np.sqrt(2000)
        )
        # RMS is 1 by the identity above, to Monte Carlo error ~ 1/sqrt(2n).
        assert float(np.sqrt(np.mean(result.snr_bias**2))) == pytest.approx(1.0, abs=0.05)

    # And where the bias dominates the noise -- every case a bias gate exists to catch -- the
    # verdict is REFUSE_MODEL_BIAS in every trial, whatever the data was.
    prepared = prepare_external(scenario, nominal + 0.02 * np.sin(t / 17.0))
    result = run_trials(scenario, 300, seed=0, estimator="linear", prepared=prepared)
    assert abs(result.bias) > 5.0 * result.crlb_std
    assert all(v is VerdictKind.REFUSE_MODEL_BIAS for v in result.verdicts)


def test_a_healthy_baseline_makes_the_gate_discriminate_again() -> None:
    # Same trace, same gate, one extra argument: the bias now comes from a *healthy* plant. It
    # stops equalling the estimate, so SNR_total is free to be something other than 1.
    observer = _ecm_observer()
    scenario = external_scenario(
        name="baselined", observer=observer, current_a=_current(), soc0=SOC0
    )
    healthy = observer_voltage(scenario)
    faulty = healthy + 0.02 * np.sin(np.arange(N_TIME) / 17.0)

    degenerate = prepare_external(scenario, faulty)
    baselined = prepare_external(scenario, faulty, baseline_voltage=healthy)

    assert abs(baselined.target_bias) < 1e-9  # an ECM cannot mismatch itself
    assert abs(degenerate.target_bias) > 1e-3
    np.testing.assert_allclose(degenerate.plant_output, baselined.plant_output)


# ------------------------------------------- no PyBaMM: the identity that collapses two methods


def test_baseline_subtraction_and_paired_comparison_are_the_same_estimator() -> None:
    # The task framed these as alternatives. They are not: the projection is linear, so
    #     FIM^-1 S^T Sig^-1 (g_f - f)  -  FIM^-1 S^T Sig^-1 (g_h - f)
    #         ==  FIM^-1 S^T Sig^-1 (g_f - g_h)
    # exactly. Use an ECM "plant" perturbed by a hand-made mismatch so no solver is needed.
    observer = _ecm_observer()
    scenario = external_scenario(
        name="identity", observer=observer, current_a=_current(), soc0=SOC0, target_index=R0_TARGET
    )
    healthy = observer_voltage(scenario) + 0.01 * np.cos(np.arange(N_TIME) / 29.0)  # a phantom
    faulty = healthy - 0.004 * _current()  # plus a real ohmic fault

    evidence = build_evidence(scenario, healthy, faulty)
    assert evidence.raw_estimate - evidence.phantom == pytest.approx(
        evidence.paired_estimate, abs=1e-12
    )


def test_the_paired_estimator_recovers_an_exact_parameter_perturbation() -> None:
    # A pure series-resistance change is exactly the ECM's R0 sensitivity (dV/dR0 = -I), so the
    # differential lies exactly in the model span: no lack of fit, and the estimate is the truth.
    #
    # "Exact" bottoms out at the finite differences in `sensitivities`, not at the physics: the fd
    # step differences two ~4 V numbers to resolve a ~1 uV change, so S carries ~1e-8 relative
    # rounding and the recovered 0.20 is good to ~2e-7. The voltage-level exactness is asserted
    # directly against PyBaMM below, where nothing is differenced.
    observer = _ecm_observer()
    delta_r = 0.20 * OBSERVER_R0_OHM
    scenario = external_scenario(
        name="exact",
        observer=observer,
        current_a=_current(),
        soc0=SOC0,
        target_index=R0_TARGET,
        fault_magnitude=delta_r / OBSERVER_R0_OHM,
    )
    healthy = observer_voltage(scenario)
    evidence = build_evidence(scenario, healthy, healthy - delta_r * _current())

    assert evidence.paired_estimate == pytest.approx(0.20, abs=1e-5)
    assert evidence.misfit_paired < 1e-3  # nothing the model cannot say, to fd precision
    assert evidence.lack_of_fit == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------- no PyBaMM: the paired harness's own control


def test_the_paired_harness_is_self_consistent_on_an_identical_pair() -> None:
    # Feed the differential path two identical traces. The change is exactly zero, so the estimate
    # is pure noise at the doubled variance, coverage must be nominal, and no trial may DIAGNOSE.
    # If this fails, every PyBaMM number below is measuring the plumbing.
    observer = _ecm_observer()
    scenario = external_scenario(
        name="null", observer=observer, current_a=_current(), soc0=SOC0, target_index=R0_TARGET
    )
    trace = observer_voltage(scenario)
    evidence = build_evidence(scenario, trace, trace)

    assert evidence.paired_estimate == pytest.approx(0.0, abs=1e-12)
    assert evidence.misfit_paired == pytest.approx(0.0, abs=1e-9)

    result = run_trials(
        evidence.paired.scenario, 3000, seed=0, estimator="linear", prepared=evidence.paired
    )
    for level, got in zip(NOMINAL_LEVELS, coverage_curve(result), strict=True):
        assert abs(got - level) < 0.03

    metrics = detection_metrics(result)
    assert metrics.is_null
    assert (
        metrics.false_positive_rate < 0.005
    )  # DIAGNOSE needs 5 sigma; the null almost never gets there


def test_paired_noise_doubles_the_measurement_variance_exactly() -> None:
    # Two independently measured traces, one difference: Cov = 2 Sigma. Scaling voltage_sigma_v by
    # sqrt(2) would leave the quantisation term behind and desynchronise the sampler from the FIM.
    from astracell.sensors.noise import NoiseModel

    noise = NoiseModel()
    doubled = paired_noise(noise)
    assert doubled.voltage_variance == pytest.approx(2.0 * noise.voltage_variance, rel=1e-15)
    assert (
        doubled.voltage_rho == noise.voltage_rho
    )  # an AR(1) difference is AR(1) with the same rho
    assert doubled.temp_variance == noise.temp_variance  # the temperature channel is uninstrumented


def test_the_paired_crlb_widens_by_root_two() -> None:
    observer = _ecm_observer()
    scenario = external_scenario(name="w", observer=observer, current_a=_current(), soc0=SOC0)
    single = build_context(
        observer,
        scenario.specs,
        scenario.current_a,
        DT_S,
        scenario.topology,
        scenario.noise,
        soc0=SOC0,
    )
    doubled = build_context(
        observer,
        scenario.specs,
        scenario.current_a,
        DT_S,
        scenario.topology,
        paired_noise(scenario.noise),
        soc0=SOC0,
    )
    np.testing.assert_allclose(doubled.crlb_std / single.crlb_std, np.sqrt(2.0), rtol=1e-12)


# ------------------------------------------------------ no PyBaMM: properties of the lack-of-fit


def test_the_lack_of_fit_bias_is_invariant_to_the_noise_scale() -> None:
    # A structural error must not shrink when you buy better sensors. ||rho~|| scales as 1/sigma
    # and the CRLB std as sigma, so their product is fixed -- the same invariance parameter_bias
    # has, and the reason this quantity may sit in the same slot in `decide`.
    observer = _ecm_observer()
    scenario = external_scenario(name="inv", observer=observer, current_a=_current(), soc0=SOC0)
    ctx = build_context(
        observer,
        scenario.specs,
        scenario.current_a,
        DT_S,
        scenario.topology,
        scenario.noise,
        soc0=SOC0,
    )
    residual = _residual_tensor(0.02 * np.sin(np.arange(N_TIME) / 13.0))

    biases = []
    for scale in (0.1, 1.0, 10.0):
        noise = replace(
            scenario.noise,
            voltage_sigma_v=scenario.noise.voltage_sigma_v * scale,
            voltage_lsb_v=scenario.noise.voltage_lsb_v * scale,
        )
        biases.append(
            lack_of_fit_bias(ctx.sens, residual, scenario.topology, noise, scenario.specs)
        )

    for other in biases[1:]:
        np.testing.assert_allclose(other, biases[0], rtol=1e-10)


def test_the_noisy_lack_of_fit_norm_debiases_to_the_noiseless_one() -> None:
    # `prepare_paired` computes the norm on a noiseless differential, as `structural_residual`
    # does. A deployed estimator holds only noisy traces, and must subtract the chi^2 dof its own
    # noise contributes. Check that the deployable form recovers the one we quote.
    observer = _ecm_observer()
    scenario = external_scenario(name="dof", observer=observer, current_a=_current(), soc0=SOC0)
    ctx = build_context(
        observer,
        scenario.specs,
        scenario.current_a,
        DT_S,
        scenario.topology,
        scenario.noise,
        soc0=SOC0,
    )
    signal = 0.02 * np.sin(np.arange(N_TIME) / 13.0)
    noiseless = lack_of_fit_norm(
        ctx.sens, _residual_tensor(signal), scenario.topology, scenario.noise, scenario.specs
    )

    rng = np.random.default_rng(0)
    dof = N_TIME - len(scenario.specs)
    sigma = np.sqrt(scenario.noise.voltage_variance)
    debiased = []
    for _ in range(20):
        noisy = signal + rng.normal(0.0, sigma, N_TIME)
        raw = lack_of_fit_norm(
            ctx.sens, _residual_tensor(noisy), scenario.topology, scenario.noise, scenario.specs
        )
        debiased.append(np.sqrt(max(0.0, raw**2 - dof)))

    assert float(np.mean(debiased)) == pytest.approx(noiseless, rel=0.01)


def test_the_lack_of_fit_norm_rejects_pack_global_nuisances() -> None:
    # A Gaussian prior makes the fit MAP, not a projection, so the residual is no longer orthogonal
    # to the span and the norm would silently mean something else. Refuse rather than mislead.
    observer = _ecm_observer()
    scenario = external_scenario(name="g", observer=observer, current_a=_current(), soc0=SOC0)
    ctx = build_context(
        observer,
        scenario.specs,
        scenario.current_a,
        DT_S,
        scenario.topology,
        scenario.noise,
        soc0=SOC0,
    )
    with pytest.raises(ValueError, match="admits no prior"):
        lack_of_fit_norm(
            ctx.sens,
            _residual_tensor(np.zeros(N_TIME)),
            scenario.topology,
            scenario.noise,
            with_current_bias(scenario.specs),
        )


def test_on_a_null_scenario_false_positives_and_harmful_overclaims_coincide() -> None:
    # DIAGNOSE demands |z| >= 5; covering a truth of zero at 95% demands |z| <= 1.96. No trial can
    # be both, so on a null every false positive is a harmful overclaim and vice versa. This is an
    # identity, not a measurement, and asserting it guards the two definitions against drifting.
    observer = _ecm_observer()
    scenario = external_scenario(
        name="null", observer=observer, current_a=_current(), soc0=SOC0, target_index=R0_TARGET
    )
    trace = observer_voltage(scenario)
    evidence = build_evidence(scenario, trace, trace)
    result = run_trials(
        evidence.paired.scenario, 2000, seed=1, estimator="linear", prepared=evidence.paired
    )

    metrics = detection_metrics(result)
    assert metrics.false_positive_rate == metrics.harmful_overclaim_rate


def test_the_paired_pipeline_is_bit_reproducible_under_seed() -> None:
    observer = _ecm_observer()
    scenario = external_scenario(
        name="seed", observer=observer, current_a=_current(), soc0=SOC0, target_index=R0_TARGET
    )
    healthy = observer_voltage(scenario)
    faulty = healthy - 0.005 * _current()
    evidence = build_evidence(scenario, healthy, faulty)

    first = run_trials(
        evidence.paired.scenario, 50, seed=7, estimator="linear", prepared=evidence.paired
    )
    second = run_trials(
        evidence.paired.scenario, 50, seed=7, estimator="linear", prepared=evidence.paired
    )
    np.testing.assert_array_equal(first.delta_hat, second.delta_hat)
    assert first.verdicts == second.verdicts


# ------------------------------------------------------------------- PyBaMM: the injected physics


@pytest.fixture(scope="module")
def observer() -> PackParams:
    """PyBaMM's healthy pseudo-OCV and the one-cell ECM that borrows it. Built once."""
    pytest.importorskip("pybamm")
    soc, ocv = pybamm_pseudo_ocv(model="SPMe", n_points=150)
    curve = ocv_from_table(soc, ocv, name="pybamm", chemistry="SPMe/Chen2020")
    return build_external_observer(curve, CAPACITY_AH)


@pytest.fixture(scope="module")
def healthy_voltage() -> np.ndarray:
    pytest.importorskip("pybamm")
    return simulate_pybamm_cell(_current(), DT_S, model="SPMe", soc0=SOC0, fault=HEALTHY).voltage_v


def _evidence(observer: PackParams, healthy: np.ndarray, fault, magnitude: float, target: int):  # type: ignore[no-untyped-def]
    faulty = simulate_pybamm_cell(_current(), DT_S, model="SPMe", soc0=SOC0, fault=fault).voltage_v
    scenario = external_scenario(
        name=fault.name,
        observer=observer,
        current_a=_current(),
        soc0=SOC0,
        target_index=target,
        fault_magnitude=magnitude,
    )
    return build_evidence(scenario, healthy, faulty)


@pytest.mark.pybamm
def test_the_external_traces_are_reproducible(healthy_voltage: np.ndarray) -> None:
    # PyBaMM's solve is deterministic; the whole paired construction is worthless if it is not.
    again = simulate_pybamm_cell(_current(), DT_S, model="SPMe", soc0=SOC0, fault=HEALTHY)
    np.testing.assert_array_equal(again.voltage_v, healthy_voltage)
    assert again.fault is HEALTHY


@pytest.mark.pybamm
def test_the_injected_contact_resistance_is_exactly_an_ohmic_drop(
    healthy_voltage: np.ndarray,
) -> None:
    # The load-bearing physical fact of the positive control: the differential is -dR * I to
    # machine precision, which is *exactly* the ECM's R0 sensitivity. That is what makes the fault
    # recoverable in principle, and what makes its recovery a control rather than an achievement.
    delta_r = 0.20 * OBSERVER_R0_OHM
    faulty = simulate_pybamm_cell(
        _current(), DT_S, model="SPMe", soc0=SOC0, fault=contact_resistance(delta_r)
    ).voltage_v
    residual = (faulty - healthy_voltage) + delta_r * _current()
    assert np.abs(residual).max() < 1e-9  # volts
    assert np.all(faulty < healthy_voltage)  # under discharge, more resistance means less voltage
    # Instantaneous, unlike the diffusivity fault above: a series resistance bites on sample zero.
    assert faulty[0] < healthy_voltage[0]


def _pseudo_ocv_shift(c_rate: float, scale: float) -> float:
    """RMS volts between the healthy and faulted slow-discharge curves at this rate."""
    soc_h, ocv_h = pybamm_pseudo_ocv(model="SPMe", n_points=60, c_rate=c_rate, fault=HEALTHY)
    soc_f, ocv_f = pybamm_pseudo_ocv(
        model="SPMe", n_points=60, c_rate=c_rate, fault=slow_cathode(scale)
    )
    return float(np.sqrt(np.mean((np.interp(soc_h, soc_f, ocv_f) - ocv_h) ** 2)))


@pytest.mark.pybamm
def test_the_slow_cathode_shift_is_kinetic_so_the_capacity_truth_is_zero() -> None:
    # The confounder's honesty rests on its true (R0, capacity) deviation being zero, and it would
    # be circular to assert that from the parameter we set. Test it instead.
    #
    # A first draft of this test claimed the C/20 pseudo-OCV was unchanged. It is not: it moves
    # 9.6 mV RMS, because C/20 is not slow enough to equilibrate a 3.3x-slower cathode. But the
    # shift is an *overpotential*, and an overpotential is proportional to current. Halve the rate
    # and it halves (measured: 199 mV per C, over a 10x rate range), so it extrapolates to zero
    # and the equilibrium OCV-SOC relation -- hence the coulombic capacity -- is untouched.
    #
    # A thermodynamic capacity loss would not do this: it would shift the curve by a fixed amount
    # at every rate, including zero.
    fast, slow = _pseudo_ocv_shift(0.02, 0.3), _pseudo_ocv_shift(0.01, 0.3)
    assert fast / slow == pytest.approx(2.0, rel=0.1)  # linear in current, therefore kinetic
    assert slow < 3e-3  # and already small at C/100

    # Yet under a pulsed 0.5C load it is 10s of mV: the fault is dynamics, and dynamics is exactly
    # what a one-RC ECM cannot express. That is what makes it the confounder.
    faulty = simulate_pybamm_cell(
        _current(), DT_S, model="SPMe", soc0=SOC0, fault=slow_cathode(0.3)
    ).voltage_v
    healthy = simulate_pybamm_cell(_current(), DT_S, model="SPMe", soc0=SOC0).voltage_v

    # A diffusivity needs time to act: at t = 0 both cells sit at the same equilibrium, and only
    # then do they part. A contact resistance, by contrast, bites on the first sample -- which is
    # the whole reason one is expressible by an ECM's R0 and the other is not.
    assert faulty[0] == pytest.approx(healthy[0], abs=1e-12)
    assert np.all(faulty[1:] < healthy[1:])  # slower transport, more overpotential, less voltage
    assert float(np.sqrt(np.mean((faulty - healthy) ** 2))) > 0.01


@pytest.mark.pybamm
def test_the_healthy_phantom_fault_is_never_diagnosed(
    observer: PackParams, healthy_voltage: np.ndarray
) -> None:
    # v0.3's result, restated as a detection claim: fitting a healthy PyBaMM cell manufactures a
    # large capacity deviation, and no trial may call it a fault.
    evidence = _evidence(observer, healthy_voltage, HEALTHY, 0.0, CAPACITY_TARGET)
    assert abs(evidence.phantom) > 0.10  # the phantom is real and large

    raw = detection_metrics(run_trials(evidence.raw.scenario, 200, seed=0, prepared=evidence.raw))
    assert raw.is_null
    assert raw.diagnosis_rate == 0.0
    assert raw.false_positive_rate == 0.0

    # And the paired path, given an identical baseline, simply sees nothing -- which it should.
    paired = detection_metrics(
        run_trials(evidence.paired.scenario, 200, seed=0, prepared=evidence.paired)
    )
    assert paired.false_positive_rate == 0.0
    assert paired.coverage > 0.90


@pytest.mark.pybamm
def test_a_strong_injected_fault_is_diagnosed_with_calibrated_coverage(
    observer: PackParams, healthy_voltage: np.ndarray
) -> None:
    # The positive control proper. A 20% series-resistance rise, recovered through a -67% capacity
    # phantom, with intervals that cover at their nominal rate.
    delta_r = 0.20 * OBSERVER_R0_OHM
    evidence = _evidence(observer, healthy_voltage, contact_resistance(delta_r), 0.20, R0_TARGET)
    assert evidence.paired_estimate == pytest.approx(0.20, abs=1e-6)
    assert evidence.misfit_paired < 1e-3  # the ECM reproduces the change exactly

    metrics = detection_metrics(
        run_trials(evidence.paired.scenario, 400, seed=0, prepared=evidence.paired)
    )
    assert metrics.diagnosis_rate == 1.0
    assert metrics.true_positive_rate > 0.90
    assert metrics.coverage == pytest.approx(0.95, abs=0.04)


@pytest.mark.pybamm
def test_the_raw_path_diagnoses_the_strong_fault_at_the_wrong_magnitude(
    observer: PackParams, healthy_voltage: np.ndarray
) -> None:
    # A doubled series resistance is loud enough to clear even the phantom's gate -- and the raw
    # path then reports 91.4% against a truth of 100%, an 8-point miss inside a 0.12% interval.
    # Every one of those DIAGNOSE verdicts is a harmful overclaim. Diagnosis is not detection.
    delta_r = 1.00 * OBSERVER_R0_OHM
    evidence = _evidence(observer, healthy_voltage, contact_resistance(delta_r), 1.00, R0_TARGET)
    metrics = detection_metrics(
        run_trials(evidence.raw.scenario, 200, seed=0, prepared=evidence.raw)
    )

    assert metrics.diagnosis_rate > 0.95
    assert metrics.true_positive_rate == 0.0  # not one interval covers the truth
    assert metrics.harmful_overclaim_rate > 0.95
    assert evidence.raw_estimate == pytest.approx(1.00 + evidence.phantom, abs=1e-6)

    # The paired path, on the same data, recovers it.
    paired = detection_metrics(
        run_trials(evidence.paired.scenario, 200, seed=0, prepared=evidence.paired)
    )
    assert paired.true_positive_rate > 0.90


@pytest.mark.pybamm
def test_a_weak_injected_fault_is_not_diagnosed(
    observer: PackParams, healthy_voltage: np.ndarray
) -> None:
    # 0.05 mOhm: real, and about 2.4 sigma above the paired noise floor. Weak evidence or refusal,
    # never a diagnosis. The transition region is where an honest detector must live.
    delta_r = 0.002 * OBSERVER_R0_OHM
    evidence = _evidence(observer, healthy_voltage, contact_resistance(delta_r), 0.002, R0_TARGET)
    metrics = detection_metrics(
        run_trials(evidence.paired.scenario, 400, seed=0, prepared=evidence.paired)
    )

    assert metrics.diagnosis_rate < 0.05
    assert metrics.weak_rate + metrics.refusal_rate > 0.95
    assert metrics.coverage == pytest.approx(0.95, abs=0.04)  # the interval is still honest


@pytest.mark.pybamm
def test_the_confounded_fault_is_refused(observer: PackParams, healthy_voltage: np.ndarray) -> None:
    # A real degradation the ECM cannot express. Variance alone screams "resistance fault" at tens
    # of sigma. The differential lack-of-fit says no parameter setting produces this change, and
    # the gate refuses. True (R0, capacity) deviation: zero.
    evidence = _evidence(observer, healthy_voltage, slow_cathode(0.3), 0.0, R0_TARGET)

    # The gating inequality, stated in parameter units so it does not depend on the window length:
    # what the model cannot explain is larger than the fault it wants to claim.
    assert evidence.lack_of_fit > abs(evidence.paired_estimate)
    assert abs(evidence.paired_estimate) > 0.02  # a confident, false, resistance fault
    assert abs(evidence.paired_estimate) / evidence.paired.target_sigma > 20.0  # and a loud one

    result = run_trials(evidence.paired.scenario, 200, seed=0, prepared=evidence.paired)
    metrics = detection_metrics(result)
    assert metrics.is_null
    assert metrics.false_positive_rate == 0.0
    assert metrics.refusal_rate > 0.95
    assert all(v is VerdictKind.REFUSE_MODEL_BIAS for v in result.verdicts)

    # Without the gate the same data is a confident false positive in every trial.
    ungated = replace(evidence.paired.scenario, use_bias_gate=False)
    naive = detection_metrics(run_trials(ungated, 200, seed=0, prepared=evidence.paired))
    assert naive.false_positive_rate > 0.95


@pytest.mark.pybamm
@pytest.mark.regression
def test_the_confounded_capacity_hypothesis_clears_the_gate_by_almost_nothing(
    observer: PackParams,
) -> None:
    # The honest caveat, pinned so it cannot quietly stop being true. Swap the hypothesis from
    # resistance to capacity and the same confounder lands at 1.98 sigma -- refused, by 1%. The
    # screen is doing real work here and has no margin left; a marginally different threshold, or
    # diffusivity, or window, flips it to WEAK_EVIDENCE.
    #
    # And the window matters: at the 300 s window the rest of this module uses, the very same
    # hypothesis reaches 3.35 sigma and *is* only weakened, not refused. So this pins the 600 s
    # money window that `examples/07` and `docs/POSITIVE_CONTROL.md` quote, and the discrepancy is
    # the finding, not an inconsistency. The R0 hypothesis above refuses at both windows.
    current = pulse_profile(CAPACITY_AH, mean_c_rate=0.5, pulse_c_rate=1.0, n_time=600)
    healthy = simulate_pybamm_cell(current, DT_S, model="SPMe", soc0=SOC0).voltage_v
    faulty = simulate_pybamm_cell(
        current, DT_S, model="SPMe", soc0=SOC0, fault=slow_cathode(0.3)
    ).voltage_v
    scenario = external_scenario(
        name="confounded/capacity",
        observer=observer,
        current_a=current,
        soc0=SOC0,
        target_index=CAPACITY_TARGET,
    )
    evidence = build_evidence(scenario, healthy, faulty)

    snr_total = abs(evidence.paired_estimate) / float(
        np.hypot(evidence.paired.target_sigma, evidence.lack_of_fit)
    )
    assert 1.9 < snr_total < 2.0  # refused, and only just
