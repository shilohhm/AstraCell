"""v0.9 -- fitting the fast RC branch: sensitivities first, then the positive control.

v0.8 measured that a *fixed* second RC branch is invisible to the ``(R0, capacity)`` fit -- its
overpotential cancels in ``dV/dR0`` and ``dV/dQ`` (``test_second_order.py`` and the depth section of
``examples/08``). v0.9 stops holding the branch fixed and *fits* it: the paired estimator goes from
``(R0, capacity)`` to ``(R0, capacity, R1, C1)``. These tests pin the two things that must be true
before any real-cell number from that fit means anything:

* **The estimator can see the branch.** ``dV/dR1`` and ``dV/dC1`` are real, correctly signed, and --
  for ``C1`` -- present only while the excitation keeps the branch in transient. Under a
  near-constant 1C discharge (the Oxford regime) ``dV/dR1`` saturates to ``-I*R1``, collinear with
  ``dV/dR0``, and ``dV/dC1`` vanishes. That is the structural reason v0.9's honest bet is H2
  (confounding), and it is measured here rather than asserted.

This module grows a second half -- the pulse-excitation positive control -- once the sensitivities
are pinned. A refusal on a real cell is uninterpretable unless the same machinery provably
*succeeds* when the dynamics are identifiable; that is the v0.4 lesson, built before any real cell.
"""

from __future__ import annotations

import numpy as np
import pytest

from astracell.calibration import (
    C1_TARGET,
    R1_TARGET,
    PairedEvidence,
    build_evidence,
    build_external_observer,
    constant_profile,
    detection_metrics,
    external_scenario,
    external_specs_4param,
    pulse_profile,
    run_trials,
)
from astracell.cell.ocv import NMC_LIKE, OcvCurve, ocv_from_table
from astracell.observability.decision import VerdictKind
from astracell.observability.estimator import fit_gauss_newton
from astracell.observability.sensitivity import (
    VOLTAGE_CHANNEL,
    ParameterSpec,
    ParamKind,
    sensitivities,
)
from astracell.pack import PackTopology
from astracell.pack.params import PackParams
from astracell.pack.simulate import simulate

ONE = PackTopology(n_modules=1, cells_per_module=1)
DT = 1.0
SOC0 = 0.9

# A long, purely constant 1C discharge, so the RC branch saturates well inside the window: with
# tau = R1*C1 = 30 s, 400 s is >13 time constants and v1 -> I*R1 to ~2e-6.
N_CONST = 400
R0_OHM = 0.02
R1_OHM = 0.010
C1_FARAD = 3000.0
CAPACITY_AH = 3.0
CONST_CURRENT = np.full(N_CONST, CAPACITY_AH, dtype=float)  # exactly 1C


def _one(value: float) -> np.ndarray:
    return np.array([value], dtype=float)


def _flat_curve() -> OcvCurve:
    """A smooth, gently sloped OCV so the sensitivities are clean and SOC-drift is mild."""
    soc = np.linspace(0.0, 1.0, 64)
    ocv = 3.2 + 0.9 * soc
    return ocv_from_table(soc, ocv, name="flat_v09", chemistry="synthetic linear (test only)")


def _pack() -> PackParams:
    return PackParams(
        topology=ONE,
        curve=_flat_curve(),
        capacity_ah=_one(CAPACITY_AH),
        r0_ohm=_one(R0_OHM),
        r1_ohm=_one(R1_OHM),
        c1_farad=_one(C1_FARAD),
        heat_capacity_j_per_k=_one(900.0),
        ha_w_per_k=_one(2.5),
        ea_over_r_k=0.0,  # isothermal: R0 does not move, so the RC branch is the only R1/C1 path
    )


def _pulse_current(n: int) -> np.ndarray:
    """A 60 s-period pulse train that keeps the RC branch in perpetual transient."""
    from astracell.calibration.external import pulse_profile

    return pulse_profile(
        CAPACITY_AH, mean_c_rate=0.4, pulse_c_rate=1.0, n_time=n, period_s=60.0, duty=0.25
    )


def _four_param_specs() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec(0, ParamKind.R0),
        ParameterSpec(0, ParamKind.CAPACITY),
        ParameterSpec(0, ParamKind.R1),
        ParameterSpec(0, ParamKind.C1),
    )


# --------------------------------------------------------------- R1/C1 are ordinary cell params


def test_r1_and_c1_are_perturbable_cell_parameters() -> None:
    """The two new fitted dynamics parameters map to real PackParams fields and label cleanly.

    ``ParamKind.value`` is the field name ``perturb``/``scale_cell`` scales, so mapping R1/C1 onto
    the existing ``r1_ohm``/``c1_farad`` fields is all the plumbing the perturbation needs.
    """
    r1 = ParameterSpec(0, ParamKind.R1)
    c1 = ParameterSpec(0, ParamKind.C1)
    assert ParamKind.R1.value == "r1_ohm"
    assert ParamKind.C1.value == "c1_farad"
    assert not r1.is_global and not c1.is_global
    assert r1.label() == "R1[0]"
    assert c1.label() == "C1[0]"
    assert r1.unit() == "relative" and c1.unit() == "relative"


# --------------------------------------------------------------- the branch is visible, correctly


def test_dv_dr1_saturates_to_minus_i_r1_under_constant_current() -> None:
    """At steady state v1 = I*R1, so the *relative* sensitivity dV/dR1 is exactly ``-I*R1``.

    This is the load-bearing physics of H2: under a constant 1C discharge the R1 column collapses
    onto the same ``-I`` shape as the R0 column (``dV/dR0 = -I*R0``), so R1 and R0 are collinear and
    the fit cannot separate them without richer excitation. An exact identity here, not a bound.
    """
    specs = _four_param_specs()
    sens = sensitivities(_pack(), CONST_CURRENT, DT, specs, soc0=SOC0)
    dv_dr1 = sens[:, 0, VOLTAGE_CHANNEL, 2]  # index 2 == R1 in _four_param_specs

    i_last = float(CONST_CURRENT[-1])
    assert dv_dr1[-1] == pytest.approx(-i_last * R1_OHM, rel=1e-3)
    # Zero at t=0 -- the branch has not charged, so v1 is 0 regardless of R1 -- and strictly
    # negative once it has: more series polarisation can only drop terminal voltage on discharge.
    assert dv_dr1[0] == 0.0
    assert np.all(dv_dr1[1:] < 0.0)


def test_dv_dc1_vanishes_at_steady_state_but_lives_in_the_transient() -> None:
    """C1 sets *how fast* v1 approaches I*R1, not its steady value -- so dV/dC1 -> 0 once saturated.

    Its whole signal is the transient: nonzero early, negligible late. That is exactly why a
    near-constant discharge cannot identify C1 (H2), and why the positive control must pulse.
    """
    specs = _four_param_specs()
    sens = sensitivities(_pack(), CONST_CURRENT, DT, specs, soc0=SOC0)
    dv_dr1 = sens[:, 0, VOLTAGE_CHANNEL, 2]
    dv_dc1 = sens[:, 0, VOLTAGE_CHANNEL, 3]  # index 3 == C1

    # Saturated: C1 is invisible at the last sample, orders of magnitude below the R1 signal there.
    assert abs(dv_dc1[-1]) < 1e-3 * abs(dv_dr1[-1])
    # Alive in the transient: the branch charges over the first few tau, so C1 is seen early.
    assert abs(dv_dc1[:60]).max() > 1e-4


def test_pulse_excitation_keeps_c1_visible_where_constant_current_loses_it() -> None:
    """Identifiability contrast in raw sensitivity: pulses re-excite C1, constant current cannot.

    Measured deep in the window (t >= 300 s = 10 tau), where a constant discharge has let C1 go dark
    (|dV/dC1| ~ 1.4e-5) while a pulse train re-opens the transient at every edge (~6.3e-3): a ~460x
    gap on this pack. Asserted at 50x, with margin, so it pins the mechanism and not the exact
    ratio. This is the excitation axis the positive control rides on.
    """
    specs = _four_param_specs()
    late = slice(300, None)  # >= 10 tau: constant-current C1 has fully decayed, a pulse's has not

    const = sensitivities(_pack(), CONST_CURRENT, DT, specs, soc0=SOC0)
    pulse_i = _pulse_current(N_CONST)
    pulse = sensitivities(_pack(), pulse_i, DT, specs, soc0=SOC0)

    c1_const_late = np.abs(const[late, 0, VOLTAGE_CHANNEL, 3]).max()
    c1_pulse_late = np.abs(pulse[late, 0, VOLTAGE_CHANNEL, 3]).max()
    assert c1_pulse_late > 50.0 * c1_const_late


# ========================================================= the pulse-excitation positive control
#
# A refusal on a real cell means nothing unless the same 4-parameter machinery provably *succeeds*
# when the dynamics are identifiable -- the v0.4 lesson. These build ECM-on-ECM pairs (a known R1 or
# C1 change injected into the observer's own model) and check that under a pulse train the fit
# recovers it, while under the 1C discharge that confounds R1 with R0 the gate refuses instead.
# Every threshold below is measured, not guessed.

PC_CAP = 5.0
PC_SOC0 = 0.9
PC_N = 600


def _pc_observer() -> PackParams:
    """The one-cell ECM observer the positive control fits: R0 = 25 mOhm, tau1 = R1*C1 = 30 s."""
    return build_external_observer(NMC_LIKE, PC_CAP)


def _pc_pulse() -> np.ndarray:
    """Rich excitation: a 0.5C/1C pulse train that keeps the RC branch in perpetual transient."""
    return pulse_profile(PC_CAP, mean_c_rate=0.5, pulse_c_rate=1.0, n_time=PC_N)


def _pc_const() -> np.ndarray:
    """Poor excitation: a 1C discharge, the Oxford regime, where dV/dR1 collapses onto dV/dR0."""
    return constant_profile(PC_CAP, c_rate=1.0, n_time=PC_N)


def _ecm_pair(kind: str, delta: float, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Healthy and faulted ECM voltage traces; the fault is a relative change to a branch param."""
    healthy = _pc_observer()
    faulted = healthy.scale_cell(kind, 0, 1.0 + delta)
    hv = np.asarray(simulate(healthy, current, DT, soc0=PC_SOC0).voltage_v[:, 0], dtype=float)
    fv = np.asarray(simulate(faulted, current, DT, soc0=PC_SOC0).voltage_v[:, 0], dtype=float)
    return hv, fv


def _pc_evidence(kind: str, target: int, delta: float, current: np.ndarray) -> PairedEvidence:
    """The deployable 4-param paired evidence for an injected branch fault under one excitation."""
    hv, fv = _ecm_pair(kind, delta, current)
    scenario = external_scenario(
        name=f"pc_{kind}",
        observer=_pc_observer(),
        current_a=current,
        soc0=PC_SOC0,
        specs=external_specs_4param(),
        target_index=target,
        fault_magnitude=delta,
    )
    return build_evidence(scenario, hv, fv)


def test_the_four_param_estimator_is_unbiased_on_matched_pulse_data() -> None:
    """The keystone: under rich excitation the 4-param fit returns injected R1/C1, and nothing else.

    ECM-on-ECM, so the change is exactly expressible and a correct estimator must recover it with no
    bias of its own -- mirroring v0.8's ``test_second_order_adapter_pair_is_self_consistent``. The
    nonlinear fit is the honest instrument: R1 and C1 enter V through tau = R1*C1, so a large step
    carries curvature the *linear* fit only tracks approximately (that cost is a later test). Both
    branch parameters are identifiable under this pulse train -- VIF well below 10.
    """
    pulse = _pc_pulse()
    r1 = _pc_evidence("r1_ohm", R1_TARGET, 0.15, pulse)
    gn_r1 = fit_gauss_newton(r1.paired.fit_ctx, r1.paired.plant_output, max_iter=50)
    assert gn_r1.converged
    assert gn_r1.delta[R1_TARGET] == pytest.approx(0.15, abs=1e-4)
    assert np.abs(np.delete(gn_r1.delta, R1_TARGET)).max() < 1e-3  # no cross-talk into R0/Q/C1

    c1 = _pc_evidence("c1_farad", C1_TARGET, 0.30, pulse)
    gn_c1 = fit_gauss_newton(c1.paired.fit_ctx, c1.paired.plant_output, max_iter=50)
    assert gn_c1.converged
    assert gn_c1.delta[C1_TARGET] == pytest.approx(0.30, abs=1e-4)
    assert np.abs(np.delete(gn_c1.delta, C1_TARGET)).max() < 1e-3

    assert r1.paired.fit_ctx.vif[R1_TARGET] < 10.0
    assert c1.paired.fit_ctx.vif[C1_TARGET] < 10.0


def test_a_pulse_identifies_the_r1_that_a_constant_discharge_confounds() -> None:
    """The positive control and its shadow: one injected R1 fault, diagnosed under pulses, refused
    under a 1C discharge -- because there dV/dR1 collapses onto dV/dR0 (VIF ~ 67 > 10).

    This is what makes a real-cell refusal interpretable. The machinery is not refusing everything:
    hand it excitation that separates R1 from R0 and it diagnoses the very fault it just refused.
    """
    delta = 0.15
    pulse_ev = _pc_evidence("r1_ohm", R1_TARGET, delta, _pc_pulse())
    const_ev = _pc_evidence("r1_ohm", R1_TARGET, delta, _pc_const())

    assert pulse_ev.paired.fit_ctx.vif[R1_TARGET] < 10.0
    assert const_ev.paired.fit_ctx.vif[R1_TARGET] > 10.0  # measured ~67: R1 and R0 collinear

    pulse_res = run_trials(pulse_ev.paired.scenario, 200, seed=0, prepared=pulse_ev.paired)
    const_res = run_trials(const_ev.paired.scenario, 200, seed=0, prepared=const_ev.paired)

    assert detection_metrics(pulse_res).diagnosis_rate == 1.0
    assert all(v is VerdictKind.REFUSE_CONFOUNDED for v in const_res.verdicts)


def test_the_deployable_linear_estimate_recovers_r1_within_curvature() -> None:
    """The real-cell loop fits *linearly*, so pin what that costs: the linear paired estimate
    recovers an injected R1 change to within its curvature -- close, biased low, never exact.

    Measured: a +15% R1 injection returns ~+14.5% (a ~3% low bias), tracking the nonlinear fit to
    under a point. Honest consequence -- the linear interval is curvature-limited, not nominal, so
    the exactness above is quoted from Gauss-Newton and the loop will report both.
    """
    ev = _pc_evidence("r1_ohm", R1_TARGET, 0.15, _pc_pulse())
    gn = fit_gauss_newton(ev.paired.fit_ctx, ev.paired.plant_output, max_iter=50)

    assert ev.paired_estimate == pytest.approx(0.15, rel=0.05)  # within curvature
    assert ev.paired_estimate < 0.15  # systematically low: tau curvature, not noise
    assert abs(ev.paired_estimate - gn.delta[R1_TARGET]) < 0.01  # linear tracks nonlinear
    assert ev.misfit_paired < 1.0  # expressible: misfit is curvature, not omitted dynamics


def test_the_four_param_paired_path_is_null_on_identical_traces() -> None:
    """Feed the 4-param differential two identical traces: every estimate and the misfit are zero.

    Hygiene: the added R1/C1 columns must not manufacture a phantom on a null, as the 2-param path
    does not. ``delta = 0`` scales R1 by 1.0, so the healthy and faulted traces are bit-identical.
    """
    ev = _pc_evidence("r1_ohm", R1_TARGET, 0.0, _pc_pulse())
    assert ev.paired_estimate == pytest.approx(0.0, abs=1e-9)
    assert ev.misfit_paired == pytest.approx(0.0, abs=1e-9)
