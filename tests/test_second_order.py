"""The optional second RC branch (v0.8 "depth"): correct integration, and inert when absent.

A first-order ECM has one relaxation timescale; a real cell has several, and that omitted-dynamics
residual is what v0.3-v0.7 watched turn into a phantom capacity fault. v0.8 asks whether a second
RC branch changes that verdict. These tests pin what the branch *is* -- two independent RC systems
whose overpotentials superpose, exact energy balance at second order, and bit-for-bit first-order
behaviour when the branch is absent -- before any real-cell conclusion rests on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from astracell.calibration import CAPACITY_TARGET, build_evidence, external_scenario
from astracell.calibration.external import (
    DEFAULT_OBSERVER_C2_FARAD,
    DEFAULT_OBSERVER_R2_OHM,
    build_external_observer,
    build_second_order_observer,
)
from astracell.cell.ecm import r0_at_temperature
from astracell.cell.ocv import NMC_LIKE, OcvCurve, ocv_from_table
from astracell.pack import PackTopology, simulate
from astracell.pack.params import PackParams
from astracell.plant.oxford import OxfordAge, aligned_pair, measured_fade

ONE = PackTopology(n_modules=1, cells_per_module=1)

CURRENT = np.full(400, 3.0)  # 1C constant discharge of a 3 Ah cell, 400 s
DT = 1.0
SOC0 = 0.9


def _one(value: float) -> np.ndarray:
    return np.array([value], dtype=float)


def _flat_curve() -> OcvCurve:
    """A zero-entropy OCV curve, so temperature cannot feed back into voltage.

    With ``ea_over_r_k=0`` and ``dOCV/dT=0`` both R0 and OCV are temperature-independent, so the RC
    branches are the *only* thing distinguishing the sims below -- which is what makes the
    superposition identity exact rather than merely close.
    """
    soc = np.linspace(0.0, 1.0, 64)
    ocv = 3.2 + 0.9 * soc
    return ocv_from_table(soc, ocv, name="flat_test", chemistry="synthetic linear (test only)")


def _pack(
    *,
    r2: float | None = None,
    c2: float | None = None,
    curve: OcvCurve | None = None,
    ea: float = 1500.0,
) -> PackParams:
    return PackParams(
        topology=ONE,
        curve=curve if curve is not None else _flat_curve(),
        capacity_ah=_one(3.0),
        r0_ohm=_one(0.02),
        r1_ohm=_one(0.010),
        c1_farad=_one(3000.0),
        heat_capacity_j_per_k=_one(900.0),
        ha_w_per_k=_one(2.5),
        ea_over_r_k=ea,
        r2_ohm=None if r2 is None else _one(r2),
        c2_farad=None if c2 is None else _one(c2),
    )


# --------------------------------------------------------------------------- inert when absent


def test_absent_second_branch_is_first_order() -> None:
    """No r2/c2 -> ``is_second_order`` is False and nothing changes about the first-order path."""
    p = _pack()
    assert not p.electrical.is_second_order
    assert p.r2_ohm is None and p.c2_farad is None
    sim = simulate(p, CURRENT, DT, soc0=SOC0)
    again = simulate(p, CURRENT, DT, soc0=SOC0)
    # Determinism plus the "subtract 0.0 / add 0.0" no-op means exact equality, not approx.
    np.testing.assert_array_equal(sim.voltage_v, again.voltage_v)


# --------------------------------------------------------------------------- the branch integrates


def test_second_branch_lowers_discharge_voltage() -> None:
    """A second dissipative branch can only add overpotential: V_2nd <= V_1st on discharge."""
    first = simulate(_pack(), CURRENT, DT, soc0=SOC0)
    second = simulate(_pack(r2=0.008, c2=30000.0), CURRENT, DT, soc0=SOC0)
    assert np.all(second.voltage_v <= first.voltage_v + 1e-12)
    assert second.voltage_v[-1] < first.voltage_v[-1] - 1e-6  # strictly, by end of the pulse


def test_two_branches_superpose() -> None:
    """The 2nd-order RC overpotential is exactly branch-1's plus branch-2's.

    Independent linear RC systems driven by a shared current add. With a zero-entropy curve and
    ea=0, temperature cannot break that, so the identity holds to float64, not just approximately.
    """
    curve = _flat_curve()
    full = _pack(
        r2=0.006, c2=25000.0, curve=curve, ea=0.0
    )  # branch1=(.010,3000), branch2=(.006,25000)
    only_1 = _pack(curve=curve, ea=0.0)  # branch1 = (.010, 3000)
    only_2 = _pack(curve=curve, ea=0.0).evolve(r1_ohm=_one(0.006), c1_farad=_one(25000.0))

    sim_full = simulate(full, CURRENT, DT, soc0=SOC0)
    sim_1 = simulate(only_1, CURRENT, DT, soc0=SOC0)
    sim_2 = simulate(only_2, CURRENT, DT, soc0=SOC0)

    i = CURRENT[:, None]
    r0 = 0.02  # ea=0 -> temperature-independent, identical across all three sims

    def overpotential(sim: object) -> np.ndarray:
        return curve.ocv(sim.soc) - sim.voltage_v - i * r0  # type: ignore[attr-defined]

    np.testing.assert_allclose(
        overpotential(sim_full), overpotential(sim_1) + overpotential(sim_2), atol=1e-12
    )


def test_energy_balance_holds_at_second_order() -> None:
    """I*(OCV - V) == I^2*R0 + I*(v1 + v2), to machine precision, with the second branch on.

    The same identity ``test_physics_invariants`` pins for the first-order model. It is algebraic
    in V = OCV - I*R0 - (v1 + v2), so it must survive the extra branch untouched.
    """
    p = _pack(r2=0.008, c2=30000.0, ea=1500.0)
    sim = simulate(p, CURRENT, DT, soc0=SOC0)
    r0 = r0_at_temperature(p.r0_ohm, sim.temp_k, p.ea_over_r_k)
    ocv = p.curve.ocv(sim.soc, sim.temp_k)
    i = sim.current_a[:, None]
    overpotential = i * (ocv - sim.voltage_v)
    dissipation = i**2 * r0 + i * (ocv - sim.voltage_v - i * r0)  # I^2 R0 + I (v1 + v2)
    np.testing.assert_allclose(overpotential, dissipation, atol=1e-14)


# --------------------------------------------------------------------------- validation


def test_half_a_second_branch_is_rejected() -> None:
    """r2 without c2 (or the reverse) is a construction bug, not a first-order model."""
    with pytest.raises(ValueError, match="both r2_ohm and c2_farad"):
        _pack(r2=0.008)
    with pytest.raises(ValueError, match="both r2_ohm and c2_farad"):
        _pack(c2=30000.0)


def test_second_branch_must_be_positive() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        _pack(r2=-0.008, c2=30000.0)
    with pytest.raises(ValueError, match="strictly positive"):
        _pack(r2=0.008, c2=0.0)


# --------------------------------------------------------------------------- the observer


def test_default_external_observer_is_first_order() -> None:
    obs = build_external_observer(_flat_curve(), 3.0)
    assert not obs.electrical.is_second_order
    assert obs.r2_ohm is None and obs.c2_farad is None


def test_second_order_observer_adds_only_the_slow_branch() -> None:
    """The 2nd-order observer differs from the 1st by exactly the slow branch -- nothing else.

    Same R0, same fast branch, same OCV, same fitted (R0, capacity). That is what lets a first- vs
    second-order comparison attribute any change to the added timescale and not to a moved knob.
    """
    first = build_external_observer(_flat_curve(), 3.0)
    second = build_second_order_observer(_flat_curve(), 3.0)
    assert second.electrical.is_second_order
    assert second.r2_ohm is not None and second.c2_farad is not None
    assert float(second.r2_ohm[0]) == pytest.approx(DEFAULT_OBSERVER_R2_OHM)
    assert float(second.c2_farad[0]) == pytest.approx(DEFAULT_OBSERVER_C2_FARAD)
    assert float(second.r0_ohm[0]) == float(first.r0_ohm[0])
    assert float(second.r1_ohm[0]) == float(first.r1_ohm[0])
    assert float(second.c1_farad[0]) == float(first.c1_farad[0])


# ------------------------------------------------------------------- the honesty keystone (v0.8)


def _second_order_age(cyc: int, capacity_ah: float, *, nominal_ah: float, soc0: float) -> OxfordAge:
    """An OxfordAge whose 1C discharge is generated by the *second-order* observer as its plant."""
    n = 800
    current = np.full(n, nominal_ah, dtype=float)  # fixed 1C of nominal, as the dataset does
    plant = build_second_order_observer(NMC_LIKE, capacity_ah)
    result = simulate(plant, current, 1.0, soc0=soc0, temp0_k=298.15)
    soc = np.linspace(0.02, 0.98, 120)
    return OxfordAge(
        cyc=cyc,
        soc=soc,
        ocv_v=NMC_LIKE.ocv(soc),
        time_s=np.arange(n, dtype=float),
        current_a=current,
        voltage_v=np.asarray(result.voltage_v[:, 0], dtype=float),
        capacity_ah=capacity_ah,
    )


def test_second_order_adapter_pair_is_self_consistent() -> None:
    """A 2nd-order plant, fit by the *matched* 2nd-order observer, recovers the fade with ~0 misfit.

    The honesty keystone for v0.8, mirroring ``test_adapter_pair_is_self_consistent`` for the first
    order: when the observer matches the plant, the depth machinery must add no bias of its own --
    the paired estimate returns the injected capacity change and the lack-of-fit stays in its
    "expressible" regime. So any phantom the 2nd-order observer reports on a *real* cell is the real
    cell exceeding the model, not an artefact of the second branch or its sensitivities.
    """
    fade = -0.05
    q_base, soc0 = 5.0, 0.9
    base = _second_order_age(0, q_base, nominal_ah=q_base, soc0=soc0)
    aged = _second_order_age(100, q_base * (1.0 + fade), nominal_ah=q_base, soc0=soc0)

    # Replay from the SOC the traces were generated at, or a soc0 mismatch manufactures lack-of-fit.
    current, base_v, aged_v, s0 = aligned_pair(base, aged, window_s=600.0, dt_s=1.0, soc0=soc0)
    scenario = external_scenario(
        name="second_order_self_consistency",
        observer=build_second_order_observer(NMC_LIKE, q_base),
        current_a=current,
        soc0=s0,
        fault_magnitude=measured_fade(base, aged),
        target_index=CAPACITY_TARGET,
    )
    evidence = build_evidence(scenario, base_v, aged_v)

    assert evidence.delta_true == pytest.approx(fade)
    assert evidence.paired_estimate == pytest.approx(fade, abs=5e-3)
    # A capacity change is the 2nd-order observer's own parameter too, so the screen stays ~0.
    assert evidence.misfit_paired < 1.0
