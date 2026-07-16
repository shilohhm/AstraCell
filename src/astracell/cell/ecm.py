"""Thevenin equivalent-circuit model: R0 plus one *or two* RC pairs.

Sign convention: ``current_a > 0`` is **discharge**.

    V = OCV(z, T) - I * R0(T) - v1 [- v2]
    dz/dt  = -eta * I / (Q * 3600)
    dv1/dt = -v1 / (R1 * C1) + I / C1
    dv2/dt = -v2 / (R2 * C2) + I / C2      (second branch, optional)

The second RC pair (``r2_ohm`` / ``c2_farad``) is **optional and defaults to
absent**: with it unset the model is exactly the first-order Thevenin ECM every
version through v0.7 used, bit-for-bit. Set both to add a second relaxation
timescale -- the "depth" v0.8 asks about: can a richer observer express what a
real cell does to a first-order fit, or does the refusal survive model order?
The two branches are independent linear systems driven by the same current, so
their overpotentials simply superpose.

R0 carries an Arrhenius temperature dependence. This matters far more than it
looks: it is the pathway by which a purely *thermal* fault leaks into the
*voltage* channel, and therefore the reason cooling faults and resistance
faults become confounded when you have no local thermometer. See
``docs`` and ``LIMITATIONS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astracell.cell.ocv import T_REF_K, OcvCurve

FloatArray = NDArray[np.float64]

# Activation energy over the gas constant, in kelvin. A 25 K rise near room
# temperature lowers R0 by roughly 12% at this value. Order-of-magnitude
# representative; not fitted.
DEFAULT_EA_OVER_R_K: float = 1500.0


@dataclass(frozen=True)
class CellElectricalParams:
    """Per-cell electrical parameters. Every field broadcasts over cells.

    ``r2_ohm`` / ``c2_farad`` are the optional second RC branch; leave them
    ``None`` (the default) for the first-order ECM. They are validated as a
    both-or-neither pair by the owning :class:`~astracell.pack.params.PackParams`.
    """

    capacity_ah: FloatArray
    r0_ohm: FloatArray  # at T_REF_K
    r1_ohm: FloatArray
    c1_farad: FloatArray
    coulombic_efficiency: float = 1.0
    ea_over_r_k: float = DEFAULT_EA_OVER_R_K
    r2_ohm: FloatArray | None = None
    c2_farad: FloatArray | None = None

    @property
    def is_second_order(self) -> bool:
        """True when the optional second RC branch is present."""
        return self.r2_ohm is not None and self.c2_farad is not None


def r0_at_temperature(
    r0_ref_ohm: FloatArray, temp_k: FloatArray, ea_over_r_k: float = DEFAULT_EA_OVER_R_K
) -> FloatArray:
    """Arrhenius-scaled ohmic resistance. Colder cell, higher resistance."""
    return r0_ref_ohm * np.exp(ea_over_r_k * (1.0 / temp_k - 1.0 / T_REF_K))


def terminal_voltage(
    soc: FloatArray,
    v_rc: FloatArray,
    temp_k: FloatArray,
    current_a: float,
    params: CellElectricalParams,
    curve: OcvCurve,
) -> FloatArray:
    """Terminal voltage [V]. ``current_a > 0`` is discharge."""
    r0 = r0_at_temperature(params.r0_ohm, temp_k, params.ea_over_r_k)
    return curve.ocv(soc, temp_k) - current_a * r0 - v_rc


def _rc_update(
    v_rc: FloatArray, current_a: float, dt_s: float, r_ohm: FloatArray, c_farad: FloatArray
) -> FloatArray:
    """Exact update of one RC branch for piecewise-constant current.

    Unconditionally stable, so the sampling period is free to equal the sensor
    sampling period rather than being dictated by the RC time constant. Shared by
    both branches so the second pair cannot integrate differently from the first.
    """
    tau = r_ohm * c_farad
    decay = np.exp(-dt_s / tau)
    return v_rc * decay + r_ohm * (1.0 - decay) * current_a


def rc_step(
    v_rc: FloatArray, current_a: float, dt_s: float, params: CellElectricalParams
) -> FloatArray:
    """Exact update of the first RC branch (``R1``/``C1``)."""
    return _rc_update(v_rc, current_a, dt_s, params.r1_ohm, params.c1_farad)


def rc2_step(
    v_rc: FloatArray, current_a: float, dt_s: float, params: CellElectricalParams
) -> FloatArray:
    """Exact update of the optional second RC branch (``R2``/``C2``)."""
    if params.r2_ohm is None or params.c2_farad is None:
        raise ValueError("second RC branch (r2_ohm, c2_farad) is not set")
    return _rc_update(v_rc, current_a, dt_s, params.r2_ohm, params.c2_farad)


def soc_step(
    soc: FloatArray, current_a: float, dt_s: float, params: CellElectricalParams
) -> FloatArray:
    """Exact coulomb count for piecewise-constant current."""
    return soc - params.coulombic_efficiency * current_a * dt_s / (params.capacity_ah * 3600.0)
