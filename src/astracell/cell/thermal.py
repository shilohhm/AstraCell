"""Lumped single-node thermal model per cell.

    C_th * dT/dt = Q_irrev + Q_rev - conduction_out - hA * (T - T_coolant)

Two heat terms, both kept:

``Q_irrev = I * (OCV - V) = I^2 * R0 + I * v1``
    Always non-negative for the ECM (both terms are I times something with the
    same sign as I). This identity is exact for this model and is asserted in
    ``tests/test_physics_invariants.py``.

``Q_rev = -I * T * dOCV/dT``
    The entropic term. Most hobby implementations drop it. It is 10-30% of heat
    generation at low C-rate, and critically its **sign flips between charge and
    discharge**, which is a distinguishable telemetry signature. Keeping it costs
    one line and changes the thermal residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CellThermalParams:
    """Per-cell thermal parameters. Every field broadcasts over cells."""

    heat_capacity_j_per_k: FloatArray  # C_th, lumped
    ha_w_per_k: FloatArray  # convective coupling to coolant
    coolant_temp_k: float


def irreversible_heat(current_a: float, r0_ohm: FloatArray, v_rc: FloatArray) -> FloatArray:
    """Joule + polarisation dissipation [W]. Equals I * (OCV - V) exactly."""
    return current_a * current_a * r0_ohm + current_a * v_rc


def reversible_heat(current_a: float, temp_k: FloatArray, docv_dtemp: FloatArray) -> FloatArray:
    """Entropic heat [W]. Signed: exothermic on discharge where dOCV/dT < 0."""
    return -current_a * temp_k * docv_dtemp


def heat_generation(
    current_a: float,
    r0_ohm: FloatArray,
    v_rc: FloatArray,
    temp_k: FloatArray,
    docv_dtemp: FloatArray,
) -> FloatArray:
    """Total volumetric heat generation per cell [W]."""
    return irreversible_heat(current_a, r0_ohm, v_rc) + reversible_heat(
        current_a, temp_k, docv_dtemp
    )


def thermal_derivative(
    temp_k: FloatArray,
    heat_w: FloatArray,
    conductance: NDArray[np.float64],
    thermal: CellThermalParams,
) -> FloatArray:
    """dT/dt [K/s].

    ``conductance`` is the graph Laplacian L of the cell-to-cell conduction
    network, so ``(L @ T)[i] == sum_j k_ij * (T_i - T_j)`` is the net heat
    conducted *out* of cell i.
    """
    conducted_out = conductance @ temp_k
    convected_out = thermal.ha_w_per_k * (temp_k - thermal.coolant_temp_k)
    return (heat_w - conducted_out - convected_out) / thermal.heat_capacity_j_per_k
