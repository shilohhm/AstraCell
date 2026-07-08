"""Fixed-step pack simulator.

Integration scheme, and why:

* SOC and the RC branch use their **exact** discrete solutions for
  piecewise-constant current. No stability constraint, no truncation error.
* Temperature uses forward Euler. The thermal time constant here is
  ``C_th / (hA + sum_j k_ij)`` ~ 200-250 s, so a 1 s step is three orders of
  magnitude inside stability and the local error is negligible relative to the
  0.5 K sensor noise.

Consequence: the sampling period may be set equal to the sensor sampling period.
That matters, because the Fisher information scales with the number of
independent samples, so an integrator that forced ``dt << dt_sensor`` would
tempt you into computing information you never actually measured.

The simulator returns **true** states. Nothing here knows about sensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astracell.cell.ecm import r0_at_temperature, rc_step, soc_step
from astracell.cell.thermal import heat_generation, thermal_derivative
from astracell.pack.params import PackParams

FloatArray = NDArray[np.float64]

SOC_MIN: float = 0.02
SOC_MAX: float = 0.98


class SimulationError(RuntimeError):
    """Raised when the simulation leaves the region where the model is meaningful."""


def _initial_state(value: float | FloatArray, n_cells: int, name: str) -> FloatArray:
    """Broadcast a scalar or per-cell initial condition to a ``(n_cells,)`` array."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(n_cells, float(array))
    if array.shape != (n_cells,):
        raise ValueError(f"{name} has shape {array.shape}, expected scalar or ({n_cells},)")
    return np.array(array, copy=True)


@dataclass(frozen=True)
class SimResult:
    """True (noiseless) pack trajectory. Time-major: ``(n_time, n_cells)``."""

    time_s: FloatArray
    current_a: FloatArray  # (n_time,) pack current, > 0 discharge
    soc: FloatArray  # (n_time, n_cells)
    voltage_v: FloatArray  # (n_time, n_cells)
    temp_k: FloatArray  # (n_time, n_cells)

    @property
    def n_time(self) -> int:
        return self.time_s.size

    @property
    def n_cells(self) -> int:
        return self.voltage_v.shape[1]

    @property
    def dt_s(self) -> float:
        return float(self.time_s[1] - self.time_s[0])

    @property
    def pack_voltage_v(self) -> FloatArray:
        return self.voltage_v.sum(axis=1)


def simulate(
    params: PackParams,
    current_a: FloatArray,
    dt_s: float,
    *,
    soc0: float | FloatArray = 0.75,
    temp0_k: float | FloatArray | None = None,
    check_bounds: bool = True,
) -> SimResult:
    """Integrate the pack forward over the given current profile.

    Raises ``SimulationError`` if SOC leaves ``[SOC_MIN, SOC_MAX]``. We raise
    rather than clip: an out-of-range SOC means the duty cycle is wrong for this
    pack, and silently clipping would put a kink in the very sensitivities the
    observability analysis differentiates.
    """
    n = params.n_cells
    n_time = int(np.asarray(current_a).size)
    if n_time < 2:
        raise ValueError("current profile must have at least two samples")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    current = np.asarray(current_a, dtype=float)
    elec, therm = params.electrical, params.thermal
    lap = params.conductance_matrix()
    curve = params.curve

    soc = _initial_state(soc0, n, "soc0")
    v_rc = np.zeros(n)
    temp = _initial_state(params.coolant_temp_k if temp0_k is None else temp0_k, n, "temp0_k")

    soc_out = np.empty((n_time, n))
    volt_out = np.empty((n_time, n))
    temp_out = np.empty((n_time, n))

    for k in range(n_time):
        i_k = float(current[k])
        r0 = r0_at_temperature(elec.r0_ohm, temp, elec.ea_over_r_k)
        volt = curve.ocv(soc, temp) - i_k * r0 - v_rc

        # Record the state *at* sample k, before stepping to k+1.
        soc_out[k] = soc
        volt_out[k] = volt
        temp_out[k] = temp

        if k == n_time - 1:
            break

        heat = heat_generation(i_k, r0, v_rc, temp, curve.docv_dtemp(soc))
        dtemp = thermal_derivative(temp, heat, lap, therm)

        soc = soc_step(soc, i_k, dt_s, elec)
        v_rc = rc_step(v_rc, i_k, dt_s, elec)
        temp = temp + dt_s * dtemp

    if check_bounds:
        lo, hi = float(soc_out.min()), float(soc_out.max())
        if lo < SOC_MIN or hi > SOC_MAX:
            raise SimulationError(
                f"SOC left [{SOC_MIN}, {SOC_MAX}] (observed [{lo:.4f}, {hi:.4f}]). "
                "Shorten the profile, lower the mean C-rate, or move soc0."
            )

    return SimResult(
        time_s=np.arange(n_time, dtype=float) * dt_s,
        current_a=current,
        soc=soc_out,
        voltage_v=volt_out,
        temp_k=temp_out,
    )
