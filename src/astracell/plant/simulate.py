"""The plant: a higher-fidelity battery than the one the observer believes in.

The integration loop is written out again here rather than bolted onto ``pack.simulate``,
but it calls the **same cell primitives** (``r0_at_temperature``, ``rc_step``, ``soc_step``,
``heat_generation``). So be precise about what the zero-mismatch reduction proves: it shows
the extra structure is *wired to nothing* when its knobs are zero -- that no residual leaks
in from a stray term, a different heat formula, or an off-by-one in the recording order. It
does **not** independently corroborate the shared physics, and it is not, on its own, proof
that the bias numbers mean anything.

The load-bearing evidence against tautology is elsewhere, in ``tests/test_mismatch.py``:

* a residual orthogonal to the sensitivity span produces exactly zero bias, and a residual
  equal to ``S @ delta`` produces exactly bias ``delta`` -- the projection is doing algebra,
  not rubber-stamping;
* the bias is invariant under rescaling every noise sigma, so it is not a noise artifact;
* the linearised bias predicts where an actual Gauss-Newton fit of the ECM to the plant
  comes to rest. That is the claim that matters, and it is checked against a real fit.

Integration mirrors the observer where the observer is right:

* SOC and both RC branches use their **exact** discrete solutions.
* The temperature sensor lag uses its exact zero-order-hold solution, so ``tau -> 0``
  recovers "no lag" exactly rather than leaving a one-sample delay.
* The thermal nodes use forward Euler at the sampling period, exactly as the observer does.
  With ``core_surface_resistance_k_per_w == 0`` the two nodes collapse into one and the
  update becomes, operation for operation, ``pack.simulate``'s.

The last point costs something: a small core-to-surface resistance is a *stiff* fast mode,
and forward Euler at 1 s will diverge. We raise rather than silently substep, because
silently refining the plant's integrator while leaving the observer's alone would inject a
discretisation mismatch that has nothing to do with the physics under study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astracell.cell.ecm import r0_at_temperature, rc_step, soc_step
from astracell.cell.thermal import heat_generation, reversible_heat, thermal_derivative
from astracell.pack.params import PackParams
from astracell.pack.simulate import SOC_MAX, SOC_MIN, SimulationError, _initial_state
from astracell.plant.mismatch import MismatchModel

FloatArray = NDArray[np.float64]

__all__ = ["PlantResult", "PlantStabilityError", "simulate_plant"]


class PlantStabilityError(SimulationError):
    """The plant's fast thermal mode is faster than the sampling period can resolve."""


@dataclass(frozen=True)
class PlantResult:
    """True (noiseless) plant trajectory. Time-major: ``(n_time, n_cells)``."""

    time_s: FloatArray
    current_a: FloatArray
    soc: FloatArray
    voltage_v: FloatArray
    temp_core_k: FloatArray
    temp_surface_k: FloatArray
    temp_measured_k: FloatArray
    """What a thermocouple would read: the *surface*, through the sensor's own lag."""

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
    def core_surface_gradient_k(self) -> FloatArray:
        """``T_core - T_surface``. Zero for a lumped cell; the observer assumes it is."""
        return self.temp_core_k - self.temp_surface_k


def _fast_thermal_rate(params: PackParams, mismatch: MismatchModel) -> float:
    """Eigenvalue [1/s] of the core<->surface mode. Zero when the cell is lumped."""
    if mismatch.core_surface_resistance_k_per_w == 0.0:
        return 0.0
    k_cs = 1.0 / mismatch.core_surface_resistance_k_per_w
    c_th = np.asarray(params.heat_capacity_j_per_k, dtype=float)
    c_core = mismatch.core_heat_fraction * c_th
    c_surf = c_th - c_core
    return float(np.max(k_cs * (1.0 / c_core + 1.0 / c_surf)))


def _check_stability(params: PackParams, mismatch: MismatchModel, dt_s: float) -> None:
    rate = _fast_thermal_rate(params, mismatch)
    if rate * dt_s >= 2.0:
        tau = 1.0 / rate
        raise PlantStabilityError(
            f"the core<->surface thermal mode has tau = {tau:.3g} s, and forward Euler at "
            f"dt = {dt_s:g} s requires dt < 2*tau = {2 * tau:.3g} s. Raise "
            f"core_surface_resistance_k_per_w (currently "
            f"{mismatch.core_surface_resistance_k_per_w:g} K/W) or shorten dt. We refuse to "
            "substep the plant silently: refining one model and not the other would inject "
            "a discretisation mismatch unrelated to the physics under study."
        )


def simulate_plant(
    params: PackParams,
    current_a: FloatArray,
    dt_s: float,
    mismatch: MismatchModel,
    *,
    soc0: float | FloatArray = 0.75,
    temp0_k: float | FloatArray | None = None,
    check_bounds: bool = True,
) -> PlantResult:
    """Integrate the higher-fidelity plant over the given current profile.

    Given ``mismatch.is_exact``, every recorded array equals ``pack.simulate``'s to the
    last bit, and ``temp_core_k == temp_surface_k == temp_measured_k``.
    """
    n = params.n_cells
    n_time = int(np.asarray(current_a).size)
    if n_time < 2:
        raise ValueError("current profile must have at least two samples")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    _check_stability(params, mismatch, dt_s)

    current = np.asarray(current_a, dtype=float)
    elec, therm = params.electrical, params.thermal
    lap = params.conductance_matrix()
    curve = params.curve

    soc = _initial_state(soc0, n, "soc0")
    soc_ref = soc.copy()  # R0's SOC dependence is centred here, so it vanishes at t=0
    v_rc = np.zeros(n)
    v_rc2 = np.zeros(n)
    temp_core = _initial_state(params.coolant_temp_k if temp0_k is None else temp0_k, n, "temp0_k")
    temp_surf = temp_core.copy()
    temp_meas = temp_core.copy()

    lumped = mismatch.core_surface_resistance_k_per_w == 0.0
    if lumped:
        c_core = c_surf = k_cs = None
    else:
        k_cs = 1.0 / mismatch.core_surface_resistance_k_per_w
        c_core = mismatch.core_heat_fraction * np.asarray(therm.heat_capacity_j_per_k, dtype=float)
        c_surf = np.asarray(therm.heat_capacity_j_per_k, dtype=float) - c_core

    slow = mismatch.second_rc_r_ohm > 0.0
    rc2_decay = np.exp(-dt_s / mismatch.second_rc_tau_s) if slow else 0.0
    lagged = mismatch.temp_sensor_tau_s > 0.0
    sensor_decay = np.exp(-dt_s / mismatch.temp_sensor_tau_s) if lagged else 0.0

    soc_out = np.empty((n_time, n))
    volt_out = np.empty((n_time, n))
    core_out = np.empty((n_time, n))
    surf_out = np.empty((n_time, n))
    meas_out = np.empty((n_time, n))

    for k in range(n_time):
        i_k = float(current[k])

        # R0 depends on the *core* temperature (where the current actually flows) and,
        # unlike the observer's, on state of charge.
        r0 = r0_at_temperature(elec.r0_ohm, temp_core, elec.ea_over_r_k)
        if mismatch.r0_soc_slope != 0.0:
            r0 = r0 * (1.0 + mismatch.r0_soc_slope * (soc_ref - soc))

        volt = curve.ocv(soc, temp_core) - i_k * r0 - v_rc
        if slow:
            volt = volt - v_rc2

        soc_out[k] = soc
        volt_out[k] = volt
        core_out[k] = temp_core
        surf_out[k] = temp_surf
        meas_out[k] = temp_meas

        if k == n_time - 1:
            break

        docv_dtemp = curve.docv_dtemp(soc)
        polarisation = v_rc + v_rc2 if slow else v_rc
        if lumped:
            # Identical arithmetic to pack.simulate, so the reduction is exact.
            heat = heat_generation(i_k, r0, polarisation, temp_core, docv_dtemp)
            dtemp = thermal_derivative(temp_core, heat, lap, therm)
            temp_core = temp_core + dt_s * dtemp
            temp_surf = temp_core  # rebound after the update, so the two stay in step
        else:
            assert c_core is not None and c_surf is not None and k_cs is not None
            heat = i_k * i_k * r0 + i_k * polarisation
            heat = heat + reversible_heat(i_k, temp_core, docv_dtemp)
            conducted = k_cs * (temp_core - temp_surf)
            convected = therm.ha_w_per_k * (temp_surf - therm.coolant_temp_k)
            d_core = (heat - conducted) / c_core
            d_surf = (conducted - lap @ temp_surf - convected) / c_surf
            temp_core = temp_core + dt_s * d_core
            temp_surf = temp_surf + dt_s * d_surf

        soc = soc_step(soc, i_k, dt_s, elec)
        v_rc = rc_step(v_rc, i_k, dt_s, elec)
        if slow:
            v_rc2 = v_rc2 * rc2_decay + mismatch.second_rc_r_ohm * (1.0 - rc2_decay) * i_k

        # Zero-order hold on the *new* surface temperature, so tau -> 0 recovers no lag
        # exactly, rather than leaving a one-sample delay behind.
        if lagged:
            temp_meas = sensor_decay * temp_meas + (1.0 - sensor_decay) * temp_surf
        else:
            temp_meas = temp_surf

    if check_bounds:
        lo, hi = float(soc_out.min()), float(soc_out.max())
        if lo < SOC_MIN or hi > SOC_MAX:
            raise SimulationError(
                f"SOC left [{SOC_MIN}, {SOC_MAX}] (observed [{lo:.4f}, {hi:.4f}]). "
                "Shorten the profile, lower the mean C-rate, or move soc0."
            )

    return PlantResult(
        time_s=np.arange(n_time, dtype=float) * dt_s,
        current_a=current,
        soc=soc_out,
        voltage_v=volt_out,
        temp_core_k=core_out,
        temp_surface_k=surf_out,
        temp_measured_k=meas_out,
    )
