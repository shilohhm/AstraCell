"""An external, higher-fidelity plant: a single PyBaMM cell.

This is the first time AstraCell is tested against a plant it did not write. v0.1's
``plant.mismatch`` was a hand-built intermediate model -- honest, but its mismatch was four
terms *we* chose, so "the observer is simpler than the plant" was true by construction and by an
amount we set. PyBaMM's SPMe/DFN carry electrolyte and particle diffusion that a first-order ECM
cannot express at all, and we did not design the gap. Whether AstraCell's calibrated abstention
survives *that* is the external-validity question §13 of ``LIMITATIONS.md`` left open.

Two deliberate limits keep this small and honest:

* **One cell, voltage only.** No pack, no thermal model (the default is isothermal, so a
  temperature channel would be a flat line). The observer is a one-cell ECM. Pack-scale
  electrochemistry is explicitly out of scope for v0.3.
* **Optional dependency.** The core repository stays numpy-only; the entire point of AstraCell
  is that identifiability, mismatch, and calibration do not need an electrochemical solver.
  ``PYBAMM_AVAILABLE`` gates every entry point, and the tests and notebook skip cleanly without
  it. ``import pybamm`` happens lazily inside the functions, never at module load.

What this still does **not** establish is in ``docs/EXTERNAL_PLANT.md``: PyBaMM is a model, not a
cell, so this tests external *model* mismatch, not physical truth.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: True when the optional PyBaMM dependency is importable. Computed without importing it, so this
#: module is always safe to import.
PYBAMM_AVAILABLE: bool = importlib.util.find_spec("pybamm") is not None

DEFAULT_PARAMETER_SET: str = "Chen2020"
DEFAULT_MODEL: str = "SPMe"
_MODELS = ("SPM", "SPMe", "DFN")


class PyBaMMUnavailableError(RuntimeError):
    """PyBaMM was asked for but is not installed. Install with ``pip install -e '.[pybamm]'``."""


def require_pybamm() -> None:
    if not PYBAMM_AVAILABLE:
        raise PyBaMMUnavailableError(
            "this experiment needs the optional PyBaMM dependency; "
            "install it with `pip install -e '.[pybamm]'`"
        )


@dataclass(frozen=True)
class PyBaMMPlant:
    """One PyBaMM run reduced to what AstraCell's ECM observer needs to be fitted against.

    ``voltage_v`` is the terminal voltage sampled on exactly the supplied ``time_s`` grid, so it
    drops straight into the ``plant_output`` slot of a calibration ``ScenarioContext``. ``soc0``
    and ``capacity_ah`` are the truth the ECM's capacity estimate is measured against.
    """

    model: str
    parameter_set: str
    capacity_ah: float
    soc0: float
    time_s: FloatArray
    current_a: FloatArray
    voltage_v: FloatArray

    @property
    def n_time(self) -> int:
        return self.time_s.size

    @property
    def dt_s(self) -> float:
        return float(self.time_s[1] - self.time_s[0])


def _model_instance(model: str):  # type: ignore[no-untyped-def]
    import pybamm

    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}")
    classes = {
        "SPM": pybamm.lithium_ion.SPM,
        "SPMe": pybamm.lithium_ion.SPMe,
        "DFN": pybamm.lithium_ion.DFN,
    }
    return classes[model]()


def simulate_pybamm_cell(
    current_a: FloatArray,
    dt_s: float,
    *,
    model: str = DEFAULT_MODEL,
    parameter_set: str = DEFAULT_PARAMETER_SET,
    soc0: float = 0.9,
) -> PyBaMMPlant:
    """Run a single PyBaMM cell on the given current profile; return voltage on that grid.

    ``current_a`` uses AstraCell's sign convention (positive = discharge), which is PyBaMM's too.
    The solution is sampled at the profile's own timestamps, so the returned voltage aligns with
    what the ECM observer produces for the same current -- the residual between them is the whole
    object of the experiment.
    """
    require_pybamm()
    import pybamm

    current_a = np.asarray(current_a, dtype=float)
    time_s = np.arange(current_a.size, dtype=float) * dt_s

    parameters = pybamm.ParameterValues(parameter_set)
    capacity_ah = float(parameters["Nominal cell capacity [A.h]"])
    parameters["Current function [A]"] = pybamm.Interpolant(
        time_s, current_a, pybamm.t, name="current"
    )

    simulation = pybamm.Simulation(_model_instance(model), parameter_values=parameters)
    solution = simulation.solve(t_eval=time_s, initial_soc=soc0)
    voltage = np.asarray(solution["Terminal voltage [V]"](time_s), dtype=float)
    if not np.all(np.isfinite(voltage)):
        raise RuntimeError(
            f"PyBaMM {model} produced non-finite voltage; the duty cycle may be too harsh"
        )

    return PyBaMMPlant(
        model=model,
        parameter_set=parameter_set,
        capacity_ah=capacity_ah,
        soc0=soc0,
        time_s=time_s,
        current_a=current_a,
        voltage_v=voltage,
    )


def pybamm_pseudo_ocv(
    *,
    model: str = DEFAULT_MODEL,
    parameter_set: str = DEFAULT_PARAMETER_SET,
    c_rate: float = 0.05,
    soc_lo: float = 0.05,
    soc_hi: float = 0.98,
    n_points: int = 200,
) -> tuple[FloatArray, FloatArray]:
    """A slow-discharge pseudo-OCV: ``(soc, ocv)`` for building the observer's OCV curve.

    Discharging at ``c_rate`` (C/20 by default) keeps the overpotential small, so terminal
    voltage tracks equilibrium OCV. Giving the observer *this* curve is the control that isolates
    the interesting mismatch: with the static voltage-SOC relationship shared, any residual under
    load is the **dynamics** the ECM cannot express, not a curve-shape disagreement. The two
    plants agree on where the cell sits and disagree only on how it gets there.
    """
    require_pybamm()
    import pybamm

    parameters = pybamm.ParameterValues(parameter_set)
    capacity_ah = float(parameters["Nominal cell capacity [A.h]"])
    current = c_rate * capacity_ah
    parameters["Current function [A]"] = current

    # Long enough to traverse the full window at this slow rate, with margin.
    horizon_s = (soc_hi - soc_lo) / c_rate * 3600.0 * 1.2
    t_eval = np.linspace(0.0, horizon_s, 4000)
    simulation = pybamm.Simulation(_model_instance(model), parameter_values=parameters)
    solution = simulation.solve(t_eval=t_eval, initial_soc=soc_hi)

    times = np.asarray(solution["Time [s]"].entries, dtype=float)
    voltage = np.asarray(solution["Terminal voltage [V]"].entries, dtype=float)
    soc = soc_hi - current * times / 3600.0 / capacity_ah

    keep = (soc >= soc_lo) & (soc <= soc_hi) & np.isfinite(voltage)
    soc, voltage = soc[keep], voltage[keep]
    order = np.argsort(soc)
    soc, voltage = soc[order], voltage[order]
    # Resample onto a uniform SOC grid so the table is well-conditioned for np.gradient.
    grid = np.linspace(soc.min(), soc.max(), n_points)
    return grid, np.interp(grid, soc, voltage)
