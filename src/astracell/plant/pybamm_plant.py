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

v0.4 adds the other half of the experiment
------------------------------------------

v0.3 ran a *healthy* PyBaMM cell and watched the ECM invent a fault. That is a negative control,
and a negative control alone cannot tell a system that refuses everything from a system that
refuses correctly. ``PyBaMMFault`` injects a known degradation so the complementary question can
be asked: when the cell really *is* damaged, is the damage recovered? See
``calibration.positive_control`` and ``docs/POSITIVE_CONTROL.md``.

Two faults, chosen to sit on opposite sides of the observer's model span:

* ``contact_resistance`` -- a series resistance the ECM's ``R0`` expresses **exactly**. The
  differential voltage is ``-dR * I`` to machine precision (``3.6e-16 V`` measured), which is
  precisely the ECM's ``R0`` sensitivity. Recoverable, and its recoverability is the point of a
  positive control, not a rig.
* ``slow_cathode`` -- reduced positive-particle diffusivity, i.e. cathode particle cracking or
  surface reconstruction. Lithium inventory and active material are untouched, so the cell's
  ``(R0, capacity)`` are unchanged and the honest deviation is **zero**. It lands in the blind
  spot of a one-RC ECM, which reads it as a 119% capacity loss at 158 sigma. The confounder.

Cleanly injecting *capacity* loss was considered and rejected. ``Nominal cell capacity [A.h]``
only normalises the C-rate and the initial-SOC lookup; it changes no electrode capacity. Real
capacity fade is loss of lithium inventory or of active material, and both move the stoichiometry
window, hence the OCV-SOC map -- which would break the shared-pseudo-OCV control that isolates
*dynamic* mismatch in the first place. ``LIMITATIONS.md`` §15 records the omission.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: PyBaMM parameter names touched by fault injection. Named once so a version bump that renames
#: one fails loudly at the lookup rather than silently simulating a healthy cell.
CONTACT_RESISTANCE_KEY = "Contact resistance [Ohm]"
CATHODE_DIFFUSIVITY_KEY = "Positive particle diffusivity [m2.s-1]"

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
class PyBaMMFault:
    """A known degradation applied to the external plant's parameters, and nothing else.

    Both knobs default to "no fault", so ``PyBaMMFault('healthy')`` is the v0.3 cell exactly and
    the healthy trace is not a different code path from the faulted one -- it is the same call
    with the same solver at ``dR = 0``, which is what makes the paired difference meaningful.

    ``delta_r0_ohm`` is an **absolute** series resistance. The observer's ``R0`` is a nominal it
    chose, not a quantity the electrochemical cell possesses, so the fault cannot be stated as a
    fraction of it without smuggling the observer into the plant. Convert at the boundary:
    ``fault_magnitude = delta_r0_ohm / observer_r0_ohm``.
    """

    name: str
    delta_r0_ohm: float = 0.0
    cathode_diffusivity_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.delta_r0_ohm < 0.0:
            raise ValueError("delta_r0_ohm must be non-negative; a series resistance cannot fall")
        if not 0.0 < self.cathode_diffusivity_scale <= 1.0:
            raise ValueError("cathode_diffusivity_scale must lie in (0, 1]")

    @property
    def is_healthy(self) -> bool:
        return self.delta_r0_ohm == 0.0 and self.cathode_diffusivity_scale == 1.0

    @property
    def needs_contact_resistance(self) -> bool:
        """Whether the PyBaMM model must be built with the contact-resistance submodel.

        Enabling it at ``dR = 0`` reproduces the plain model bit-for-bit (measured: ``max|dV| =
        0.0``), so gating on the magnitude rather than on the fault kind costs nothing and keeps
        the healthy solve on the default code path.
        """
        return self.delta_r0_ohm > 0.0

    def describe(self) -> str:
        if self.is_healthy:
            return "none (healthy cell)"
        parts = []
        if self.delta_r0_ohm:
            parts.append(f"contact resistance +{1e3 * self.delta_r0_ohm:.3f} mOhm")
        if self.cathode_diffusivity_scale != 1.0:
            parts.append(f"cathode diffusivity x{self.cathode_diffusivity_scale:.2f}")
        return ", ".join(parts)


#: The v0.3 cell: no degradation at all.
HEALTHY = PyBaMMFault("healthy")


def contact_resistance(delta_r0_ohm: float) -> PyBaMMFault:
    """A series resistance rise: corroded tab weld, degraded interconnect.

    Exactly expressible by the observer's ``R0``, and that is why it is the primary positive
    control. If the pipeline cannot recover a fault its own model can represent, no result about
    a fault the model *cannot* represent means anything.
    """
    return PyBaMMFault(
        f"contact_resistance/{1e3 * delta_r0_ohm:.3g}mOhm", delta_r0_ohm=delta_r0_ohm
    )


def slow_cathode(scale: float) -> PyBaMMFault:
    """Reduced positive-particle diffusivity: cathode cracking, surface reconstruction.

    A real physical change whose true ``(R0, capacity)`` deviation is **zero** -- no lithium and
    no active material is lost, and at C/20 the deliverable capacity is unchanged. It perturbs
    only transport, in the one direction a first-order ECM is blind to, so the observer reads it
    as an enormous capacity fault. The confounder.
    """
    return PyBaMMFault(f"slow_cathode/x{scale:.2g}", cathode_diffusivity_scale=scale)


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
    fault: PyBaMMFault = HEALTHY

    @property
    def n_time(self) -> int:
        return self.time_s.size

    @property
    def dt_s(self) -> float:
        return float(self.time_s[1] - self.time_s[0])


def _model_instance(model: str, fault: PyBaMMFault = HEALTHY):  # type: ignore[no-untyped-def]
    import pybamm

    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}")
    classes = {
        "SPM": pybamm.lithium_ion.SPM,
        "SPMe": pybamm.lithium_ion.SPMe,
        "DFN": pybamm.lithium_ion.DFN,
    }
    options = {"contact resistance": "true"} if fault.needs_contact_resistance else {}
    return classes[model](options=options)


def _apply_fault(parameters, fault: PyBaMMFault) -> None:  # type: ignore[no-untyped-def]
    """Mutate a fresh ``ParameterValues`` in place. Missing keys raise, never pass silently."""
    if fault.needs_contact_resistance:
        if CONTACT_RESISTANCE_KEY not in parameters:
            raise KeyError(f"this PyBaMM/parameter set has no {CONTACT_RESISTANCE_KEY!r}")
        parameters[CONTACT_RESISTANCE_KEY] = fault.delta_r0_ohm
    if fault.cathode_diffusivity_scale != 1.0:
        if CATHODE_DIFFUSIVITY_KEY not in parameters:
            raise KeyError(f"this PyBaMM/parameter set has no {CATHODE_DIFFUSIVITY_KEY!r}")
        parameters[CATHODE_DIFFUSIVITY_KEY] = (
            parameters[CATHODE_DIFFUSIVITY_KEY] * fault.cathode_diffusivity_scale
        )


def simulate_pybamm_cell(
    current_a: FloatArray,
    dt_s: float,
    *,
    model: str = DEFAULT_MODEL,
    parameter_set: str = DEFAULT_PARAMETER_SET,
    soc0: float = 0.9,
    fault: PyBaMMFault = HEALTHY,
) -> PyBaMMPlant:
    """Run a single PyBaMM cell on the given current profile; return voltage on that grid.

    ``current_a`` uses AstraCell's sign convention (positive = discharge), which is PyBaMM's too.
    The solution is sampled at the profile's own timestamps, so the returned voltage aligns with
    what the ECM observer produces for the same current -- the residual between them is the whole
    object of the experiment.

    ``fault`` degrades the cell before it is solved. The healthy and faulted traces then differ
    only in the injected parameter, on the same solver, the same grid, and the same excitation,
    which is what lets their difference be attributed to the degradation and nothing else.
    """
    require_pybamm()
    import pybamm

    current_a = np.asarray(current_a, dtype=float)
    time_s = np.arange(current_a.size, dtype=float) * dt_s

    parameters = pybamm.ParameterValues(parameter_set)
    capacity_ah = float(parameters["Nominal cell capacity [A.h]"])
    _apply_fault(parameters, fault)
    parameters["Current function [A]"] = pybamm.Interpolant(
        time_s, current_a, pybamm.t, name="current"
    )

    simulation = pybamm.Simulation(_model_instance(model, fault), parameter_values=parameters)
    solution = simulation.solve(t_eval=time_s, initial_soc=soc0)
    voltage = np.asarray(solution["Terminal voltage [V]"](time_s), dtype=float)
    if not np.all(np.isfinite(voltage)):
        raise RuntimeError(
            f"PyBaMM {model} produced non-finite voltage under fault {fault.name!r}; "
            "the duty cycle may be too harsh"
        )

    return PyBaMMPlant(
        model=model,
        parameter_set=parameter_set,
        capacity_ah=capacity_ah,
        soc0=soc0,
        time_s=time_s,
        current_a=current_a,
        voltage_v=voltage,
        fault=fault,
    )


def pybamm_pseudo_ocv(
    *,
    model: str = DEFAULT_MODEL,
    parameter_set: str = DEFAULT_PARAMETER_SET,
    c_rate: float = 0.05,
    soc_lo: float = 0.05,
    soc_hi: float = 0.98,
    n_points: int = 200,
    fault: PyBaMMFault = HEALTHY,
) -> tuple[FloatArray, FloatArray]:
    """A slow-discharge pseudo-OCV: ``(soc, ocv)`` for building the observer's OCV curve.

    Discharging at ``c_rate`` (C/20 by default) keeps the overpotential small, so terminal
    voltage tracks equilibrium OCV. Giving the observer *this* curve is the control that isolates
    the interesting mismatch: with the static voltage-SOC relationship shared, any residual under
    load is the **dynamics** the ECM cannot express, not a curve-shape disagreement. The two
    plants agree on where the cell sits and disagree only on how it gets there.

    ``fault`` exists so v0.4's claim that ``slow_cathode`` leaves the cell's *capacity* untouched
    can be checked rather than asserted: at C/20 a transport parameter has almost nowhere to act,
    so a faulted pseudo-OCV that still lies on the healthy one is the evidence that the true
    capacity deviation is zero and any capacity estimate under load is bias. Note this does *not*
    hold for ``contact_resistance``, which shifts even a C/20 curve by ``I * dR``.
    """
    require_pybamm()
    import pybamm

    parameters = pybamm.ParameterValues(parameter_set)
    capacity_ah = float(parameters["Nominal cell capacity [A.h]"])
    _apply_fault(parameters, fault)
    current = c_rate * capacity_ah
    parameters["Current function [A]"] = current

    # Long enough to traverse the full window at this slow rate, with margin.
    horizon_s = (soc_hi - soc_lo) / c_rate * 3600.0 * 1.2
    t_eval = np.linspace(0.0, horizon_s, 4000)
    simulation = pybamm.Simulation(_model_instance(model, fault), parameter_values=parameters)
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
