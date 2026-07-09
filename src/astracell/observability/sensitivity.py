"""Sensitivity of the pack's outputs to per-cell parameter perturbations.

We differentiate with respect to **relative** perturbations, so a sensitivity has
units of "volts per unit fractional change in R0". That makes the Cramer-Rao
bound directly comparable to a fault magnitude expressed as a percentage, which
is how fault magnitudes are actually specified.

Critically, we compute sensitivities of **every cell's voltage and every cell's
temperature**, whether or not that cell is instrumented. The sensor topology is
applied afterwards, as a row selection. Two consequences:

* Counterfactual sensor placement ("what if I put a thermocouple here?") costs a
  matrix slice, not a re-simulation.
* You can see, quantitatively, how much information the pack is *throwing away*
  by not instrumenting a cell.

Method: central finite differences. Central rather than forward because this
derivative is the scientific object of the whole package -- an O(eps) truncation
error here propagates straight into a claimed detection threshold. The default
``eps = 1e-3`` sits comfortably between truncation error (O(eps^2) = 1e-6
relative) and cancellation error (O(machine_eps / eps) = 2e-13 relative).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from astracell.pack.params import PackParams
from astracell.pack.simulate import simulate

FloatArray = NDArray[np.float64]

# Output channel indices in the sensitivity tensor's third axis.
VOLTAGE_CHANNEL: int = 0
TEMPERATURE_CHANNEL: int = 1
N_OUTPUT_CHANNELS: int = 2

DEFAULT_EPS: float = 1e-3


class ParamKind(StrEnum):
    """Parameters the Fisher information is computed over.

    The first three are per-cell physical parameters, perturbed *relatively*.
    ``CURRENT_BIAS`` is a pack-global **nuisance** parameter, perturbed in amps.
    """

    R0 = "r0_ohm"
    CAPACITY = "capacity_ah"
    HA = "ha_w_per_k"
    CURRENT_BIAS = "current_bias_a"


#: The per-cell physical parameters. Faults live here.
CELL_PARAM_KINDS: tuple[ParamKind, ...] = (ParamKind.R0, ParamKind.CAPACITY, ParamKind.HA)

#: Pack-global nuisance parameters. Nobody diagnoses these; we marginalise over them.
GLOBAL_PARAM_KINDS: tuple[ParamKind, ...] = (ParamKind.CURRENT_BIAS,)

#: Finite-difference step for the current bias, in amps. Small next to the ~2 A
#: accuracy spec of a real shunt, large next to float64 cancellation at ~50 A.
DEFAULT_EPS_CURRENT_A: float = 1e-2


@dataclass(frozen=True)
class ParameterSpec:
    """A single scalar parameter.

    ``cell`` is the cell index for per-cell parameters and ``None`` for pack-global
    nuisance parameters.
    """

    cell: int | None
    kind: ParamKind

    def __post_init__(self) -> None:
        if self.is_global and self.cell is not None:
            raise ValueError(f"{self.kind} is pack-global; cell must be None")
        if not self.is_global and self.cell is None:
            raise ValueError(f"{self.kind} is per-cell; cell must be an index")
        if self.cell is not None and self.cell < 0:
            raise ValueError("cell index must be non-negative")

    @property
    def is_global(self) -> bool:
        return self.kind in GLOBAL_PARAM_KINDS

    def label(self) -> str:
        if self.kind is ParamKind.CURRENT_BIAS:
            return "I_bias"
        short = {ParamKind.R0: "R0", ParamKind.CAPACITY: "Q", ParamKind.HA: "hA"}[self.kind]
        return f"{short}[{self.cell}]"

    def unit(self) -> str:
        """The unit of a one-unit perturbation of this parameter."""
        return "A" if self.is_global else "relative"


#: The pack current the BMS *recorded* differs from the current that actually flowed
#: by an unknown constant offset. Include this in the parameter set to marginalise
#: over it rather than pretending the current is known exactly.
CURRENT_BIAS_SPEC = ParameterSpec(None, ParamKind.CURRENT_BIAS)


def local_specs(cell: int, kinds: tuple[ParamKind, ...] | None = None) -> tuple[ParameterSpec, ...]:
    """The parameters of a single cell.

    Using only these gives a CRLB that marginalises over *that cell's* other
    parameters -- enough to expose the R0/capacity collinearity, which is the
    interesting confounding -- while ignoring cross-cell confounding. Cheap, and
    the right tool for a per-cell heatmap. Use ``all_specs`` when cross-cell
    confounding matters.
    """
    kinds = kinds or CELL_PARAM_KINDS
    return tuple(ParameterSpec(cell, k) for k in kinds)


def all_specs(
    n_cells: int, kinds: tuple[ParamKind, ...] | None = None
) -> tuple[ParameterSpec, ...]:
    """Every cell's every parameter. Cost is ``2 * len(specs)`` simulations."""
    kinds = kinds or CELL_PARAM_KINDS
    return tuple(ParameterSpec(c, k) for c in range(n_cells) for k in kinds)


def with_current_bias(specs: tuple[ParameterSpec, ...]) -> tuple[ParameterSpec, ...]:
    """Append the pack-global current-bias nuisance parameter, if not already present."""
    if CURRENT_BIAS_SPEC in specs:
        return specs
    return (*specs, CURRENT_BIAS_SPEC)


def perturb(params: PackParams, spec: ParameterSpec, relative: float) -> PackParams:
    """Return a new PackParams with ``spec``'s value scaled by ``1 + relative``.

    Per-cell parameters only. A global parameter does not live in ``PackParams``;
    see ``_perturbed`` for the dispatch used inside ``sensitivities``.
    """
    if spec.is_global:
        raise ValueError(f"{spec.label()} is a global parameter, not a PackParams field")
    assert spec.cell is not None  # narrowed by is_global
    return params.scale_cell(spec.kind.value, spec.cell, 1.0 + relative)


def _step_size(spec: ParameterSpec, eps_relative: float, eps_current_a: float) -> float:
    """Finite-difference step, in that parameter's own units."""
    return eps_current_a if spec.kind is ParamKind.CURRENT_BIAS else eps_relative


def apply_perturbations(
    params: PackParams,
    current_a: FloatArray,
    specs: tuple[ParameterSpec, ...],
    deltas: FloatArray,
) -> tuple[PackParams, FloatArray]:
    """Apply a whole deviation vector at once, in each parameter's own units.

    The inverse operation to reading sensitivity columns off ``sensitivities``: physical
    parameters scale the pack (relative), the current bias shifts the recorded current
    (amps). Used by the estimator to evaluate the observer at ``theta_nominal + delta``.
    """
    deltas = np.asarray(deltas, dtype=float)
    if deltas.shape != (len(specs),):
        raise ValueError(f"deltas has shape {deltas.shape}, expected ({len(specs)},)")
    perturbed_params, perturbed_current = params, np.asarray(current_a, dtype=float)
    for spec, delta in zip(specs, deltas, strict=True):
        perturbed_params, perturbed_current = _perturbed(
            perturbed_params, perturbed_current, spec, float(delta)
        )
    return perturbed_params, perturbed_current


def _perturbed(
    params: PackParams, current_a: FloatArray, spec: ParameterSpec, delta: float
) -> tuple[PackParams, FloatArray]:
    """Apply ``delta`` to one parameter, returning new (params, current profile).

    The current bias is a statement about the *data*, not the pack: the current that
    actually flowed is the recorded current plus an unknown constant. So it perturbs
    the profile, and leaves the pack alone.
    """
    if spec.kind is ParamKind.CURRENT_BIAS:
        return params, current_a + delta
    return perturb(params, spec, delta), current_a


def _stack_outputs(voltage: FloatArray, temp: FloatArray) -> FloatArray:
    """(n_time, n_cells) x2 -> (n_time, n_cells, 2)."""
    return np.stack([voltage, temp], axis=2)


def sensitivities(
    params: PackParams,
    current_a: FloatArray,
    dt_s: float,
    specs: tuple[ParameterSpec, ...],
    *,
    eps: float = DEFAULT_EPS,
    eps_current_a: float = DEFAULT_EPS_CURRENT_A,
    soc0: float | FloatArray = 0.75,
    temp0_k: float | FloatArray | None = None,
) -> FloatArray:
    """Central-difference sensitivity tensor.

    Returns
    -------
    S : ndarray, shape ``(n_time, n_cells, 2, n_params)``
        ``S[t, c, 0, p]`` is d(voltage of cell c at time t) / d(parameter p), in volts.
        ``S[t, c, 1, p]`` is the same for temperature, in kelvin. Per-cell parameters
        are differentiated with respect to a *relative* change; the current bias with
        respect to *amps*. Read units off ``ParameterSpec.unit()``.

    Notes
    -----
    Bounds checking is disabled on the perturbed runs. The nominal run is checked;
    if it stays inside the valid SOC window, a 0.1% parameter perturbation cannot
    meaningfully leave it, and we would rather not have a spurious raise at a
    boundary silently truncate a sensitivity sweep.
    """
    if not specs:
        raise ValueError("no parameters to differentiate")
    if eps <= 0.0 or eps_current_a <= 0.0:
        raise ValueError("finite-difference steps must be positive")

    current = np.asarray(current_a, dtype=float)

    # Nominal run: only to validate the operating point and fix output shapes.
    nominal = simulate(params, current, dt_s, soc0=soc0, temp0_k=temp0_k, check_bounds=True)
    n_time, n_cells = nominal.n_time, nominal.n_cells

    sens = np.empty((n_time, n_cells, N_OUTPUT_CHANNELS, len(specs)), dtype=float)

    for p, spec in enumerate(specs):
        if spec.cell is not None and spec.cell >= n_cells:
            raise IndexError(f"spec targets cell {spec.cell}, pack has {n_cells}")
        step = _step_size(spec, eps, eps_current_a)

        params_plus, current_plus = _perturbed(params, current, spec, +step)
        params_minus, current_minus = _perturbed(params, current, spec, -step)

        plus = simulate(
            params_plus, current_plus, dt_s, soc0=soc0, temp0_k=temp0_k, check_bounds=False
        )
        minus = simulate(
            params_minus, current_minus, dt_s, soc0=soc0, temp0_k=temp0_k, check_bounds=False
        )
        out_plus = _stack_outputs(plus.voltage_v, plus.temp_k)
        out_minus = _stack_outputs(minus.voltage_v, minus.temp_k)
        sens[:, :, :, p] = (out_plus - out_minus) / (2.0 * step)

    return sens
