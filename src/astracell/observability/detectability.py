"""Detectability as a function of excitation and fault magnitude.

This produces the figure that justifies the whole project:

    "A 20% resistance fault is detectable at 5 sigma above 0.8C of current
     excitation, and is not detectable below 0.3C. Outside the coloured region,
     AstraCell abstains."

Two things to understand about the resulting grid.

**It is linear in the magnitude axis by construction, and that is the point.**
Under a local linearisation the CRLB does not depend on the fault magnitude, so
``SNR = magnitude / sqrt(CRLB(excitation))`` and every column is a straight line
through the origin. The grid is therefore a statement about the *Fisher
information*, not about a particular fault size -- one simulation sweep gives you
every magnitude. Large faults will violate the linearisation; the bound then
becomes conservative in ways this code does not model.

**The confounders are per-cell, not pack-wide.** We differentiate the target
cell's R0, capacity, and hA, so the CRLB marginalises over the R0/capacity
collinearity -- the interesting confounding. Cross-cell confounding is ignored
here for cost. ``mask.grey_cell_map`` includes it; the heatmap does not. Use the
heatmap to reason about excitation, and the grey-cell map to reason about
placement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astracell.duty.profiles import DutyProfile, pulse_train
from astracell.observability.fisher import condition_number, crlb, fisher_information
from astracell.observability.mask import classify
from astracell.observability.sensitivity import ParamKind, local_specs, sensitivities
from astracell.pack.params import PackParams
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import SensorTopology

FloatArray = NDArray[np.float64]

ProfileFactory = Callable[[float], DutyProfile]


@dataclass(frozen=True)
class HeatmapResult:
    """Detectability of one fault kind on one cell, over (excitation, magnitude)."""

    cell: int
    kind: ParamKind
    excitation_c_rate: FloatArray  # (n_exc,)
    magnitude: FloatArray  # (n_mag,)
    snr: FloatArray  # (n_mag, n_exc)
    crlb_std: FloatArray  # (n_exc,) best achievable 1-sigma, as a fraction
    r0_capacity_condition: FloatArray  # (n_exc,) collinearity of the R0/Q pair
    current_std_a: FloatArray  # (n_exc,) the excitation, in amps

    def level(self, **kwargs: float) -> NDArray[np.int_]:
        return classify(self.snr, **kwargs)

    def min_detectable_magnitude(self, sigma: float = 5.0) -> FloatArray:
        """The smallest fault visible at ``sigma``, per excitation level."""
        return sigma * self.crlb_std


def default_profile_factory(
    duration_s: float = 1200.0,
    dt_s: float = 1.0,
    mean_c_rate: float = 0.2,
    period_s: float = 60.0,
    duty: float = 0.25,
) -> ProfileFactory:
    """Pulse trains of varying amplitude, at fixed mean current.

    Fixing the mean decouples the axes: moving along the excitation axis changes
    only the current *variance*, not the SOC traversed. Without that decoupling
    the heatmap would confound "more current excitation" with "more SOC sweep",
    and the resistance and capacity stories would smear into each other.
    """

    def factory(pulse_c_rate: float) -> DutyProfile:
        return pulse_train(
            duration_s,
            dt_s,
            mean_c_rate=mean_c_rate,
            pulse_c_rate=pulse_c_rate,
            period_s=period_s,
            duty=duty,
        )

    return factory


def detectability_heatmap(
    params: PackParams,
    topology: SensorTopology,
    noise: NoiseModel,
    *,
    cell: int,
    kind: ParamKind = ParamKind.R0,
    excitation_c_rate: FloatArray | None = None,
    magnitude: FloatArray | None = None,
    profile_factory: ProfileFactory | None = None,
    soc0: float = 0.75,
) -> HeatmapResult:
    """Sweep excitation, compute the CRLB, and scale to every fault magnitude.

    Cost is ``6 * n_excitation`` simulations (three per-cell parameters, central
    differences), independent of how many magnitudes you ask for.
    """
    excitation = np.asarray(
        excitation_c_rate if excitation_c_rate is not None else np.linspace(0.05, 2.5, 12),
        dtype=float,
    )
    # Log-spaced: the Cramer-Rao floor for R0 sits near 0.1% while a cooling fault
    # needs tens of percent, so the interesting range spans three decades.
    magnitudes = np.asarray(
        magnitude if magnitude is not None else np.geomspace(1e-3, 0.5, 24), dtype=float
    )
    factory = profile_factory or default_profile_factory()

    specs = local_specs(cell)
    target_index = next(i for i, s in enumerate(specs) if s.kind is kind)
    r0_index = next(i for i, s in enumerate(specs) if s.kind is ParamKind.R0)
    cap_index = next(i for i, s in enumerate(specs) if s.kind is ParamKind.CAPACITY)

    std = np.empty(excitation.size)
    cond = np.empty(excitation.size)
    current_std = np.empty(excitation.size)

    for k, c_rate in enumerate(excitation):
        profile = factory(float(c_rate))
        sens = sensitivities(params, profile.current_a, profile.dt_s, specs, soc0=soc0)
        fim = fisher_information(sens, topology, noise)
        std[k] = np.sqrt(crlb(fim)[target_index])
        pair = np.ix_([r0_index, cap_index], [r0_index, cap_index])
        cond[k] = condition_number(fim[pair])
        current_std[k] = profile.current_std_a

    # SNR = magnitude / sqrt(CRLB), and CRLB does not depend on magnitude under the
    # local linearisation -- so one simulation sweep yields every magnitude at once.
    # That identity is asserted in tests/test_observability.py, not here.
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = magnitudes[:, None] / std[None, :]
    snr = np.where(np.isfinite(snr), snr, 0.0)

    return HeatmapResult(
        cell=cell,
        kind=kind,
        excitation_c_rate=excitation,
        magnitude=magnitudes,
        snr=snr,
        crlb_std=std,
        r0_capacity_condition=cond,
        current_std_a=current_std,
    )
