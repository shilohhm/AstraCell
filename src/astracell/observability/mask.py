"""Turning a Cramer-Rao bound into a decision about what the pack map may show.

    SNR_j = fault_magnitude / sqrt(CRLB_jj)

is the number of standard deviations by which the best possible estimator could
separate "this fault, at this magnitude" from "no fault". It is an *upper bound
on the performance of any detector*. If it is below 2, no algorithm you write
will reliably see the fault, and the honest thing to render is grey.

Thresholds are conventions, and are exposed rather than buried:

===============  ==================================================
``>= 5 sigma``   OBSERVABLE. Standard particle-physics-flavoured bar.
``2 - 5 sigma``  WEAK. Detectable in principle, not reliably.
``< 2 sigma``    UNOBSERVABLE. Refuse. Recommend a measurement.
===============  ==================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from astracell.observability.fisher import crlb, fisher_information
from astracell.observability.sensitivity import ParameterSpec, ParamKind, all_specs, sensitivities
from astracell.pack.params import PackParams
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import SensorTopology

FloatArray = NDArray[np.float64]

DEFAULT_WEAK_SIGMA: float = 2.0
DEFAULT_STRONG_SIGMA: float = 5.0


class Observability(IntEnum):
    """What the pack map is allowed to say about a cell."""

    UNOBSERVABLE = 0
    WEAK = 1
    OBSERVABLE = 2

    @property
    def colour(self) -> str:
        return {0: "#9e9e9e", 1: "#f9a825", 2: "#2e7d32"}[int(self)]

    @property
    def label(self) -> str:
        return {0: "unobservable", 1: "weak", 2: "observable"}[int(self)]


def detection_snr(crlb_variance: FloatArray, magnitude: float) -> FloatArray:
    """``magnitude / sqrt(CRLB)``, with ``inf`` variance mapping to zero SNR."""
    if magnitude <= 0.0:
        raise ValueError("fault magnitude must be positive")
    variance = np.asarray(crlb_variance, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = magnitude / np.sqrt(variance)
    return np.where(np.isfinite(snr), snr, 0.0)


def classify(
    snr: FloatArray,
    *,
    weak_sigma: float = DEFAULT_WEAK_SIGMA,
    strong_sigma: float = DEFAULT_STRONG_SIGMA,
) -> NDArray[np.int_]:
    """Map an SNR array to ``Observability`` levels."""
    if not 0.0 < weak_sigma <= strong_sigma:
        raise ValueError("require 0 < weak_sigma <= strong_sigma")
    snr = np.asarray(snr, dtype=float)
    out = np.full(snr.shape, int(Observability.UNOBSERVABLE), dtype=int)
    out[snr >= weak_sigma] = int(Observability.WEAK)
    out[snr >= strong_sigma] = int(Observability.OBSERVABLE)
    return out


@dataclass(frozen=True)
class GreyCellMap:
    """Per-cell identifiability of one fault kind at one magnitude."""

    kind: ParamKind
    magnitude: float
    snr: FloatArray  # (n_cells,)
    level: NDArray[np.int_]  # (n_cells,) Observability
    crlb_std: FloatArray  # (n_cells,) best achievable 1-sigma, as a fraction
    topology: SensorTopology

    def counts(self) -> dict[str, int]:
        return {
            level.label: int(np.count_nonzero(self.level == int(level))) for level in Observability
        }

    def unobservable_cells(self) -> tuple[int, ...]:
        return tuple(int(c) for c in np.flatnonzero(self.level == int(Observability.UNOBSERVABLE)))


def grey_cell_map(
    params: PackParams,
    current_a: FloatArray,
    dt_s: float,
    topology: SensorTopology,
    noise: NoiseModel,
    *,
    kind: ParamKind,
    magnitude: float,
    confounders: tuple[ParamKind, ...] | None = None,
    weak_sigma: float = DEFAULT_WEAK_SIGMA,
    strong_sigma: float = DEFAULT_STRONG_SIGMA,
    soc0: float = 0.75,
) -> GreyCellMap:
    """Per-cell identifiability of ``kind`` at ``magnitude``, over the whole pack.

    The Fisher information is built over **all** cells' parameters in
    ``confounders`` (default: all three kinds), so the resulting CRLB marginalises
    over both within-cell confounding (R0 vs capacity vs hA on the same cell) and
    cross-cell confounding (a hot neighbour looks like a hot cell).

    Cost is ``2 * n_cells * len(confounders)`` simulations. For a 32-cell pack with
    all three kinds that is 192 short simulations -- a few seconds. Restrict
    ``confounders`` to trade honesty for speed, and say so if you do.
    """
    confounders = confounders or tuple(ParamKind)
    if kind not in confounders:
        raise ValueError(f"target kind {kind} must appear in confounders {confounders}")

    specs = all_specs(params.n_cells, confounders)
    sens = sensitivities(params, current_a, dt_s, specs, soc0=soc0)
    fim = fisher_information(sens, topology, noise)
    variance = crlb(fim)

    target = np.array([i for i, s in enumerate(specs) if s.kind is kind], dtype=int)
    order = np.argsort([specs[i].cell for i in target])
    target = target[order]

    var_target = variance[target]
    snr = detection_snr(var_target, magnitude)
    return GreyCellMap(
        kind=kind,
        magnitude=magnitude,
        snr=snr,
        level=classify(snr, weak_sigma=weak_sigma, strong_sigma=strong_sigma),
        crlb_std=np.sqrt(var_target),
        topology=topology,
    )


def snr_for_topology(
    sens: FloatArray,
    topology: SensorTopology,
    noise: NoiseModel,
    specs: tuple[ParameterSpec, ...],
    target: ParameterSpec,
    magnitude: float,
) -> float:
    """SNR of one parameter under a hypothetical sensor topology.

    Reuses a precomputed sensitivity tensor, so sweeping over candidate sensor
    placements does not re-simulate anything.
    """
    try:
        index = specs.index(target)
    except ValueError as exc:  # pragma: no cover - programmer error
        raise ValueError(f"{target.label()} is not among the differentiated parameters") from exc
    fim = fisher_information(sens, topology, noise)
    return float(detection_snr(crlb(fim)[index : index + 1], magnitude)[0])


def recommend_temp_sensor(
    sens: FloatArray,
    topology: SensorTopology,
    noise: NoiseModel,
    specs: tuple[ParameterSpec, ...],
    target: ParameterSpec,
    magnitude: float,
    *,
    candidates: tuple[int, ...] | None = None,
) -> list[tuple[int, float]]:
    """Where should the next thermocouple go?

    Returns ``[(cell, resulting_snr), ...]`` sorted best-first, for every candidate
    cell that does not already carry a thermocouple. This is the "what additional
    measurement would make this identifiable?" half of the thesis, and it is a
    counterfactual over the *same* sensitivity tensor -- no re-simulation.
    """
    pool = candidates if candidates is not None else tuple(range(topology.n_cells))
    scored = [
        (
            cell,
            snr_for_topology(
                sens, topology.with_temp_sensor_at(cell), noise, specs, target, magnitude
            ),
        )
        for cell in pool
        if cell not in topology.temp_cells
    ]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
