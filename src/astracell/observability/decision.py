"""The refusal.

This module is the thesis. Given a fault hypothesis and the data actually in hand,
AstraCell either answers or declines -- and when it declines, it says which
measurement would change its mind.

There is no classifier here, and deliberately so. A classifier answers "which
fault?". This answers the logically prior question: "is that question answerable?"
Bolting a classifier on before this is settled produces a system that is confident
exactly where it should be silent.

Two gates, in this order:

1. **Isolation.** Is the target parameter separable from the ones it could be
   confused with? Measured by the variance inflation factor, not the condition
   number -- see ``fisher.variance_inflation`` for why that distinction matters.
2. **Detection.** Given that it is separable, is a fault of this magnitude large
   enough to rise above the Cramer-Rao floor?

Isolation is checked first because a pair of parameters can be jointly
well-determined -- high SNR on their sum -- while being individually
unidentifiable. Reporting "resistance fault, 8 sigma" when the data cannot
separate resistance from capacity would be a confident lie.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from astracell.observability.fisher import crlb, fisher_information, variance_inflation
from astracell.observability.mask import (
    DEFAULT_STRONG_SIGMA,
    DEFAULT_WEAK_SIGMA,
    Observability,
    detection_snr,
    recommend_temp_sensor,
)
from astracell.observability.sensitivity import ParameterSpec
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import SensorTopology

FloatArray = NDArray[np.float64]

# Conventional regression-diagnostics threshold for serious multicollinearity.
# Inherited, not invented.
DEFAULT_MAX_VIF: float = 10.0


class VerdictKind(StrEnum):
    DIAGNOSE = "diagnose"
    WEAK_EVIDENCE = "weak_evidence"
    REFUSE_UNOBSERVABLE = "refuse_unobservable"
    REFUSE_CONFOUNDED = "refuse_confounded"


@dataclass(frozen=True)
class Verdict:
    """What AstraCell is prepared to say about one fault hypothesis."""

    kind: VerdictKind
    target: ParameterSpec
    magnitude: float
    snr: float
    crlb_std: float
    vif: float
    reason: str
    recommendation: str | None = None

    @property
    def will_diagnose(self) -> bool:
        return self.kind is VerdictKind.DIAGNOSE

    @property
    def refused(self) -> bool:
        return self.kind in (VerdictKind.REFUSE_UNOBSERVABLE, VerdictKind.REFUSE_CONFOUNDED)

    def render(self) -> str:
        floor = "n/a" if not np.isfinite(self.crlb_std) else f"+/- {100 * self.crlb_std:.2f}%"
        lines = [
            f"  hypothesis   : {self.target.kind.value} on cell {self.target.cell}"
            f" at {100 * self.magnitude:.0f}%",
            f"  verdict      : {self.kind.value.upper()}",
            f"  detection SNR: {self.snr:.2f} sigma",
            f"  CRLB (1s)    : {floor}   <- best possible, over every unbiased estimator",
            f"  VIF          : {self.vif:.2f}",
            f"  reason       : {self.reason}",
        ]
        if self.recommendation:
            lines.append(f"  recommend    : {self.recommendation}")
        return "\n".join(lines)


def decide(
    target: ParameterSpec,
    magnitude: float,
    snr: float,
    crlb_std: float,
    vif: float,
    *,
    weak_sigma: float = DEFAULT_WEAK_SIGMA,
    strong_sigma: float = DEFAULT_STRONG_SIGMA,
    max_vif: float = DEFAULT_MAX_VIF,
    recommendation: str | None = None,
) -> Verdict:
    """Refuse first; diagnose only if nothing objects."""
    if not np.isfinite(vif) or vif > max_vif:
        return Verdict(
            VerdictKind.REFUSE_CONFOUNDED,
            target,
            magnitude,
            snr,
            crlb_std,
            vif,
            reason=(
                f"{target.label()} is not separable from the other parameters in this window "
                f"(VIF = {vif:.3g} > {max_vif:.0f}); their joint effect may be plainly visible "
                "but they cannot be told apart"
            ),
            recommendation=recommendation,
        )

    if snr < weak_sigma:
        floor = float("inf") if not np.isfinite(crlb_std) else 100.0 * strong_sigma * crlb_std
        floor_txt = "unbounded" if not np.isfinite(floor) else f"{floor:.0f}%"
        return Verdict(
            VerdictKind.REFUSE_UNOBSERVABLE,
            target,
            magnitude,
            snr,
            crlb_std,
            vif,
            reason=(
                f"no unbiased estimator can resolve a {100 * magnitude:.0f}% fault here "
                f"(SNR {snr:.2f} < {weak_sigma:.0f} sigma); the smallest fault this data could "
                f"see at {strong_sigma:.0f} sigma is {floor_txt}"
            ),
            recommendation=recommendation,
        )

    if snr < strong_sigma:
        return Verdict(
            VerdictKind.WEAK_EVIDENCE,
            target,
            magnitude,
            snr,
            crlb_std,
            vif,
            reason=(
                f"marginal: SNR {snr:.2f} sigma lies between {weak_sigma:.0f} and "
                f"{strong_sigma:.0f}; a detector would work, unreliably"
            ),
            recommendation=recommendation,
        )

    return Verdict(
        VerdictKind.DIAGNOSE,
        target,
        magnitude,
        snr,
        crlb_std,
        vif,
        reason=(
            f"identifiable: SNR {snr:.2f} >= {strong_sigma:.0f} sigma and VIF {vif:.2f} "
            f"<= {max_vif:.0f}, so the parameter is both visible and separable"
        ),
    )


def assess(
    sens: FloatArray,
    topology: SensorTopology,
    noise: NoiseModel,
    specs: tuple[ParameterSpec, ...],
    target: ParameterSpec,
    magnitude: float,
    **decide_kwargs: float | str | None,
) -> Verdict:
    """Compute the FIM for this topology and hand the numbers to ``decide``."""
    try:
        index = specs.index(target)
    except ValueError as exc:
        raise ValueError(f"{target.label()} is not among the differentiated parameters") from exc

    # Passing specs applies Gaussian priors to nuisance parameters (the current bias).
    fim = fisher_information(sens, topology, noise, specs=specs)
    variance = crlb(fim)[index]
    snr = float(detection_snr(np.array([variance]), magnitude)[0])
    vif = float(variance_inflation(fim)[index])
    return decide(target, magnitude, snr, float(np.sqrt(variance)), vif, **decide_kwargs)  # type: ignore[arg-type]


def sensor_recommendation(
    sens: FloatArray,
    topology: SensorTopology,
    noise: NoiseModel,
    specs: tuple[ParameterSpec, ...],
    target: ParameterSpec,
    magnitude: float,
    *,
    strong_sigma: float = DEFAULT_STRONG_SIGMA,
) -> str | None:
    """Phrase the counterfactual: which thermocouple would rescue this hypothesis?

    This is the "what additional measurement would make it identifiable?" half of
    the thesis. It is a counterfactual over the *same* sensitivity tensor, because
    ``sensitivity.sensitivities`` differentiates every cell's temperature whether
    or not that cell is instrumented. No re-simulation.
    """
    ranked = recommend_temp_sensor(sens, topology, noise, specs, target, magnitude)
    if not ranked:
        return None
    cell, snr = ranked[0]
    if snr < DEFAULT_WEAK_SIGMA:
        return (
            f"no single additional thermocouple makes this identifiable "
            f"(best candidate is cell {cell}, reaching only {snr:.2f} sigma)"
        )
    level = Observability.OBSERVABLE if snr >= strong_sigma else Observability.WEAK
    return f"add a temperature sensor on cell {cell} -> SNR {snr:.2f} sigma ({level.label})"
