"""The refusal.

This module is the thesis. Given a fault hypothesis and the data actually in hand,
AstraCell either answers or declines -- and when it declines, it says which
measurement would change its mind.

There is no classifier here, and deliberately so. A classifier answers "which
fault?". This answers the logically prior question: "is that question answerable?"
Bolting a classifier on before this is settled produces a system that is confident
exactly where it should be silent.

Three gates, in this order:

1. **Isolation.** Is the target parameter separable from the ones it could be
   confused with? Measured by the variance inflation factor, not the condition
   number -- see ``fisher.variance_inflation`` for why that distinction matters.
2. **Detection.** Given that it is separable, is a fault of this magnitude large
   enough to rise above the Cramer-Rao floor?
3. **Credibility.** Given that it rises above the noise, does it rise above the
   error our own model is making? See ``observability.bias``.

Isolation is checked first because a pair of parameters can be jointly
well-determined -- high SNR on their sum -- while being individually
unidentifiable. Reporting "resistance fault, 8 sigma" when the data cannot
separate resistance from capacity would be a confident lie.

Credibility is checked last and is the only gate that **cannot be argued with by
collecting more of the same data**. Gates 1 and 2 describe variance: repeat the
experiment a hundred times and they both yield. Gate 3 describes bias, and repeating
the experiment leaves it bit-for-bit unchanged -- so a system reporting only gates 1
and 2 grows unboundedly *more confident* and *no less wrong* the more data you feed
it. That is the failure this module exists to prevent, and ``REFUSE_MODEL_BIAS`` is
what it looks like when prevented.

Changing the experiment is a different matter, and not the escape it sounds like.
A harder pulse train does move the bias -- off one parameter and onto another. See
``observability.bias``. Variance can be reduced; bias can only be moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from astracell.observability.bias import parameter_bias
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
    REFUSE_MODEL_BIAS = "refuse_model_bias"


_REFUSALS = frozenset(
    {
        VerdictKind.REFUSE_UNOBSERVABLE,
        VerdictKind.REFUSE_CONFOUNDED,
        VerdictKind.REFUSE_MODEL_BIAS,
    }
)


@dataclass(frozen=True)
class Verdict:
    """What AstraCell is prepared to say about one fault hypothesis.

    ``snr`` is the *variance-limited* SNR, ``magnitude / sqrt(CRLB)`` -- what the repository
    reported before it knew any better. ``bias`` is the structural error of the observer's
    own model, in the same relative units, and is ``None`` when no mismatch analysis was
    supplied. Where both are present, ``snr_total`` is the number that decides.
    """

    kind: VerdictKind
    target: ParameterSpec
    magnitude: float
    snr: float
    crlb_std: float
    vif: float
    reason: str
    recommendation: str | None = None
    bias: float | None = None

    @property
    def will_diagnose(self) -> bool:
        return self.kind is VerdictKind.DIAGNOSE

    @property
    def refused(self) -> bool:
        return self.kind in _REFUSALS

    @property
    def snr_total(self) -> float:
        """``m / sqrt(CRLB + b^2)``. Falls back to ``snr`` when the bias is unknown."""
        if self.bias is None:
            return self.snr
        total = np.hypot(self.crlb_std, self.bias)
        return float(self.magnitude / total) if np.isfinite(total) and total > 0 else 0.0

    @property
    def bias_ceiling(self) -> float:
        """``m / |b|``: what ``snr_total`` converges to as the variance is driven to zero.

        Infinite when the model is exact. Otherwise this is the most **this experiment**
        could ever claim, however many times you repeat it. A different excitation has a
        different bias and therefore a different ceiling -- higher for some parameters,
        lower for others. See ``observability.bias``.
        """
        if self.bias is None:
            return float("inf")
        return float(self.magnitude / abs(self.bias)) if self.bias != 0.0 else float("inf")

    def render(self) -> str:
        floor = "n/a" if not np.isfinite(self.crlb_std) else f"+/- {100 * self.crlb_std:.2f}%"
        lines = [
            f"  hypothesis   : {self.target.kind.value} on cell {self.target.cell}"
            f" at {100 * self.magnitude:.0f}%",
            f"  verdict      : {self.kind.value.upper()}",
            f"  detection SNR: {self.snr:.2f} sigma   <- variance only",
            f"  CRLB (1s)    : {floor}   <- best possible, over every unbiased estimator",
            f"  VIF          : {self.vif:.2f}",
        ]
        if self.bias is not None:
            ceiling = self.bias_ceiling
            ceiling_txt = "unbounded" if not np.isfinite(ceiling) else f"{ceiling:.2f} sigma"
            lines += [
                f"  model bias   : {100 * self.bias:+.2f}%   <- structural; does not average away",
                f"  total SNR    : {self.snr_total:.2f} sigma   <- variance AND bias",
                f"  bias ceiling : {ceiling_txt}   <- this experiment, repeated forever",
            ]
        lines.append(f"  reason       : {self.reason}")
        if self.recommendation:
            lines.append(f"  recommend    : {self.recommendation}")
        return "\n".join(lines)


def _model_bias_recommendation(target: ParameterSpec, bias: float, ceiling: float) -> str:
    """What to do about a bias refusal, which is never 'collect more data'."""
    ceiling_txt = "0" if ceiling == 0.0 else f"{ceiling:.2f}"
    return (
        f"do not measure harder -- measure differently. The observer's model is wrong by "
        f"{100 * abs(bias):.2f}% on {target.label()}, and that error is a bias, not a variance: "
        f"repeating this experiment drives the Cramer-Rao floor to zero and leaves it exactly "
        f"where it is, so even with unlimited data this hypothesis tops out at {ceiling_txt} "
        f"sigma. A different excitation would move the bias between parameters rather than "
        f"remove it. The reliable fix is the model: a second RC branch, SOC-dependent R0, a "
        f"core/surface thermal split -- or fit over a window where the missing dynamics are "
        f"inactive"
    )


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
    bias: float | None = None,
) -> Verdict:
    """Refuse first; diagnose only if nothing objects.

    ``bias`` is optional. Omitted, this behaves exactly as it did before the mismatch layer
    existed -- which is to say, optimistically. Supplied, the detection thresholds are
    applied to ``snr_total`` rather than ``snr``, and a hypothesis that clears the noise but
    not the model's own error is refused as ``REFUSE_MODEL_BIAS``.
    """
    common = dict(recommendation=recommendation, bias=bias)

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
            **common,  # type: ignore[arg-type]
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
            **common,  # type: ignore[arg-type]
        )

    # The variance gates are satisfied: the fault is visible and separable. Now ask whether
    # it is larger than the error we are making by fitting the wrong model.
    probe = Verdict(VerdictKind.DIAGNOSE, target, magnitude, snr, crlb_std, vif, "", bias=bias)
    effective, ceiling = probe.snr_total, probe.bias_ceiling

    if bias is not None and effective < weak_sigma:
        return Verdict(
            VerdictKind.REFUSE_MODEL_BIAS,
            target,
            magnitude,
            snr,
            crlb_std,
            vif,
            reason=(
                f"the noise says yes and the model says no: {target.label()} clears the "
                f"Cramer-Rao floor at {snr:.2f} sigma, but the observer's structural bias is "
                f"{100 * abs(bias):.2f}% against a {100 * magnitude:.0f}% fault, leaving "
                f"{effective:.2f} < {weak_sigma:.0f} sigma of credible evidence. This is a "
                "false positive waiting to happen, not a detection"
            ),
            recommendation=_model_bias_recommendation(target, bias, ceiling),
            bias=bias,
        )

    if effective < strong_sigma:
        # Bound as a float, not a bool: this is the branch where the variance gate would
        # have said DIAGNOSE and only the bias pulled it down. `None` means "not squeezed".
        squeezed = bias if (bias is not None and snr >= strong_sigma) else None
        advice: str | None
        if squeezed is not None:
            reason = (
                f"variance alone would call this at {snr:.2f} sigma, but structural bias of "
                f"{100 * abs(squeezed):.2f}% pulls the credible evidence down to "
                f"{effective:.2f} sigma; report it, do not act on it"
            )
            advice = _model_bias_recommendation(target, squeezed, ceiling)
        else:
            reason = (
                f"marginal: SNR {snr:.2f} sigma lies between {weak_sigma:.0f} and "
                f"{strong_sigma:.0f}; a detector would work, unreliably"
            )
            advice = recommendation
        return Verdict(
            VerdictKind.WEAK_EVIDENCE,
            target,
            magnitude,
            snr,
            crlb_std,
            vif,
            reason=reason,
            recommendation=advice,
            bias=bias,
        )

    credible = (
        ""
        if bias is None
        else (
            f", and survives a structural bias of {100 * abs(bias):.2f}% "
            f"(total {effective:.2f} sigma, ceiling {ceiling:.2f})"
        )
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
            f"<= {max_vif:.0f}, so the parameter is both visible and separable{credible}"
        ),
        bias=bias,
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


def assess_under_mismatch(
    sens: FloatArray,
    topology: SensorTopology,
    noise: NoiseModel,
    specs: tuple[ParameterSpec, ...],
    target: ParameterSpec,
    magnitude: float,
    residual: FloatArray,
    **decide_kwargs: float | str | None,
) -> Verdict:
    """``assess``, plus the third gate.

    ``residual`` is the structural residual of the plant against the observer's model, from
    ``bias.structural_residual``. It is projected onto the same whitened sensitivity basis
    that produced the FIM, so the bias and the CRLB are commensurable by construction.

    The bias here is the **linearised** one. Quote ``bias.pseudo_true_bias`` where the exact
    figure matters; see that module for how far apart they get.
    """
    try:
        index = specs.index(target)
    except ValueError as exc:
        raise ValueError(f"{target.label()} is not among the differentiated parameters") from exc

    fim = fisher_information(sens, topology, noise, specs=specs)
    variance = crlb(fim)[index]
    snr = float(detection_snr(np.array([variance]), magnitude)[0])
    vif = float(variance_inflation(fim)[index])
    bias = float(parameter_bias(sens, residual, topology, noise, specs)[index])
    return decide(
        target,
        magnitude,
        snr,
        float(np.sqrt(variance)),
        vif,
        bias=bias,
        **decide_kwargs,  # type: ignore[arg-type]
    )


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
