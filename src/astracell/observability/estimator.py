"""An estimator, at last -- so the Cramer-Rao bound can be checked instead of asserted.

Every number the repository published before v0.2 was a property of the *design*: the CRLB
bounds the variance of an unbiased estimator, the VIF measures separability, the bias measures
structural error. None of them ran an estimator on data. ``LIMITATIONS.md`` §10 said so
plainly: "No CRLB has been shown to be attainable... nobody has run an estimator on noisy data
and compared its scatter to the bound."

This module runs one. It fits the *simpler ECM observer* to a noisy measurement vector and
returns the fault estimate, the CRLB-derived interval, and the VIF -- everything ``decide``
needs to turn one realisation into one verdict. Feed it many noise draws (``calibration``) and
the verdicts and intervals acquire a frequency interpretation: coverage, false-positive rate,
abstention quality.

Two estimators, deliberately:

``fit_linear``
    One weighted-least-squares projection at the linearisation point, ``delta_hat =
    FIM^-1 S^T Sigma^-1 (y - f(theta_nominal))``. For a linear-Gaussian model this *is* the
    maximum-likelihood estimator and its covariance is *exactly* ``FIM^-1`` -- the reported
    CRLB. So its matched-model coverage is nominal by construction, which makes it the honest
    workhorse for the coverage curves (the interesting results -- mismatch undercoverage,
    the bias gate -- do not depend on it being nominal, they depend on ``b``) and a razor for
    catching noise/whitening/interval bugs. It is general: pass a current-bias nuisance and it
    marginalises correctly, because the projection is pure linear algebra on the precomputed
    ``S`` and ``FIM``.

``fit_gauss_newton``
    Damped fixed-information Gauss-Newton against the true *nonlinear* observer. Its fixed
    point is where ``S^T Sigma^-1 (y - f(theta_nominal + delta)) = 0`` with ``f`` evaluated
    nonlinearly, so its scatter matches the CRLB only up to the model's curvature. That makes
    "scatter tracks the CRLB" a real measurement rather than a tautology -- it is how we show
    the linearisation is good enough at these fault sizes. Physical parameters only, like
    ``pseudo_true_bias``; the nuisance is the CRLB's job to marginalise, not the fit's.

The reported ``crlb_std`` and ``vif`` come from the FIM at the healthy nominal, exactly as the
rest of the repository reports them -- so an interval built here is the interval the pack map
would have drawn, now with an estimate inside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astracell.observability.bias import residual_score, solve_bias
from astracell.observability.decision import Verdict, decide
from astracell.observability.fisher import crlb, fisher_information, variance_inflation
from astracell.observability.sensitivity import (
    N_OUTPUT_CHANNELS,
    TEMPERATURE_CHANNEL,
    VOLTAGE_CHANNEL,
    ParameterSpec,
    apply_perturbations,
    sensitivities,
)
from astracell.pack.params import PackParams
from astracell.pack.simulate import simulate
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import SensorTopology

FloatArray = NDArray[np.float64]

DEFAULT_MAX_STEP: float = 0.2
DEFAULT_MAX_ITER: int = 30
DEFAULT_STEP_TOL: float = 1e-9

#: A fault estimate is essentially never exactly zero, but ``decide`` and ``bias_aware_snr``
#: require a positive magnitude. Floor the estimated magnitude here rather than special-case
#: every caller.
_MIN_MAGNITUDE: float = 1e-12


def output_tensor(voltage: FloatArray, temp: FloatArray) -> FloatArray:
    """Stack a voltage and a temperature field into the ``(t, cell, channel)`` layout.

    The temperature field is whatever plays the role of the measured temperature: the
    observer's lumped node, or the plant's laggy surface reading. The caller decides which;
    this only lays it out so ``residual_score`` can project it.
    """
    tensor = np.empty((voltage.shape[0], voltage.shape[1], N_OUTPUT_CHANNELS))
    tensor[:, :, VOLTAGE_CHANNEL] = voltage
    tensor[:, :, TEMPERATURE_CHANNEL] = temp
    return tensor


@dataclass(frozen=True)
class FitContext:
    """Everything about the observer and the design that no noise draw can change.

    Built once per scenario; reused across every trial. ``sens``, ``fim``, ``crlb_std`` and
    ``vif`` are the healthy-nominal design objects; ``observer_nominal`` is ``f(theta_nominal)``,
    the point the measurement residual is taken against.
    """

    params: PackParams
    specs: tuple[ParameterSpec, ...]
    current_a: FloatArray
    dt_s: float
    topology: SensorTopology
    noise: NoiseModel
    soc0: float | FloatArray
    temp0_k: float | FloatArray | None
    sens: FloatArray
    fim: FloatArray
    crlb_std: FloatArray
    vif: FloatArray
    observer_nominal: FloatArray

    @property
    def n_params(self) -> int:
        return len(self.specs)


def build_context(
    params: PackParams,
    specs: tuple[ParameterSpec, ...],
    current_a: FloatArray,
    dt_s: float,
    topology: SensorTopology,
    noise: NoiseModel,
    *,
    soc0: float | FloatArray = 0.75,
    temp0_k: float | FloatArray | None = None,
) -> FitContext:
    """Precompute the noise-independent design objects for one scenario."""
    sens = sensitivities(params, current_a, dt_s, specs, soc0=soc0, temp0_k=temp0_k)
    fim = fisher_information(sens, topology, noise, specs=specs)
    variance = crlb(fim)
    nominal = simulate(params, current_a, dt_s, soc0=soc0, temp0_k=temp0_k)
    return FitContext(
        params=params,
        specs=specs,
        current_a=np.asarray(current_a, dtype=float),
        dt_s=dt_s,
        topology=topology,
        noise=noise,
        soc0=soc0,
        temp0_k=temp0_k,
        sens=sens,
        fim=fim,
        crlb_std=np.sqrt(variance),
        vif=variance_inflation(fim),
        observer_nominal=output_tensor(nominal.voltage_v, nominal.temp_k),
    )


def fit_linear(ctx: FitContext, measurement: FloatArray) -> FloatArray:
    """One weighted-least-squares step from the linearisation point.

    ``delta_hat = FIM^-1 S^T Sigma^-1 (y - f(theta_nominal))``. Exact for a linear-Gaussian
    model, general in the nuisance parameters, and free per trial once the context is built.
    """
    score = residual_score(ctx.sens, measurement - ctx.observer_nominal, ctx.topology, ctx.noise)
    return solve_bias(ctx.fim, score)


@dataclass(frozen=True)
class GaussNewtonResult:
    delta: FloatArray
    converged: bool
    iterations: int


def fit_gauss_newton(
    ctx: FitContext,
    measurement: FloatArray,
    *,
    max_step: float = DEFAULT_MAX_STEP,
    max_iter: int = DEFAULT_MAX_ITER,
    step_tol: float = DEFAULT_STEP_TOL,
) -> GaussNewtonResult:
    """Damped fixed-information Gauss-Newton fit of the nonlinear observer to ``measurement``.

    Physical parameters only. Relative steps compose multiplicatively -- ``(1+d)(1+s)-1`` --
    because ``apply_perturbations`` scales rather than shifts them, matching ``pseudo_true_bias``.
    Unlike that function this does not raise on non-convergence: a Monte Carlo must not die on
    one unlucky draw, so it returns the last iterate with ``converged=False`` and lets the
    caller account for it.
    """
    if any(spec.is_global for spec in ctx.specs):
        raise ValueError("fit_gauss_newton fits per-cell parameters only; use fit_linear")

    delta = np.zeros(ctx.n_params)
    for iteration in range(1, max_iter + 1):
        fitted_params, fitted_current = apply_perturbations(
            ctx.params, ctx.current_a, ctx.specs, delta
        )
        observer = simulate(
            fitted_params,
            fitted_current,
            ctx.dt_s,
            soc0=ctx.soc0,
            temp0_k=ctx.temp0_k,
            check_bounds=False,
        )
        residual = measurement - output_tensor(observer.voltage_v, observer.temp_k)
        step = solve_bias(ctx.fim, residual_score(ctx.sens, residual, ctx.topology, ctx.noise))
        if not np.all(np.isfinite(step)):
            return GaussNewtonResult(delta, converged=False, iterations=iteration)

        largest = float(np.abs(step).max())
        if largest < step_tol:
            return GaussNewtonResult(delta, converged=True, iterations=iteration)
        step = step * min(1.0, max_step / largest)
        delta = (1.0 + delta) * (1.0 + step) - 1.0

    return GaussNewtonResult(delta, converged=False, iterations=max_iter)


def verdict_from_estimate(
    ctx: FitContext,
    target_index: int,
    delta: FloatArray,
    *,
    bias: float | None = None,
    **decide_kwargs: float | str | None,
) -> Verdict:
    """Turn one realised estimate into one verdict, with the estimate as the magnitude.

    The detection statistic is ``|delta_hat| / sqrt(CRLB)`` -- the *estimated* fault against
    the noise floor -- so a large noise excursion can trip DIAGNOSE and a small one REFUSE.
    That randomness is exactly what calibration measures. ``bias`` (the structural bias of the
    target) engages the credibility gate; pass ``None`` to reproduce the variance-only verdict.
    """
    target = ctx.specs[target_index]
    magnitude = max(abs(float(delta[target_index])), _MIN_MAGNITUDE)
    sigma = float(ctx.crlb_std[target_index])
    snr = magnitude / sigma if np.isfinite(sigma) and sigma > 0.0 else 0.0
    vif = float(ctx.vif[target_index])
    return decide(target, magnitude, snr, sigma, vif, bias=bias, **decide_kwargs)  # type: ignore[arg-type]
