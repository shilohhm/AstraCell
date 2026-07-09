"""Structural bias: the error the Cramer-Rao bound cannot see.

The CRLB bounds the *variance* of an unbiased estimator, given that the model is right. The
model is not right. When a first-order ECM is fitted to a plant that has SOC-dependent
resistance, a slow diffusion branch, a core-surface temperature gradient and a laggy
thermocouple, the fit does not converge on the true parameter ``theta*``. It converges on
the **pseudo-true** parameter

    theta_0 = argmin_theta || g(theta*, u) - f(theta, u) ||^2_{Sigma^-1}

where ``g`` is the plant and ``f`` the observer's model. Linearising ``f`` about ``theta*``,
with ``r = g(theta*) - f(theta*)`` the **structural residual**:

    argmin_delta || W (r - S delta) ||^2   =>   b = FIM^-1 S^T Sigma^-1 r

This is the omitted-variable-bias formula, and it needs nothing this repository does not
already have: the sensitivity tensor ``S``, the whitening ``W = Sigma^-1/2``, and the FIM.

Read it geometrically. The residual splits into a part inside the span of the sensitivity
columns and a part orthogonal to it. The orthogonal part is *harmless* -- it inflates the
fit residual and nothing else. The part inside the span is **indistinguishable from a real
parameter change**, and the estimator dutifully attributes it to one. That is the bias.

Why this changes the verdict
----------------------------

Total squared error of the estimate is ``b^2 + Var``, and ``Var`` is floored by the CRLB.
So the honest detection statistic for a fault of size ``m`` is

    SNR_total = m / sqrt(CRLB + b^2)

and its behaviour is the whole point. **Repeat the same experiment** a hundred times, or
improve every sensor's ``sigma`` tenfold, and ``CRLB`` collapses while ``b`` does not move by
one bit -- both invariances are exact, and both are tested. So

    SNR_total  ->  m / |b|            as the variance is driven to zero

a hard **ceiling** (``bias_ceiling``) *for that experiment*. A system reporting only the CRLB
therefore grows unboundedly confident and no less wrong the more data you feed it. This
module is what stops that.

Be careful with the ceiling, because it is easy to overclaim
------------------------------------------------------------

The ceiling is a property of an **experiment**, not of a pack. Change the experiment --
lengthen the window, harden the pulse train, move a thermocouple -- and ``b`` changes too.
It does not thereby go away. Measured on the demo pack (``examples/04_model_mismatch.py``),
raising the pulse amplitude from 0.25C to 2.5C moves ``R0``'s bias from ``-10.1%`` through
zero to ``+1.1%``, lifting its ceiling from 2.0 to 32 sigma -- while capacity's bias grows
from ``-2.2%`` to ``-19.3%`` and its ceiling falls from 2.3 to 0.26 sigma.

**Excitation does not remove structural error. It decides which parameter absorbs it.**

Which is the real asymmetry between the two gates in ``decision``: variance can be reduced,
bias can only be moved. There is no duty cycle for which every ``b`` vanishes at once,
because the model is wrong and no amount of stirring makes it right.

Two estimators of the bias, and when to use which
-------------------------------------------------

``parameter_bias`` is the formula above: **one** Gauss-Newton step from ``theta*``. It costs
one plant simulation and one projection, it is exact to first order in the mismatch, and it
is what the cheap sweeps use.

``pseudo_true_bias`` iterates that step to convergence, so it returns ``theta_0 - theta*``
itself. It costs a dozen simulations. Use it for anything you intend to quote.

The gap between them is not academic. At the mismatch magnitudes in ``REALISTIC_MISMATCH``,
the linearisation **overstates** the capacity bias by ~29% and **understates** the ``hA``
bias by ~45%: it errs conservatively on one parameter and optimistically on another, which
is precisely why the headline numbers use the iterated fit. Their agreement as the mismatch
shrinks (relative error 45% -> 0.4% over a 300x range) is the evidence that either means
anything at all, and it is asserted in
``tests/test_mismatch.py::test_the_linearised_bias_is_first_order_accurate``.

Caveats, since the point of the module is to stop overclaiming:

* ``b`` is computed for *our* guess of the plant. A real pack's mismatch is unknown, so
  ``b`` is an estimate of the *scale* of an error we cannot average away -- not a correction
  we may subtract. Subtracting it would just assume the plant we guessed. This is why the
  decision layer *widens* the uncertainty rather than shifting the estimate.
* ``b`` depends on **what else you let float**. Free the current-sensor bias as a nuisance
  parameter and it soaks up whatever part of the residual looks like a current offset: on
  the demo pack, capacity's bias falls from ``-18.5%`` to ``+0.3%`` -- and the fit hands back
  ``+1.11 A`` of "shunt calibration" that is nothing of the kind. Expanding the model with a
  nuisance regressor that happens to span the residual is the textbook cure for
  omitted-variable bias, and it is real: the capacity estimate really does become almost
  unbiased. The price is that the nuisance estimate becomes meaningless, and the cure only
  works for the residual directions the nuisance happens to span.
* Because the plant has no parameter called "R0", ``theta*`` is a convention. See
  ``plant.mismatch``.
* The residual is evaluated on the *nominal healthy* pack. A pack that already carries a
  fault would project a slightly different residual. Second order; not modelled.
* The mismatch is pack-global, so its residual is common-mode. A **cell-specific** structural
  error -- one cell whose diffusion branch has degraded -- would not be absorbed by any
  pack-global nuisance, and is not modelled here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astracell.observability.fisher import (
    DEFAULT_LEAK_TOL,
    DEFAULT_RCOND,
    channel_slices,
    fisher_information,
    whiten_ar1,
)
from astracell.observability.sensitivity import (
    N_OUTPUT_CHANNELS,
    TEMPERATURE_CHANNEL,
    VOLTAGE_CHANNEL,
    ParameterSpec,
    perturb,
    sensitivities,
)
from astracell.pack.params import PackParams
from astracell.pack.simulate import simulate
from astracell.plant.mismatch import MismatchModel
from astracell.plant.simulate import simulate_plant
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import SensorTopology

FloatArray = NDArray[np.float64]

DEFAULT_MAX_STEP: float = 0.2
DEFAULT_MAX_ITER: int = 40
DEFAULT_STEP_TOL: float = 1e-10

__all__ = [
    "BiasConvergenceError",
    "bias_aware_snr",
    "bias_ceiling",
    "parameter_bias",
    "pseudo_true_bias",
    "residual_score",
    "solve_bias",
    "structural_residual",
]


class BiasConvergenceError(RuntimeError):
    """The Gauss-Newton fit of the observer to the plant did not settle."""


def structural_residual(
    params: PackParams,
    mismatch: MismatchModel,
    current_a: FloatArray,
    dt_s: float,
    *,
    soc0: float | FloatArray = 0.75,
    temp0_k: float | FloatArray | None = None,
) -> FloatArray:
    """``r = g(theta*) - f(theta*)``, shape ``(n_time, n_cells, N_OUTPUT_CHANNELS)``.

    Both simulations run at the **same** parameters. Any difference is therefore purely
    structural: it is what the observer's model cannot express, not a parameter it happens
    to have wrong. Noiseless by construction -- bias is not a noise phenomenon.

    The temperature channel compares what a thermocouple *reads* on the plant (the surface,
    through the sensor's lag) with what the observer's single lumped node predicts.
    """
    observer = simulate(params, current_a, dt_s, soc0=soc0, temp0_k=temp0_k)
    plant = simulate_plant(params, current_a, dt_s, mismatch, soc0=soc0, temp0_k=temp0_k)

    residual = np.empty((observer.n_time, observer.n_cells, N_OUTPUT_CHANNELS))
    residual[:, :, VOLTAGE_CHANNEL] = plant.voltage_v - observer.voltage_v
    residual[:, :, TEMPERATURE_CHANNEL] = plant.temp_measured_k - observer.temp_k
    return residual


def residual_score(
    sens: FloatArray,
    residual: FloatArray,
    topology: SensorTopology,
    noise: NoiseModel,
) -> FloatArray:
    """``S^T Sigma^-1 r``, shape ``(n_params,)``.

    The residual projected onto each parameter's sensitivity direction, in the whitened
    metric. Only *measured* channels contribute: a residual on a cell with no thermocouple
    cannot bias anything, because nobody looked.
    """
    if residual.shape[:2] != sens.shape[:2] or residual.shape[2] != sens.shape[2]:
        raise ValueError(
            f"residual shape {residual.shape} does not match sensitivity tensor {sens.shape}"
        )

    score = np.zeros(sens.shape[3])
    for index, channel, sigma, rho in channel_slices(topology, noise):
        whitened_sens = whiten_ar1(sens[:, index, channel, :], rho) / sigma
        whitened_residual = whiten_ar1(residual[:, index, channel], rho) / sigma
        score += np.einsum("tcp,tc->p", whitened_sens, whitened_residual)
    return score


def solve_bias(
    fim: FloatArray,
    score: FloatArray,
    *,
    rcond: float = DEFAULT_RCOND,
    leak_tol: float = DEFAULT_LEAK_TOL,
) -> FloatArray:
    """``FIM^-1 @ score``, with the same rank discipline as ``crlb``.

    We do not use ``pinv``. A parameter the data cannot identify has no meaningful bias
    either -- the minimum-norm solution would hand back a small, confident, invented number.
    Such parameters get ``inf``, exactly as ``crlb`` gives them ``inf`` variance, and both
    routes lead the decision layer to the same refusal.
    """
    fim = 0.5 * (np.asarray(fim, dtype=float) + np.asarray(fim, dtype=float).T)
    n = fim.shape[0]
    if score.shape != (n,):
        raise ValueError(f"score has shape {score.shape}, expected ({n},)")

    eigvals, eigvecs = np.linalg.eigh(fim)
    lam_max = float(eigvals.max()) if n else 0.0
    if not np.isfinite(lam_max) or lam_max <= 0.0:
        return np.full(n, np.inf)

    identified = eigvals > rcond * lam_max
    if not identified.any():
        return np.full(n, np.inf)

    kept = eigvecs[:, identified]
    bias = kept @ ((kept.T @ score) / eigvals[identified])
    leakage = (eigvecs**2)[:, ~identified].sum(axis=1)
    return np.where(leakage > leak_tol, np.inf, bias)


def parameter_bias(
    sens: FloatArray,
    residual: FloatArray,
    topology: SensorTopology,
    noise: NoiseModel,
    specs: tuple[ParameterSpec, ...] | None = None,
    *,
    rcond: float = DEFAULT_RCOND,
    leak_tol: float = DEFAULT_LEAK_TOL,
) -> FloatArray:
    """Signed structural bias per parameter, in the same units as ``crlb_std``.

    Pass the same ``specs`` you pass to ``crlb`` so the nuisance prior is applied to both;
    the estimator is then MAP rather than least squares, and its bias is shrunk consistently.
    Omit ``specs`` to get the pure least-squares bias, which is invariant to the overall
    noise scale -- a fact worth having, and tested.
    """
    fim = fisher_information(sens, topology, noise, specs=specs)
    score = residual_score(sens, residual, topology, noise)
    return solve_bias(fim, score, rcond=rcond, leak_tol=leak_tol)


def pseudo_true_bias(
    params: PackParams,
    mismatch: MismatchModel,
    current_a: FloatArray,
    dt_s: float,
    topology: SensorTopology,
    noise: NoiseModel,
    specs: tuple[ParameterSpec, ...],
    *,
    soc0: float | FloatArray = 0.75,
    temp0_k: float | FloatArray | None = None,
    max_step: float = DEFAULT_MAX_STEP,
    max_iter: int = DEFAULT_MAX_ITER,
    step_tol: float = DEFAULT_STEP_TOL,
) -> FloatArray:
    """``theta_0 - theta*``: where an actual weighted least-squares fit comes to rest.

    Damped Gauss-Newton against the *noiseless* plant output, so the answer is the exact
    structural bias rather than its first-order approximation. This is the number to quote.

    Relative perturbations compose multiplicatively -- ``(1+theta)(1+step) - 1``, not
    ``theta + step`` -- because ``perturb`` scales a parameter rather than shifting it.
    Steps are capped at ``max_step`` so a nearly-unidentifiable direction (a cooling
    coefficient on a cell with no thermocouple) cannot fling a resistance negative on the
    first iteration. Ill-conditioning is a property of the problem, not licence to diverge.

    Only per-cell parameters may be fitted: ``perturb`` refuses pack-global nuisances, and
    marginalising over the current bias is the CRLB's job, not the fit's.
    """
    if any(spec.is_global for spec in specs):
        raise ValueError("pseudo_true_bias fits per-cell parameters only; drop the nuisances")

    plant = simulate_plant(params, current_a, dt_s, mismatch, soc0=soc0, temp0_k=temp0_k)
    theta = np.zeros(len(specs))

    for _ in range(max_iter):
        fitted = params
        for spec, value in zip(specs, theta, strict=True):
            fitted = perturb(fitted, spec, float(value))

        observer = simulate(fitted, current_a, dt_s, soc0=soc0, temp0_k=temp0_k)
        residual = np.empty((observer.n_time, observer.n_cells, N_OUTPUT_CHANNELS))
        residual[:, :, VOLTAGE_CHANNEL] = plant.voltage_v - observer.voltage_v
        residual[:, :, TEMPERATURE_CHANNEL] = plant.temp_measured_k - observer.temp_k

        sens = sensitivities(fitted, current_a, dt_s, specs, soc0=soc0, temp0_k=temp0_k)
        fim = fisher_information(sens, topology, noise)
        step = solve_bias(fim, residual_score(sens, residual, topology, noise))
        if not np.all(np.isfinite(step)):
            raise BiasConvergenceError(
                "a fitted parameter is unidentifiable, so its pseudo-true value is undefined; "
                "use parameter_bias, which reports inf for exactly these directions"
            )

        largest = float(np.abs(step).max())
        if largest < step_tol:
            return theta
        step = step * min(1.0, max_step / largest)
        theta = (1.0 + theta) * (1.0 + step) - 1.0

    raise BiasConvergenceError(
        f"Gauss-Newton did not converge in {max_iter} iterations "
        f"(last step {largest:.3g}); the mismatch may be too large for a local fit"
    )


def bias_aware_snr(crlb_variance: FloatArray, bias: FloatArray, magnitude: float) -> FloatArray:
    """``m / sqrt(CRLB + b^2)``: the fault against the *total* error, not just its variance.

    Reduces to ``mask.detection_snr`` when ``b == 0``, and is never larger. An unidentifiable
    parameter (infinite variance) or an unbounded bias both give zero SNR.
    """
    if magnitude <= 0.0:
        raise ValueError("fault magnitude must be positive")
    variance = np.asarray(crlb_variance, dtype=float)
    squared_bias = np.asarray(bias, dtype=float) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = magnitude / np.sqrt(variance + squared_bias)
    return np.where(np.isfinite(snr), snr, 0.0)


def bias_ceiling(bias: FloatArray, magnitude: float) -> FloatArray:
    """``m / |b|``: the limit of ``bias_aware_snr`` as the variance is driven to zero.

    The most useful number in this module. Repeat *this* experiment as often as you like, or
    fit better sensors to it, and the diagnosis converges here -- neither touches ``b``.

    It is a ceiling for **this experiment**, not for this pack. A different duty cycle has a
    different residual and therefore a different ``b``; see the module docstring for a case
    where harder pulsing raises one parameter's ceiling 16x while dropping another's by 9x.
    If the ceiling is below the decision threshold, no amount of *this* measurement resolves
    the hypothesis. Only a different experiment, or a better model, can.
    """
    if magnitude <= 0.0:
        raise ValueError("fault magnitude must be positive")
    magnitude_of_bias = np.abs(np.asarray(bias, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(magnitude_of_bias > 0.0, magnitude / magnitude_of_bias, np.inf)
