"""Fisher information and the Cramer-Rao lower bound.

For a measurement model ``y = h(theta) + e``, ``e ~ N(0, Sigma)`` with Sigma
diagonal, the Fisher information about theta is

    FIM = S^T Sigma^-1 S,     S[i, j] = dy_i / dtheta_j

and any unbiased estimator of theta_j has variance at least ``[FIM^-1]_jj``. That
lower bound is a statement about *every possible estimator*, not about a
particular algorithm. It is the right object for the question "could I resolve
this fault?", because it answers it without committing to how.

Two implementation details that matter more than they look:

**Do not use ``pinv``.** The Moore-Penrose pseudo-inverse of a rank-deficient FIM
returns the minimum-norm solution, which produces *finite* variances for
parameters that are in fact completely unidentified. Using it would make the
system claim it can see things it cannot -- precisely the failure mode this whole
package exists to prevent. Instead we eigendecompose, discard directions whose
eigenvalue is at the level of floating-point noise, and return ``inf`` for any
parameter with support on a discarded direction.

**Rank deficiency is rare; near-degeneracy is the norm.** A parameter with tiny
but genuine information (the hA of a cell four positions from the nearest
thermocouple) gets a finite, enormous CRLB, not ``inf``. It is the *SNR
threshold* in ``mask.py`` that calls it unobservable, not a numerical rank cut.
That is deliberate: the honest statement is "you would need a 4000% cooling fault
to see this", not "this is mathematically invisible".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astracell.observability.sensitivity import (
    TEMPERATURE_CHANNEL,
    VOLTAGE_CHANNEL,
    ParameterSpec,
    ParamKind,
)
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import SensorTopology

FloatArray = NDArray[np.float64]

# Eigenvalues below rcond * lambda_max are indistinguishable from the numerical
# noise of accumulating ~1e4-1e5 outer products in float64.
DEFAULT_RCOND: float = 1e-11
# A parameter with more than this fraction of its squared eigen-projection on
# discarded directions has unbounded variance.
DEFAULT_LEAK_TOL: float = 1e-8


def design_matrix(sens: FloatArray, topology: SensorTopology) -> tuple[FloatArray, FloatArray]:
    """Select the measured rows out of the full sensitivity tensor.

    Parameters
    ----------
    sens : ``(n_time, n_cells, 2, n_params)`` from ``observability.sensitivity``.
    topology : which cells report voltage, which report temperature.

    Returns
    -------
    rows : ``(n_measurements, n_params)`` stacked ``[all voltages; all temps]``.
    channel_kind : ``(n_measurements,)`` of 0 for voltage rows, 1 for temperature.
    """
    if sens.ndim != 4:
        raise ValueError(f"expected a 4-D sensitivity tensor, got shape {sens.shape}")
    n_time, n_cells, _, n_params = sens.shape
    if n_cells != topology.n_cells:
        raise ValueError(f"sensitivity has {n_cells} cells, topology expects {topology.n_cells}")

    blocks: list[FloatArray] = []
    kinds: list[FloatArray] = []

    if topology.n_voltage:
        v_block = sens[:, topology.voltage_index, VOLTAGE_CHANNEL, :]
        blocks.append(v_block.reshape(n_time * topology.n_voltage, n_params))
        kinds.append(np.zeros(n_time * topology.n_voltage, dtype=int))
    if topology.n_temp:
        t_block = sens[:, topology.temp_index, TEMPERATURE_CHANNEL, :]
        blocks.append(t_block.reshape(n_time * topology.n_temp, n_params))
        kinds.append(np.ones(n_time * topology.n_temp, dtype=int))

    if not blocks:
        return np.zeros((0, n_params)), np.zeros(0, dtype=int)
    return np.vstack(blocks), np.concatenate(kinds)


def whiten_ar1(block: FloatArray, rho: float) -> FloatArray:
    """Cholesky-whiten an AR(1) noise process along the leading (time) axis.

    For noise ``e_t = rho * e_{t-1} + u_t`` with unit stationary variance, the exact
    whitening transform -- the Cholesky factor ``L`` of ``R^-1 = L^T L`` -- is

        x~[0] = x[0]
        x~[t] = ( x[t] - rho * x[t-1] ) / sqrt(1 - rho^2)        for t >= 1

    so that ``x~^T x~ == x^T R^-1 x`` exactly, where ``R[i,j] = rho^|i-j|``. O(n), and no
    n x n matrix is ever formed. ``rho = 0`` returns the input unchanged, so the
    white-noise case is recovered bit-for-bit rather than approximately.

    Getting the ``sqrt(1 - rho^2)`` onto the right row matters: put it on ``x~[0]``
    instead and every information number is scaled by ``(1 - rho^2)``, silently. That is
    an easy mistake and it is caught by
    ``tests/test_noise_correlation.py::test_whitening_reproduces_the_inverse_correlation_quadratic_form``,
    which builds ``R`` densely and inverts it.

    What this does to information is the whole robustness story. The transform is a scaled
    first difference. A **constant** sensitivity keeps only a fraction ``(1-rho)/(1+rho)``
    of its information; an **alternating** one *gains* a factor ``(1+rho)/(1-rho)``. Slow
    signatures pay. Fast ones are paid.
    """
    if rho == 0.0:
        return block
    out = np.empty_like(block)
    out[0] = block[0]
    out[1:] = (block[1:] - rho * block[:-1]) / np.sqrt(1.0 - rho * rho)
    return out


def channel_slices(
    topology: SensorTopology, noise: NoiseModel
) -> list[tuple[NDArray[np.intp], int, float, float]]:
    """``[(cell_index, channel, sigma, rho), ...]`` for every instrumented channel group.

    The single source of truth for "which cells are measured, on which channel, with what
    noise". ``fisher_information`` and ``bias.residual_score`` both index through this, so a
    residual can never be whitened differently from the sensitivities it is projected onto.
    """
    slices: list[tuple[NDArray[np.intp], int, float, float]] = []
    if topology.n_voltage:
        slices.append(
            (
                topology.voltage_index,
                VOLTAGE_CHANNEL,
                float(np.sqrt(noise.voltage_variance)),
                noise.voltage_rho,
            )
        )
    if topology.n_temp:
        slices.append(
            (
                topology.temp_index,
                TEMPERATURE_CHANNEL,
                float(np.sqrt(noise.temp_variance)),
                noise.temp_rho,
            )
        )
    return slices


def _channel_blocks(
    sens: FloatArray, topology: SensorTopology, noise: NoiseModel
) -> list[tuple[FloatArray, float, float]]:
    """``[(block, sigma, rho), ...]`` with block shaped ``(n_time, n_chan, n_params)``."""
    return [
        (sens[:, index, channel, :], sigma, rho)
        for index, channel, sigma, rho in channel_slices(topology, noise)
    ]


def fisher_information(
    sens: FloatArray,
    topology: SensorTopology,
    noise: NoiseModel,
    specs: tuple[ParameterSpec, ...] | None = None,
) -> FloatArray:
    """``FIM = S^T Sigma^-1 S``, shape ``(n_params, n_params)``.

    ``Sigma`` is block-diagonal by channel; within a channel the noise is AR(1) with
    correlation ``noise.voltage_rho`` / ``noise.temp_rho``. With rho = 0 this reduces
    exactly to independent samples, and the FIM grows linearly with sample count --
    denser sampling really does buy information, *if* the noise is white.

    Pass ``specs`` to add Gaussian priors on nuisance parameters, which turns this into
    a Bayesian (Van Trees) information matrix. Currently the only such parameter is the
    current-sensor bias; see ``prior_information``.
    """
    n_params = sens.shape[3]
    fim = np.zeros((n_params, n_params))

    for block, sigma, rho in _channel_blocks(sens, topology, noise):
        n_time, n_chan, _ = block.shape
        whitened = whiten_ar1(block, rho).reshape(n_time * n_chan, n_params) / sigma
        fim += whitened.T @ whitened

    if specs is not None:
        if len(specs) != n_params:
            raise ValueError(f"{len(specs)} specs for {n_params} parameter columns")
        fim = fim + prior_information(specs, noise)

    return 0.5 * (fim + fim.T)  # symmetrise away accumulation asymmetry


def prior_information(specs: tuple[ParameterSpec, ...], noise: NoiseModel) -> FloatArray:
    """Diagonal prior information for nuisance parameters, ``1 / sigma_prior^2``.

    The current-sensor bias is not determined by the pack's voltages alone -- a constant
    offset in the current aliases, in part, with a common-mode resistance change. What
    pins it down is the shunt's own accuracy specification, which is exactly a Gaussian
    prior of width ``noise.current_bias_sigma_a``.

    Adding it here makes the resulting bound a **Bayesian Cramer-Rao bound** (Van Trees).
    Without the prior, a marginal current bias would be unidentifiable under constant
    current and the CRLB for every cell's R0 would blow up -- which would be pessimistic
    rather than honest, because we do in fact know the shunt is within spec.

    Physical (per-cell) parameters get no prior. We are not willing to assume a cell is
    healthy in order to conclude that it is healthy.
    """
    n = len(specs)
    prior = np.zeros((n, n))
    for i, spec in enumerate(specs):
        if spec.kind is ParamKind.CURRENT_BIAS:
            prior[i, i] = 1.0 / noise.current_bias_sigma_a**2
    return prior


def crlb(
    fim: FloatArray, *, rcond: float = DEFAULT_RCOND, leak_tol: float = DEFAULT_LEAK_TOL
) -> FloatArray:
    """Cramer-Rao lower bound on the *variance* of each parameter estimate.

    Returns ``inf`` for any parameter that is not identifiable -- that is, one
    whose eigen-projection has non-negligible support on a direction the data
    carries no information about.

    Units are (relative perturbation)^2, so ``sqrt(crlb)[j] == 0.04`` means "no
    unbiased estimator can pin down parameter j to better than +/-4% (1 sigma)".
    """
    fim = np.asarray(fim, dtype=float)
    if fim.ndim != 2 or fim.shape[0] != fim.shape[1]:
        raise ValueError(f"FIM must be square, got shape {fim.shape}")
    n = fim.shape[0]

    fim = 0.5 * (fim + fim.T)
    eigvals, eigvecs = np.linalg.eigh(fim)
    lam_max = float(eigvals.max()) if n else 0.0
    if not np.isfinite(lam_max) or lam_max <= 0.0:
        return np.full(n, np.inf)

    identified = eigvals > rcond * lam_max
    if not identified.any():
        return np.full(n, np.inf)

    projection_sq = eigvecs**2  # rows: parameters, cols: eigen-directions
    variance = (projection_sq[:, identified] / eigvals[identified]).sum(axis=1)
    leakage = projection_sq[:, ~identified].sum(axis=1)

    return np.where(leakage > leak_tol, np.inf, variance)


def crlb_std(fim: FloatArray, **kwargs: float) -> FloatArray:
    """``sqrt(crlb)``: the best achievable 1-sigma uncertainty, as a fraction."""
    return np.sqrt(crlb(fim, **kwargs))


def variance_inflation(
    fim: FloatArray, *, rcond: float = DEFAULT_RCOND, leak_tol: float = DEFAULT_LEAK_TOL
) -> FloatArray:
    """Variance inflation factor per parameter: ``FIM_jj * [FIM^-1]_jj``.

    This is the right scalar for the question "can I isolate parameter j from the
    others?", and it is not the condition number.

    * ``VIF_j == 1`` means parameter j is orthogonal to every other parameter: the
      data determines it as well as it would if all the others were known exactly.
    * ``VIF_j == 25`` means confounding with the other parameters has inflated its
      variance 25-fold; its standard error is 5x worse than it looks.

    The condition number of the whole FIM cannot do this job. It is a property of
    the matrix, not of a parameter, and it is dominated by whichever direction is
    worst-informed. A pack whose ``hA`` is nearly invisible will have a huge
    ``cond(FIM)`` even when its ``R0`` is perfectly isolated -- gating on that
    would refuse every diagnosis.

    ``VIF > 10`` is the conventional regression-diagnostics threshold for serious
    multicollinearity. We inherit it rather than invent one.
    """
    fim = np.asarray(fim, dtype=float)
    fim = 0.5 * (fim + fim.T)
    diagonal = np.diag(fim)
    marginal_variance = crlb(fim, rcond=rcond, leak_tol=leak_tol)
    with np.errstate(invalid="ignore"):
        vif = diagonal * marginal_variance
    # A parameter with no information at all is not "confounded", it is absent.
    return np.where(diagonal > 0.0, vif, np.inf)


def condition_number(fim: FloatArray) -> float:
    """``lambda_max / lambda_min`` of the FIM.

    A large value means some linear combination of the parameters is far better
    determined than another -- i.e. the parameters are confounded. For the pair
    (R0, capacity) of one cell, this is the quantitative form of "the IR drop and
    the OCV shift look alike over this window". Returns ``inf`` if singular.
    """
    fim = np.asarray(fim, dtype=float)
    eigvals = np.linalg.eigvalsh(0.5 * (fim + fim.T))
    lam_min, lam_max = float(eigvals.min()), float(eigvals.max())
    if lam_min <= 0.0 or not np.isfinite(lam_max):
        return float("inf")
    return lam_max / lam_min


def gaussian_entropy(fim: FloatArray) -> float:
    """Differential entropy [nats] of a Gaussian posterior with covariance ``FIM^-1``.

        H = 0.5 * ( k*log(2*pi*e) - log det FIM )

    Infinite when the FIM is singular: an unidentified direction carries unbounded
    uncertainty, and saying so is the point.
    """
    fim = np.asarray(fim, dtype=float)
    k = fim.shape[0]
    sign, logdet = np.linalg.slogdet(fim)
    if sign <= 0:
        return float("inf")
    return 0.5 * (k * float(np.log(2.0 * np.pi * np.e)) - float(logdet))


def information_gain(fim_before: FloatArray, fim_after: FloatArray) -> float:
    """``0.5 * log det(FIM_after / FIM_before)`` in nats: the D-optimality score.

    Exactly ``gaussian_entropy(before) - gaussian_entropy(after)``: the reduction in the
    differential entropy of a Gaussian posterior when the information improves. Used to
    rank candidate sensor placements and candidate diagnostic tests.

    Returns ``inf`` when the prior information is singular but the posterior is not --
    an experiment that makes an unidentified parameter identifiable gains, formally,
    unbounded information. Returns ``0.0`` when the posterior is still singular.
    """
    sign_a, logdet_a = np.linalg.slogdet(np.asarray(fim_after, dtype=float))
    sign_b, logdet_b = np.linalg.slogdet(np.asarray(fim_before, dtype=float))
    if sign_a <= 0:
        return 0.0
    if sign_b <= 0:
        return float("inf")
    return 0.5 * float(logdet_a - logdet_b)
