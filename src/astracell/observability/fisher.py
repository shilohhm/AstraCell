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

from astracell.observability.sensitivity import TEMPERATURE_CHANNEL, VOLTAGE_CHANNEL
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


def fisher_information(sens: FloatArray, topology: SensorTopology, noise: NoiseModel) -> FloatArray:
    """``FIM = S^T Sigma^-1 S``, shape ``(n_params, n_params)``.

    Assumes independent Gaussian noise on every sample of every channel. Under a
    white-noise model this is exact, and it is why the FIM grows linearly with the
    number of samples: denser sampling really does buy information. Real AFE noise
    is not white, so this is an upper bound on the information -- see
    ``sensors/noise.py`` and ``LIMITATIONS.md``.
    """
    rows, kinds = design_matrix(sens, topology)
    if rows.shape[0] == 0:
        return np.zeros((sens.shape[3], sens.shape[3]))

    variances = np.where(kinds == 0, noise.voltage_variance, noise.temp_variance)
    weighted = rows / np.sqrt(variances)[:, None]
    fim = weighted.T @ weighted
    return 0.5 * (fim + fim.T)  # symmetrise away accumulation asymmetry


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


def information_gain(fim_before: FloatArray, fim_after: FloatArray) -> float:
    """``0.5 * log det(FIM_after / FIM_before)`` in nats: the D-optimality score.

    This is the expected reduction in the entropy of a Gaussian posterior when the
    information improves from ``fim_before`` to ``fim_after``. Used to rank
    candidate sensor placements and candidate diagnostic tests.
    """
    sign_a, logdet_a = np.linalg.slogdet(np.asarray(fim_after, dtype=float))
    sign_b, logdet_b = np.linalg.slogdet(np.asarray(fim_before, dtype=float))
    if sign_a <= 0 or sign_b <= 0:
        return float("inf") if sign_a > 0 else 0.0
    return 0.5 * float(logdet_a - logdet_b)
