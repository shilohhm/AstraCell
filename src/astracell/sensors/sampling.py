"""Draw correlated measurement noise.

Everywhere else the repository only ever *whitens* noise (``fisher.whiten_ar1``); it computes
information as though a realisation existed without drawing one, because the Cramer-Rao bound
needs only the covariance. Calibration needs the realisation: to ask whether an estimator's
scatter matches the bound you must run the estimator on data, and data is signal plus a
specific noise draw.

The noise drawn here has covariance **exactly** ``Sigma = sigma^2 R``, ``R_ij = rho^|i-j|`` --
the same ``Sigma`` the Fisher information inverts. That identity is the whole point: it is what
makes an empirical estimator covariance comparable to the CRLB rather than to some other
matrix. Two deliberate choices keep it exact.

* AR(1) is generated in its **stationary** form, ``x[0] ~ N(0, sigma^2)``, so
  ``Cov(x_s, x_t) = sigma^2 rho^|s-t|`` with no burn-in transient. ``whiten_ar1`` inverts
  exactly this; start it at ``N(0, 1)`` instead and the first sample's variance is wrong.
* Quantisation is modelled as its variance-equivalent Gaussian term (``lsb^2/12``, already in
  ``NoiseModel.voltage_variance``), **not** as rounding. Rounding would make the noise
  non-Gaussian and ``Sigma_sample != Sigma_FIM``, so the CRLB comparison would be against the
  wrong matrix. At a 100 uV LSB the two differ by 0.08% of the variance; nothing is lost and
  the identity is kept. ``sensors.noise.measure`` rounds, because it models the ADC; this
  samples the *statistical model the FIM assumes*, which is a different and deliberate thing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astracell.observability.fisher import channel_slices
from astracell.observability.sensitivity import N_OUTPUT_CHANNELS
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import SensorTopology

FloatArray = NDArray[np.float64]


def ar1_series(n_time: int, n_series: int, rho: float, rng: np.random.Generator) -> FloatArray:
    """``(n_time, n_series)`` unit-variance stationary AR(1) noise.

    ``rho = 0`` short-circuits to i.i.d. standard normals -- the white case, exactly, with no
    per-step recursion. Otherwise ``x[t] = rho*x[t-1] + sqrt(1-rho^2) w[t]`` started from a
    stationary first sample, so the marginal variance is 1 at every ``t``.
    """
    if n_time <= 0:
        raise ValueError("n_time must be positive")
    white = rng.standard_normal((n_time, n_series))
    if rho == 0.0:
        return white
    out = np.empty_like(white)
    out[0] = white[0]
    scale = np.sqrt(1.0 - rho * rho)
    for t in range(1, n_time):
        out[t] = rho * out[t - 1] + scale * white[t]
    return out


def sample_measurement_noise(
    topology: SensorTopology,
    noise: NoiseModel,
    n_time: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Additive noise for one experiment, shaped like the sensitivity/residual tensor.

    Returns ``(n_time, n_cells, N_OUTPUT_CHANNELS)``. Only *instrumented* channels are
    non-zero; an uninstrumented cell carries no noise because it carries no measurement.
    Each instrumented group is drawn with its own ``sigma`` and ``rho`` so that adding this
    to ``plant - observer`` and projecting with ``residual_score`` reproduces ``S^T Sigma^-1``
    acting on a genuine ``N(0, Sigma)`` draw.
    """
    out = np.zeros((n_time, topology.n_cells, N_OUTPUT_CHANNELS))
    for index, channel, sigma, rho in channel_slices(topology, noise):
        out[:, index, channel] = sigma * ar1_series(n_time, index.size, rho, rng)
    return out
