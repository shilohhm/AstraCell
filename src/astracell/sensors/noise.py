"""Sensor noise model.

Numbers are chosen to be defensible against a datasheet, not optimistic:

* **Voltage**: sigma = 1 mV. An automotive AFE (LTC6813 / ADBMS / MAX17843 class)
  quotes ~1 mV total measurement error over temperature. Quantisation at 100 uV
  (16-bit over a ~6.5 V span) adds ``lsb^2/12`` in variance -- negligible next to
  the 1 mV, but present so nobody has to wonder.
* **Temperature**: sigma = 0.5 K. NTC + ADC, +/-1 K is typical spec; 0.5 K is a
  charitable read of it. Sampled at 1 Hz.
* **Current**: 0.5% of a 400 A full scale = 2 A random error, plus a systematic
  offset of the same size carried separately as ``current_bias_sigma_a``.
* **Correlation**: ``voltage_rho`` / ``temp_rho`` give each channel's noise an
  AR(1) lag-1 correlation, so 1/f behaviour can be modelled. Default 0 (white).

This module once carried a docstring asserting that relaxing the white-noise and
known-current idealisations "would only ever make the reported identifiability
*worse*, never better, which is the safe direction". **Both are now implemented,
both were measured, and that claim was false.** See
``examples/02_noise_robustness.py``:

* Correlated noise does not uniformly hurt; it *reallocates*. Whitening an AR(1)
  process is a scaled first difference, which suppresses a DC sensitivity by
  ``(1-rho)/(1+rho)`` and *amplifies* an alternating one by the reciprocal. At
  ``rho = 0.99`` the resistance bound is 2.6x **tighter** than under white noise,
  while capacity -- whose signature is a slow SOC ramp -- degrades 10x.
* Carrying the current bias as a nuisance parameter costs essentially nothing
  under pulsed excitation (32 voltage channels pin a common-mode offset far
  better than the shunt does), and is catastrophic under constant current, where
  every variance inflation factor crosses the multicollinearity threshold.

The defaults here (``rho = 0``, and ``include_current_bias=False`` in
``grey_cell_map``) remain the *optimistic* choice, so the shipped bounds are still
upper bounds. But "optimistic" is a statement about the defaults, not a theorem
about the idealisations. ``LIMITATIONS.md`` §2 records what was actually measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astracell.pack.simulate import SimResult
from astracell.sensors.topology import SensorTopology

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NoiseModel:
    """Additive Gaussian noise plus uniform quantisation, per channel type.

    Two fields deserve attention because they control how *optimistic* the Fisher
    information downstream is.

    ``voltage_rho`` / ``temp_rho``
        Lag-1 autocorrelation of the measurement noise at the sampling interval,
        modelled as AR(1). Zero recovers the white-noise case exactly.

        White noise means N samples buy N units of information: a 1200-sample window
        averages the noise down by sqrt(1200) = 35x. Real AFE noise has a 1/f
        component, so successive samples are correlated and the effective independent
        sample count is far lower. AR(1) is a tractable, honest proxy for that -- not
        a faithful 1/f spectrum, but it has the right qualitative effect and an exact,
        O(n) whitening transform.

        Crucially, AR(1) whitening acts as a first difference `x[t] - rho*x[t-1]`.
        It annihilates a *constant* sensitivity and barely touches one that oscillates
        at the excitation frequency. So a slowly-varying signature (a thermal
        response) loses far more information than a pulsed one (an IR drop). That is
        the same reason lock-in amplifiers exist.

    ``current_bias_sigma_a``
        A *systematic* offset of the shunt, distinct from its per-sample noise. The
        current that actually flowed differs from the recorded current by an unknown
        constant. Rather than pretend the current is known exactly, we carry this as a
        nuisance parameter with a Gaussian prior of this width, and marginalise. See
        ``observability.fisher.prior_information``.

    Both are set to their non-conservative defaults (rho = 0) so that turning them on
    is a deliberate, visible act. ``examples/02_honesty_pass.py`` measures what they cost.
    """

    voltage_sigma_v: float = 1.0e-3
    voltage_lsb_v: float = 100.0e-6
    voltage_rho: float = 0.0
    temp_sigma_k: float = 0.5
    temp_lsb_k: float = 0.0625
    temp_rho: float = 0.0
    current_sigma_frac_fs: float = 0.005
    current_full_scale_a: float = 400.0
    current_bias_sigma_a: float = 2.0  # 0.5% of a 400 A full scale

    def __post_init__(self) -> None:
        for name in ("voltage_rho", "temp_rho"):
            rho = getattr(self, name)
            if not -1.0 < rho < 1.0:
                raise ValueError(f"{name} must lie strictly inside (-1, 1), got {rho}")
        if self.current_bias_sigma_a <= 0.0:
            raise ValueError("current_bias_sigma_a must be positive")

    @property
    def voltage_variance(self) -> float:
        """Stationary variance of a voltage sample: Gaussian plus quantisation.

        When ``voltage_rho != 0`` this is treated as the stationary variance of the
        AR(1) process. Folding the (white) quantisation term into a correlated process
        is an approximation, and a harmless one here: at a 100 uV LSB the quantisation
        contributes 0.08% of the total variance.
        """
        return self.voltage_sigma_v**2 + self.voltage_lsb_v**2 / 12.0

    @property
    def temp_variance(self) -> float:
        return self.temp_sigma_k**2 + self.temp_lsb_k**2 / 12.0

    @property
    def current_sigma_a(self) -> float:
        """Per-sample current noise. Not the same thing as the systematic bias."""
        return self.current_sigma_frac_fs * self.current_full_scale_a

    def channel_variances(self, topology: SensorTopology) -> FloatArray:
        """Variance of each measurement channel, ordered [voltages..., temps...].

        This vector is the diagonal of Sigma in ``FIM = S^T Sigma^-1 S``.
        """
        return np.concatenate(
            [
                np.full(topology.n_voltage, self.voltage_variance),
                np.full(topology.n_temp, self.temp_variance),
            ]
        )


@dataclass(frozen=True)
class Measurements:
    """What the BMS reports. Time-major."""

    time_s: FloatArray
    voltage_v: FloatArray  # (n_time, n_voltage_channels)
    temp_k: FloatArray  # (n_time, n_temp_channels)
    current_a: FloatArray  # (n_time,)
    topology: SensorTopology

    @property
    def n_time(self) -> int:
        return self.time_s.size

    def as_vector(self) -> FloatArray:
        """Flatten to the measurement vector the Fisher information is built on."""
        return np.concatenate([self.voltage_v.ravel(), self.temp_k.ravel()])


def _quantise(values: FloatArray, lsb: float) -> FloatArray:
    if lsb <= 0.0:
        return values
    return np.round(values / lsb) * lsb


def measure(
    sim: SimResult,
    topology: SensorTopology,
    noise: NoiseModel,
    rng: np.random.Generator,
) -> Measurements:
    """Sample the true trajectory through the sensor topology and noise model."""
    if sim.n_cells != topology.n_cells:
        raise ValueError(f"sim has {sim.n_cells} cells, topology expects {topology.n_cells}")

    v_true = sim.voltage_v[:, topology.voltage_index]
    t_true = sim.temp_k[:, topology.temp_index]

    v_meas = _quantise(
        v_true + rng.normal(0.0, noise.voltage_sigma_v, v_true.shape), noise.voltage_lsb_v
    )
    t_meas = _quantise(t_true + rng.normal(0.0, noise.temp_sigma_k, t_true.shape), noise.temp_lsb_k)

    if topology.measures_pack_current:
        i_meas = sim.current_a + rng.normal(0.0, noise.current_sigma_a, sim.current_a.shape)
    else:
        i_meas = np.full_like(sim.current_a, np.nan)

    return Measurements(
        time_s=sim.time_s, voltage_v=v_meas, temp_k=t_meas, current_a=i_meas, topology=topology
    )
