"""Current excitation profiles.

Sign convention: positive is discharge. C-rate is relative to a nominal 60 Ah
cell unless another capacity is passed.

Why these four:

``constant_current``
    The *worst* excitation for separating a resistance fault from a capacity
    fault, because a constant IR offset and a slowly-drifting OCV offset look
    alike over a short window. Included so the identifiability analysis has
    something to fail on.

``pulse_train``
    Independent control of *mean* current (which sweeps SOC, exposing capacity)
    and *pulse amplitude* (which excites IR drop, exposing resistance). This is
    the profile the detectability heatmap sweeps over.

``random_walk``
    Broadband, in the spirit of NASA's Randomized Use battery data set. Realistic
    for driving.

``rest_then_pulse``
    The candidate diagnostic test. Rest lets the RC branch relax and the pack
    approach thermal equilibrium; the pulse then injects a clean current step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

NOMINAL_CAPACITY_AH: float = 60.0


@dataclass(frozen=True)
class DutyProfile:
    """A sampled current profile."""

    name: str
    time_s: FloatArray
    current_a: FloatArray

    @property
    def dt_s(self) -> float:
        return float(self.time_s[1] - self.time_s[0])

    @property
    def n_samples(self) -> int:
        return self.time_s.size

    @property
    def mean_current_a(self) -> float:
        return float(self.current_a.mean())

    @property
    def current_std_a(self) -> float:
        """Standard deviation of the current. Directly proportional to the
        Fisher information available about a series-resistance perturbation."""
        return float(self.current_a.std())


def _grid(duration_s: float, dt_s: float) -> FloatArray:
    return np.arange(0.0, duration_s, dt_s, dtype=float)


def constant_current(
    duration_s: float, dt_s: float, c_rate: float, capacity_ah: float = NOMINAL_CAPACITY_AH
) -> DutyProfile:
    t = _grid(duration_s, dt_s)
    return DutyProfile("constant_current", t, np.full(t.size, c_rate * capacity_ah))


def pulse_train(
    duration_s: float,
    dt_s: float,
    *,
    mean_c_rate: float = 0.2,
    pulse_c_rate: float = 1.0,
    period_s: float = 60.0,
    duty: float = 0.25,
    capacity_ah: float = NOMINAL_CAPACITY_AH,
) -> DutyProfile:
    """Square pulses of amplitude ``pulse_c_rate`` superimposed on a DC mean.

    The pulses are zero-mean by construction, so ``mean_c_rate`` alone controls
    charge throughput. That decoupling is what makes the detectability heatmap
    interpretable: moving along the excitation axis changes *only* the excitation,
    not how much SOC the pack traversed.
    """
    t = _grid(duration_s, dt_s)
    phase = np.mod(t, period_s) / period_s
    high = phase < duty
    # Zero-mean square wave: +A*(1-duty) when high, -A*duty when low.
    pulse = np.where(high, pulse_c_rate * (1.0 - duty), -pulse_c_rate * duty)
    return DutyProfile("pulse_train", t, (mean_c_rate + pulse) * capacity_ah)


def random_walk(
    duration_s: float,
    dt_s: float,
    *,
    mean_c_rate: float = 0.2,
    sigma_c_rate: float = 0.6,
    tau_s: float = 20.0,
    seed: int = 0,
    capacity_ah: float = NOMINAL_CAPACITY_AH,
) -> DutyProfile:
    """Ornstein-Uhlenbeck current: broadband but with a finite correlation time."""
    t = _grid(duration_s, dt_s)
    rng = np.random.default_rng(seed)
    decay = np.exp(-dt_s / tau_s)
    kick = sigma_c_rate * np.sqrt(1.0 - decay**2)
    x = np.empty(t.size)
    x[0] = 0.0
    for k in range(1, t.size):
        x[k] = decay * x[k - 1] + kick * rng.standard_normal()
    return DutyProfile("random_walk", t, (mean_c_rate + x) * capacity_ah)


def rest_then_pulse(
    duration_s: float,
    dt_s: float,
    *,
    rest_s: float = 300.0,
    pulse_c_rate: float = 2.0,
    pulse_s: float = 10.0,
    capacity_ah: float = NOMINAL_CAPACITY_AH,
) -> DutyProfile:
    """The candidate test: relax, then a short high-amplitude current step."""
    t = _grid(duration_s, dt_s)
    current = np.zeros(t.size)
    active = (t >= rest_s) & (t < rest_s + pulse_s)
    current[active] = pulse_c_rate * capacity_ah
    return DutyProfile("rest_then_pulse", t, current)
