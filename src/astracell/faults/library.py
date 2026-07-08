"""The fault library.

Faults split into two families, and the split is not cosmetic:

**Physical faults** change the pack. They perturb ``PackParams``, propagate
through the physics, and show up in the measurements as the model says they must.

**Sensor faults** change only the instrument. The pack is healthy; the number is
wrong.

Distinguishing these two families is the harder half of battery diagnosis, and
it is exactly where a system that reasons about observability earns its keep: a
stuck thermocouple and a genuine local hotspot produce the same reading, and the
only thing that separates them is whether the reading obeys the physics that the
*other* sensors are simultaneously reporting.

Magnitude conventions -- fixed here, once, so nothing downstream has to guess:

===========================  ============================================
``HIGH_R``, m > 0            ``r0[cell] *= (1 + m)``
``REDUCED_CAPACITY``, m > 0  ``capacity_ah[cell] *= (1 - m)``
``COOLING_WEAKNESS``, m > 0  ``ha_w_per_k[cell] *= (1 - m)``
``VOLTAGE_SENSOR_BIAS``      additive offset in volts on one channel
``TEMPERATURE_SENSOR_BIAS``  additive offset in kelvin on one channel
===========================  ============================================

All faults here are **step** faults applied for the whole run. Ramped onsets are
the honest next step (a step fault is easier to detect than a real one) and are
deliberately deferred rather than faked -- see ``LIMITATIONS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PhysicalFaultKind(StrEnum):
    """Faults that perturb the pack's ground-truth parameters."""

    HIGH_R = "high_internal_resistance"
    REDUCED_CAPACITY = "reduced_capacity"
    COOLING_WEAKNESS = "cooling_weakness"


class SensorFaultKind(StrEnum):
    """Faults that perturb only the measurement, leaving the pack healthy."""

    VOLTAGE_SENSOR_BIAS = "voltage_sensor_bias"
    TEMPERATURE_SENSOR_BIAS = "temperature_sensor_bias"


# Maps a physical fault to the PackParams field it scales, and the sign of the
# relative perturbation for a positive magnitude.
_FIELD_AND_SIGN: dict[PhysicalFaultKind, tuple[str, float]] = {
    PhysicalFaultKind.HIGH_R: ("r0_ohm", +1.0),
    PhysicalFaultKind.REDUCED_CAPACITY: ("capacity_ah", -1.0),
    PhysicalFaultKind.COOLING_WEAKNESS: ("ha_w_per_k", -1.0),
}


@dataclass(frozen=True)
class PhysicalFault:
    """A relative perturbation of one cell's physics."""

    kind: PhysicalFaultKind
    cell: int
    magnitude: float

    def __post_init__(self) -> None:
        if self.cell < 0:
            raise ValueError("cell index must be non-negative")
        if not 0.0 <= self.magnitude < 1.0:
            raise ValueError(f"magnitude must be in [0, 1), got {self.magnitude}")

    @property
    def field(self) -> str:
        return _FIELD_AND_SIGN[self.kind][0]

    @property
    def factor(self) -> float:
        """The multiplicative factor applied to ``field[cell]``."""
        _, sign = _FIELD_AND_SIGN[self.kind]
        return 1.0 + sign * self.magnitude

    def describe(self) -> str:
        pct = 100.0 * self.magnitude
        return (
            f"{self.kind.value} on cell {self.cell}: {self.field} x{self.factor:.3f} ({pct:.1f}%)"
        )


@dataclass(frozen=True)
class SensorFault:
    """An additive bias on one measurement channel.

    ``channel`` indexes into ``SensorTopology.voltage_cells`` or ``.temp_cells``,
    not into the pack. A pack may have 32 cells but only 4 thermocouples; a bias
    on "temperature channel 2" is a statement about the third thermocouple.
    """

    kind: SensorFaultKind
    channel: int
    bias: float

    def __post_init__(self) -> None:
        if self.channel < 0:
            raise ValueError("channel index must be non-negative")

    def describe(self) -> str:
        unit = "V" if self.kind is SensorFaultKind.VOLTAGE_SENSOR_BIAS else "K"
        return f"{self.kind.value} on channel {self.channel}: {self.bias:+.4g} {unit}"


# ---------------------------------------------------------------------------
# Constructors. Named so a reader of a script knows what happened without
# looking up an enum.
# ---------------------------------------------------------------------------
def high_internal_resistance(cell: int, magnitude: float) -> PhysicalFault:
    """R0 grows by ``magnitude`` (0.20 -> +20%)."""
    return PhysicalFault(PhysicalFaultKind.HIGH_R, cell, magnitude)


def reduced_capacity(cell: int, magnitude: float) -> PhysicalFault:
    """Usable capacity falls by ``magnitude`` (0.05 -> -5%)."""
    return PhysicalFault(PhysicalFaultKind.REDUCED_CAPACITY, cell, magnitude)


def cooling_weakness(cell: int, magnitude: float) -> PhysicalFault:
    """Convective coupling hA falls by ``magnitude`` (0.40 -> -40%)."""
    return PhysicalFault(PhysicalFaultKind.COOLING_WEAKNESS, cell, magnitude)


def voltage_sensor_bias(channel: int, bias_v: float) -> SensorFault:
    return SensorFault(SensorFaultKind.VOLTAGE_SENSOR_BIAS, channel, bias_v)


def temp_sensor_bias(channel: int, bias_k: float) -> SensorFault:
    return SensorFault(SensorFaultKind.TEMPERATURE_SENSOR_BIAS, channel, bias_k)
