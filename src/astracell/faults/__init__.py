"""Fault injection. Mutates ground truth, never the observer's model."""

from astracell.faults.injector import apply_physical_faults, apply_sensor_faults
from astracell.faults.library import (
    PhysicalFault,
    PhysicalFaultKind,
    SensorFault,
    SensorFaultKind,
    cooling_weakness,
    high_internal_resistance,
    reduced_capacity,
    temp_sensor_bias,
    voltage_sensor_bias,
)

__all__ = [
    "PhysicalFault",
    "PhysicalFaultKind",
    "SensorFault",
    "SensorFaultKind",
    "apply_physical_faults",
    "apply_sensor_faults",
    "cooling_weakness",
    "high_internal_resistance",
    "reduced_capacity",
    "temp_sensor_bias",
    "voltage_sensor_bias",
]
