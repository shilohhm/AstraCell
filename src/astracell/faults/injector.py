"""Applying faults. Everything here returns a new value; nothing is mutated.

That constraint is not stylistic. The ground-truth ``PackParams`` is the reference
against which every identifiability claim is measured. If a fault injector could
mutate it in place, a stale reference would silently become a wrong answer, and
the failure would look like a scientific result rather than a bug.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np

from astracell.faults.library import PhysicalFault, SensorFault, SensorFaultKind
from astracell.pack.params import PackParams
from astracell.sensors.noise import Measurements


def apply_physical_faults(params: PackParams, faults: Iterable[PhysicalFault]) -> PackParams:
    """Return a new PackParams with every physical fault applied.

    Faults on the same (field, cell) compose multiplicatively, which is the
    physically sensible reading of "the resistance grew 20%, then another 10%".
    """
    out = params
    for fault in faults:
        if fault.cell >= params.n_cells:
            raise IndexError(f"fault targets cell {fault.cell}, pack has {params.n_cells}")
        out = out.scale_cell(fault.field, fault.cell, fault.factor)
    return out


def apply_sensor_faults(meas: Measurements, faults: Iterable[SensorFault]) -> Measurements:
    """Return new Measurements with additive sensor biases applied."""
    voltage = np.array(meas.voltage_v, copy=True)
    temp = np.array(meas.temp_k, copy=True)

    for fault in faults:
        if fault.kind is SensorFaultKind.VOLTAGE_SENSOR_BIAS:
            if fault.channel >= meas.topology.n_voltage:
                raise IndexError(
                    f"voltage bias on channel {fault.channel}, "
                    f"topology has {meas.topology.n_voltage} voltage channels"
                )
            voltage[:, fault.channel] += fault.bias
        else:
            if fault.channel >= meas.topology.n_temp:
                raise IndexError(
                    f"temperature bias on channel {fault.channel}, "
                    f"topology has {meas.topology.n_temp} temperature channels"
                )
            temp[:, fault.channel] += fault.bias

    return replace(meas, voltage_v=voltage, temp_k=temp)
