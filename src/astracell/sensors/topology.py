"""Sensor topology: which cells are actually instrumented.

This is the constraint most battery-diagnostic projects never confront.

A production BMS measures:

===================  =========================================  ==========================
Quantity             Real coverage                              Consequence
===================  =========================================  ==========================
Cell voltage         one per cell (or parallel group), ~1 mV    per-cell V faults visible
Temperature          **4-12 sensors for ~96 cells**, +/-1 C     per-cell T faults are NOT
Current              **pack-level only**, one shunt             per-cell I is inferred
===================  =========================================  ==========================

So "highlight the faulty cell on a 3D map" is, for thermal faults, undecidable
from real telemetry. The job of this module is to make that asymmetry a
first-class object that the identifiability analysis can reason about, and that
you can ask counterfactual questions of: *if I added a thermocouple here, what
would become identifiable?*

Because ``astracell.observability`` computes sensitivities for **every** cell's
voltage and temperature -- not just the instrumented ones -- a sensor topology is
nothing more than a row-selection mask over that full sensitivity tensor.
Counterfactual sensor placement is therefore almost free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astracell.pack.topology import PackTopology

IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class SensorTopology:
    """Which cells report voltage, and which report temperature."""

    n_cells: int
    voltage_cells: tuple[int, ...]
    temp_cells: tuple[int, ...]
    measures_pack_current: bool = True

    def __post_init__(self) -> None:
        for name, chans in (("voltage_cells", self.voltage_cells), ("temp_cells", self.temp_cells)):
            if len(set(chans)) != len(chans):
                raise ValueError(f"{name} contains duplicates")
            if any(not 0 <= c < self.n_cells for c in chans):
                raise ValueError(f"{name} references a cell outside the pack")
        object.__setattr__(self, "voltage_cells", tuple(sorted(self.voltage_cells)))
        object.__setattr__(self, "temp_cells", tuple(sorted(self.temp_cells)))

    @property
    def n_voltage(self) -> int:
        return len(self.voltage_cells)

    @property
    def n_temp(self) -> int:
        return len(self.temp_cells)

    @property
    def n_channels(self) -> int:
        return self.n_voltage + self.n_temp

    @property
    def voltage_index(self) -> IntArray:
        return np.asarray(self.voltage_cells, dtype=int)

    @property
    def temp_index(self) -> IntArray:
        return np.asarray(self.temp_cells, dtype=int)

    def with_temp_sensor_at(self, cell: int) -> SensorTopology:
        """Counterfactual: what if we added a thermocouple on this cell?"""
        if cell in self.temp_cells:
            return self
        return SensorTopology(
            n_cells=self.n_cells,
            voltage_cells=self.voltage_cells,
            temp_cells=(*self.temp_cells, cell),
            measures_pack_current=self.measures_pack_current,
        )

    def without_temp_sensors(self) -> SensorTopology:
        """The pathological case, kept because it is instructive rather than absurd:
        cell voltage is a weak, heavily-confounded thermometer via R0(T)."""
        return SensorTopology(self.n_cells, self.voltage_cells, (), self.measures_pack_current)

    def summary(self) -> str:
        return (
            f"{self.n_cells} cells | {self.n_voltage} voltage channels | "
            f"{self.n_temp} temperature channels | "
            f"pack current: {'yes' if self.measures_pack_current else 'no'}"
        )


def all_cells_voltage(n_cells: int) -> tuple[int, ...]:
    return tuple(range(n_cells))


def evenly_spaced_temp_sensors(pack: PackTopology, n_sensors: int) -> tuple[int, ...]:
    """Place ``n_sensors`` thermocouples spread over the grid.

    Sensors are distributed across modules first (thermally the weak axis, so
    each module needs its own), then evenly along each module.
    """
    if n_sensors < 0:
        raise ValueError("n_sensors must be non-negative")
    if n_sensors == 0:
        return ()
    if n_sensors > pack.n_cells:
        raise ValueError(f"cannot place {n_sensors} sensors in {pack.n_cells} cells")

    # Spread over modules first (the thermally weak axis), then within each module.
    module_of = np.floor((np.arange(n_sensors) + 0.5) * pack.n_modules / n_sensors).astype(int)

    cells: list[int] = []
    for module in np.unique(module_of):
        count = int(np.count_nonzero(module_of == module))
        # Interior-biased placement: avoids putting every sensor on a pack edge.
        positions = np.floor((np.arange(count) + 0.5) * pack.cells_per_module / count).astype(int)
        cells.extend(int(pack.index(int(module), int(p))) for p in positions)
    return tuple(sorted(set(cells)))


def realistic_topology(pack: PackTopology, n_temp_sensors: int = 4) -> SensorTopology:
    """Every cell's voltage; a handful of thermocouples. The real situation."""
    return SensorTopology(
        n_cells=pack.n_cells,
        voltage_cells=all_cells_voltage(pack.n_cells),
        temp_cells=evenly_spaced_temp_sensors(pack, n_temp_sensors),
    )
