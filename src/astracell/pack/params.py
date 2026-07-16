"""Pack parameter container. Immutable by construction.

Fault injection returns a *new* ``PackParams``; nothing in this package mutates
one in place. The underlying numpy arrays are marked read-only so that an
accidental in-place write raises instead of silently corrupting the ground truth
that the whole identifiability analysis is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astracell.cell.ecm import DEFAULT_EA_OVER_R_K, CellElectricalParams
from astracell.cell.ocv import NMC_LIKE, OcvCurve
from astracell.cell.thermal import CellThermalParams
from astracell.pack.topology import PackTopology

FloatArray = NDArray[np.float64]

# Loosely representative of a large-format prismatic NMC cell. Not a datasheet.
NOMINAL_CAPACITY_AH: float = 60.0
NOMINAL_R0_OHM: float = 1.5e-3
NOMINAL_R1_OHM: float = 0.8e-3
NOMINAL_C1_FARAD: float = 2500.0  # tau = R1*C1 = 2 s
NOMINAL_HEAT_CAPACITY_J_PER_K: float = 900.0
NOMINAL_HA_W_PER_K: float = 2.5
NOMINAL_K_INTRA_W_PER_K: float = 0.6
NOMINAL_K_INTER_W_PER_K: float = 0.15
NOMINAL_COOLANT_TEMP_K: float = 298.15


def _frozen(values: FloatArray) -> FloatArray:
    out = np.array(values, dtype=float, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class PackParams:
    """Ground-truth pack parameters. Treat as a value, never as a buffer."""

    topology: PackTopology
    curve: OcvCurve
    capacity_ah: FloatArray
    r0_ohm: FloatArray
    r1_ohm: FloatArray
    c1_farad: FloatArray
    heat_capacity_j_per_k: FloatArray
    ha_w_per_k: FloatArray
    k_intra_w_per_k: float = NOMINAL_K_INTRA_W_PER_K
    k_inter_w_per_k: float = NOMINAL_K_INTER_W_PER_K
    coolant_temp_k: float = NOMINAL_COOLANT_TEMP_K
    coulombic_efficiency: float = 1.0
    ea_over_r_k: float = DEFAULT_EA_OVER_R_K
    # Optional second RC branch (v0.8 "depth"). None on both = the first-order ECM.
    r2_ohm: FloatArray | None = None
    c2_farad: FloatArray | None = None

    def __post_init__(self) -> None:
        n = self.topology.n_cells
        for field in (
            "capacity_ah",
            "r0_ohm",
            "r1_ohm",
            "c1_farad",
            "heat_capacity_j_per_k",
            "ha_w_per_k",
        ):
            arr = _frozen(getattr(self, field))
            if arr.shape != (n,):
                raise ValueError(f"{field} has shape {arr.shape}, expected ({n},)")
            if not np.all(arr > 0.0):
                raise ValueError(f"{field} must be strictly positive")
            object.__setattr__(self, field, arr)

        # The second RC branch is optional but must be complete: both resistor and
        # capacitor, or neither. Half a branch is a construction bug, not a first-order
        # model, so refuse it rather than silently ignore the orphaned field.
        if (self.r2_ohm is None) != (self.c2_farad is None):
            raise ValueError("second RC branch needs both r2_ohm and c2_farad, or neither")
        if self.r2_ohm is not None:
            for field in ("r2_ohm", "c2_farad"):
                arr = _frozen(getattr(self, field))
                if arr.shape != (n,):
                    raise ValueError(f"{field} has shape {arr.shape}, expected ({n},)")
                if not np.all(arr > 0.0):
                    raise ValueError(f"{field} must be strictly positive")
                object.__setattr__(self, field, arr)

    @property
    def n_cells(self) -> int:
        return self.topology.n_cells

    @property
    def electrical(self) -> CellElectricalParams:
        return CellElectricalParams(
            capacity_ah=self.capacity_ah,
            r0_ohm=self.r0_ohm,
            r1_ohm=self.r1_ohm,
            c1_farad=self.c1_farad,
            coulombic_efficiency=self.coulombic_efficiency,
            ea_over_r_k=self.ea_over_r_k,
            r2_ohm=self.r2_ohm,
            c2_farad=self.c2_farad,
        )

    @property
    def thermal(self) -> CellThermalParams:
        return CellThermalParams(
            heat_capacity_j_per_k=self.heat_capacity_j_per_k,
            ha_w_per_k=self.ha_w_per_k,
            coolant_temp_k=self.coolant_temp_k,
        )

    def conductance_matrix(self) -> FloatArray:
        return self.topology.conductance_matrix(self.k_intra_w_per_k, self.k_inter_w_per_k)

    def evolve(self, **changes: Any) -> PackParams:
        """Return a new PackParams with the given fields replaced."""
        return replace(self, **changes)

    def scale_cell(self, field: str, cell: int, factor: float) -> PackParams:
        """Return a new PackParams with ``field[cell] *= factor``."""
        if not hasattr(self, field):
            raise AttributeError(f"PackParams has no field {field!r}")
        arr = np.array(getattr(self, field), dtype=float, copy=True)
        arr[cell] *= factor
        return self.evolve(**{field: arr})


def nominal_pack(
    topology: PackTopology,
    *,
    curve: OcvCurve = NMC_LIKE,
    seed: int | None = 0,
    r0_spread: float = 0.03,
    capacity_spread: float = 0.015,
    ha_spread: float = 0.05,
) -> PackParams:
    """A healthy pack with realistic cell-to-cell manufacturing spread.

    ``r0_spread`` is the log-normal sigma on R0; the others are relative
    Gaussian sigmas. Set ``seed=None`` for an exactly uniform pack, which is
    useful when you want the identifiability result to be attributable to the
    sensor topology alone rather than to a lucky parameter draw.
    """
    n = topology.n_cells
    if seed is None:
        r0 = np.full(n, NOMINAL_R0_OHM)
        capacity = np.full(n, NOMINAL_CAPACITY_AH)
        ha = np.full(n, NOMINAL_HA_W_PER_K)
    else:
        rng = np.random.default_rng(seed)
        r0 = NOMINAL_R0_OHM * rng.lognormal(mean=0.0, sigma=r0_spread, size=n)
        capacity = NOMINAL_CAPACITY_AH * (1.0 + capacity_spread * rng.standard_normal(n))
        ha = NOMINAL_HA_W_PER_K * (1.0 + ha_spread * rng.standard_normal(n))

    return PackParams(
        topology=topology,
        curve=curve,
        capacity_ah=capacity,
        r0_ohm=r0,
        r1_ohm=np.full(n, NOMINAL_R1_OHM),
        c1_farad=np.full(n, NOMINAL_C1_FARAD),
        heat_capacity_j_per_k=np.full(n, NOMINAL_HEAT_CAPACITY_J_PER_K),
        ha_w_per_k=ha,
    )
