"""Pack geometry: cells laid out as a 2D grid of (module, position).

All cells are in one series string, so they share a single current. The grid is
purely *thermal* geometry -- it determines which cells conduct heat to which,
and therefore how far a thermal disturbance travels before it reaches a sensor.

    module 0:  [ 0][ 1][ 2] ... [c-1]
    module 1:  [ c][c+1] ...
    ...

Intra-module neighbours (same row) conduct well. Inter-module neighbours (same
column, adjacent row) conduct poorly, because in a real pack there is a module
wall, an air gap, and possibly a coolant plate in between.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PackTopology:
    """A ``n_modules x cells_per_module`` grid of series-connected cells."""

    n_modules: int
    cells_per_module: int

    def __post_init__(self) -> None:
        if self.n_modules < 1 or self.cells_per_module < 1:
            raise ValueError("pack must have at least one module of at least one cell")

    @property
    def n_cells(self) -> int:
        return self.n_modules * self.cells_per_module

    def index(self, module: int, position: int) -> int:
        """Flat cell index from (module, position)."""
        if not (0 <= module < self.n_modules and 0 <= position < self.cells_per_module):
            raise IndexError(
                f"({module}, {position}) outside {self.n_modules}x{self.cells_per_module}"
            )
        return module * self.cells_per_module + position

    def coords(self, cell: int) -> tuple[int, int]:
        """(module, position) from a flat cell index."""
        if not 0 <= cell < self.n_cells:
            raise IndexError(f"cell {cell} outside pack of {self.n_cells}")
        return divmod(cell, self.cells_per_module)

    def grid_distance(self, cell_a: int, cell_b: int) -> int:
        """Manhattan distance on the (module, position) grid.

        This is a *descriptive* convenience for plots and sanity checks. It is
        deliberately **not** used to decide observability -- that decision is made
        by the Fisher information, which knows about conduction anisotropy,
        thermal mass, and noise, none of which a hop count knows about.
        """
        (ma, pa), (mb, pb) = self.coords(cell_a), self.coords(cell_b)
        return abs(ma - mb) + abs(pa - pb)

    def neighbour_pairs(self) -> list[tuple[int, int, bool]]:
        """Undirected edges as ``(i, j, is_intra_module)`` with ``i < j``."""
        edges: list[tuple[int, int, bool]] = []
        for m in range(self.n_modules):
            for p in range(self.cells_per_module):
                here = self.index(m, p)
                if p + 1 < self.cells_per_module:
                    edges.append((here, self.index(m, p + 1), True))
                if m + 1 < self.n_modules:
                    edges.append((here, self.index(m + 1, p), False))
        return edges

    def conductance_matrix(self, k_intra_w_per_k: float, k_inter_w_per_k: float) -> FloatArray:
        """Graph Laplacian L of the conduction network [W/K].

        ``(L @ T)[i]`` is the net heat conducted out of cell i. L is symmetric,
        positive semi-definite, and its null space is the constant vector -- an
        isothermal pack conducts no heat internally.
        """
        n = self.n_cells
        lap = np.zeros((n, n), dtype=float)
        for i, j, intra in self.neighbour_pairs():
            k = k_intra_w_per_k if intra else k_inter_w_per_k
            lap[i, i] += k
            lap[j, j] += k
            lap[i, j] -= k
            lap[j, i] -= k
        return lap
