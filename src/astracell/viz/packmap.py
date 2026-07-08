"""The pack map.

Green, amber, grey. The grey cells are the reason this package exists: they are
the cells about which the data, given this sensor topology and this duty cycle,
cannot support a claim. They are not painted by a distance-to-sensor rule -- they
come from the Cramer-Rao bound.

Temperature sensors are drawn on the map. A reader who sees the grey regions
cluster away from the thermocouples has understood the result without being told.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from astracell.observability.mask import GreyCellMap, Observability
from astracell.pack.topology import PackTopology
from astracell.viz._figure import figure_of


def plot_pack_map(
    grey: GreyCellMap,
    pack: PackTopology,
    *,
    ax: Axes | None = None,
    annotate: bool = True,
    fault_cell: int | None = None,
    title: str | None = None,
) -> Figure:
    """Render per-cell identifiability as a module x position grid."""
    if ax is None:
        width = max(6.0, 0.9 * pack.cells_per_module)
        height = max(2.5, 0.9 * pack.n_modules + 1.6)
        _, ax = plt.subplots(figsize=(width, height))
    fig = figure_of(ax)

    for cell in range(pack.n_cells):
        module, position = pack.coords(cell)
        level = Observability(int(grey.level[cell]))
        is_sensed = cell in grey.topology.temp_cells

        ax.add_patch(
            Rectangle(
                (position, module),
                1.0,
                1.0,
                facecolor=level.colour,
                edgecolor="white",
                linewidth=1.5,
                alpha=0.30 if level is Observability.UNOBSERVABLE else 0.85,
                hatch="///" if level is Observability.UNOBSERVABLE else None,
            )
        )
        if annotate:
            snr = grey.snr[cell]
            text = f"{snr:.1f}" if snr >= 0.05 else "~0"
            ax.text(
                position + 0.5,
                module + 0.62,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                color="#212121" if level is Observability.UNOBSERVABLE else "white",
                fontweight="bold",
            )
        if is_sensed:
            ax.plot(
                position + 0.5, module + 0.24, marker="o", markersize=9, color="white", zorder=3
            )
            ax.text(
                position + 0.5,
                module + 0.24,
                "T",
                ha="center",
                va="center",
                fontsize=6.5,
                color="#111111",
                fontweight="bold",
                zorder=4,
            )
        if fault_cell is not None and cell == fault_cell:
            ax.add_patch(
                Rectangle(
                    (position, module),
                    1.0,
                    1.0,
                    facecolor="none",
                    edgecolor="#d32f2f",
                    linewidth=3.0,
                    zorder=5,
                )
            )

    ax.set_xlim(0, pack.cells_per_module)
    ax.set_ylim(pack.n_modules, 0)
    ax.set_xticks(
        np.arange(pack.cells_per_module) + 0.5, [str(i) for i in range(pack.cells_per_module)]
    )
    ax.set_yticks(np.arange(pack.n_modules) + 0.5, [f"M{i}" for i in range(pack.n_modules)])
    ax.set_xlabel("cell position within module")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    default_title = (
        f"Identifiability of {grey.kind.value} at {100 * grey.magnitude:.0f}%  |  "
        f"{grey.topology.n_temp} temperature sensors"
    )
    ax.set_title(title or default_title, fontsize=11, pad=12)

    handles: list[Any] = [
        Patch(
            facecolor=Observability.OBSERVABLE.colour, alpha=0.85, label="observable (>= 5 sigma)"
        ),
        Patch(facecolor=Observability.WEAK.colour, alpha=0.85, label="weak (2-5 sigma)"),
        Patch(
            facecolor=Observability.UNOBSERVABLE.colour,
            alpha=0.30,
            hatch="///",
            label="unobservable (< 2 sigma) - AstraCell abstains",
        ),
    ]
    if fault_cell is not None:
        handles.append(
            Patch(facecolor="none", edgecolor="#d32f2f", linewidth=2, label="injected fault")
        )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout()
    return fig
