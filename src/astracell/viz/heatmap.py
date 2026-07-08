"""The detectability heatmap.

Axes: current excitation (what the duty cycle gave you) against fault magnitude
(what you are trying to see). Colour: the best SNR any unbiased estimator could
achieve. The 5-sigma contour is the boundary of the region where a diagnosis is
defensible. Everything left of it is where AstraCell abstains.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from astracell.observability.detectability import HeatmapResult
from astracell.viz._figure import figure_of


def plot_detectability_heatmap(
    result: HeatmapResult,
    *,
    ax: Axes | None = None,
    contours: tuple[float, ...] = (2.0, 5.0, 20.0),
    title: str | None = None,
) -> Figure:
    """Filled contour of detection SNR over (excitation, magnitude)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5.0))
    fig = figure_of(ax)

    x = result.excitation_c_rate
    y = 100.0 * result.magnitude
    snr = np.clip(result.snr, 1e-3, None)

    mesh = ax.pcolormesh(
        x,
        y,
        snr,
        cmap="viridis",
        norm=LogNorm(vmin=max(snr.min(), 1e-2), vmax=snr.max()),
        shading="gouraud",
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("detection SNR (sigma)  -  upper bound over all estimators")

    lines = ax.contour(
        x, y, snr, levels=list(contours), colors="white", linewidths=1.4, linestyles="--"
    )
    ax.clabel(lines, fmt=lambda v: f"{v:.0f} sigma", fontsize=8, inline=True)

    # Shade the abstention region: below 2 sigma.
    ax.contourf(x, y, snr, levels=[0.0, 2.0], colors=["#9e9e9e"], alpha=0.55)

    # Magnitudes are geometrically spaced -- a Cramer-Rao floor of 0.1% and a
    # cooling fault of 40% have to share an axis.
    ax.set_yscale("log")
    ax.set_ylim(y.min(), y.max())
    ax.set_xlabel("current excitation (pulse amplitude, C-rate)")
    ax.set_ylabel(f"{result.kind.value} fault magnitude (%)")
    ax.set_title(
        title
        or f"Detectability of {result.kind.value} on cell {result.cell}\n"
        "grey = AstraCell abstains (< 2 sigma); dashed = iso-SNR contours",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_min_detectable(
    result: HeatmapResult, *, sigma: float = 5.0, ax: Axes | None = None
) -> Figure:
    """The single most useful line in the package: smallest visible fault vs excitation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.2))
    fig = figure_of(ax)

    ax.plot(
        result.excitation_c_rate,
        100.0 * result.min_detectable_magnitude(sigma),
        marker="o",
        color="#1565c0",
    )
    ax.set_xlabel("current excitation (pulse amplitude, C-rate)")
    ax.set_ylabel(f"smallest {result.kind.value} fault visible at {sigma:.0f} sigma (%)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"Cramer-Rao floor for cell {result.cell}: what excitation buys you", fontsize=11)
    fig.tight_layout()
    return fig
