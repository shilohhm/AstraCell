"""Narrowing helper for matplotlib's ``Axes.get_figure()`` union type.

``get_figure()`` returns ``Figure | SubFigure``. Our plotting functions promise a
``Figure``, and a ``SubFigure`` has no ``tight_layout``. Rather than sprinkle
``cast`` around, we narrow once, here, and raise if an Axes really does live in a
SubFigure -- in which case the caller should manage the layout themselves.
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def figure_of(ax: Axes) -> Figure:
    """The top-level Figure owning ``ax``."""
    fig = ax.get_figure()
    if not isinstance(fig, Figure):
        raise TypeError(
            "AstraCell plotting functions expect an Axes on a top-level Figure, "
            f"got {type(fig).__name__}. Build the layout yourself and pass ax=."
        )
    return fig
