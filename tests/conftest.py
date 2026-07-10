"""Make the ``pybamm`` marker mean what ``pyproject.toml`` says it means.

The marker is declared as "requires the optional PyBaMM dependency; skipped if it is not
installed". Until v0.4 nothing enforced the second half. Every PyBaMM test happened to request a
fixture that called ``pytest.importorskip``, so the suite passed without PyBaMM by coincidence, and
the first marked test written without such a fixture failed with ``PyBaMMUnavailableError`` on a
clean install rather than skipping.

Enforcing it here means the marker alone is sufficient, the fixtures' ``importorskip`` calls are
belt-and-braces, and a test cannot silently opt out of the optional-dependency contract by not
needing a fixture.
"""

from __future__ import annotations

import importlib.util

import pytest

PYBAMM_AVAILABLE = importlib.util.find_spec("pybamm") is not None

_SKIP_PYBAMM = pytest.mark.skip(
    reason="requires the optional PyBaMM dependency: pip install -e '.[pybamm]'"
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip every ``@pytest.mark.pybamm`` test when PyBaMM is absent."""
    if PYBAMM_AVAILABLE:
        return
    for item in items:
        if "pybamm" in item.keywords:
            item.add_marker(_SKIP_PYBAMM)
