"""The plant: a battery richer than the model AstraCell fits to it.

Used only to *measure the cost of being wrong*. Nothing in the observability layer is
allowed to fit against this, or the exercise would be circular.
"""

from astracell.plant.mismatch import NO_MISMATCH, REALISTIC_MISMATCH, MismatchModel
from astracell.plant.pybamm_plant import (
    PYBAMM_AVAILABLE,
    PyBaMMPlant,
    PyBaMMUnavailableError,
    pybamm_pseudo_ocv,
    require_pybamm,
    simulate_pybamm_cell,
)
from astracell.plant.simulate import PlantResult, PlantStabilityError, simulate_plant

__all__ = [
    "NO_MISMATCH",
    "PYBAMM_AVAILABLE",
    "REALISTIC_MISMATCH",
    "MismatchModel",
    "PlantResult",
    "PlantStabilityError",
    "PyBaMMPlant",
    "PyBaMMUnavailableError",
    "pybamm_pseudo_ocv",
    "require_pybamm",
    "simulate_pybamm_cell",
    "simulate_plant",
]
