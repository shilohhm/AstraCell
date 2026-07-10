"""The plant: a battery richer than the model AstraCell fits to it.

Used only to *measure the cost of being wrong*. Nothing in the observability layer is
allowed to fit against this, or the exercise would be circular.
"""

from astracell.plant.mismatch import NO_MISMATCH, REALISTIC_MISMATCH, MismatchModel
from astracell.plant.pybamm_plant import (
    HEALTHY,
    PYBAMM_AVAILABLE,
    PyBaMMFault,
    PyBaMMPlant,
    PyBaMMUnavailableError,
    contact_resistance,
    pybamm_pseudo_ocv,
    require_pybamm,
    simulate_pybamm_cell,
    slow_cathode,
)
from astracell.plant.simulate import PlantResult, PlantStabilityError, simulate_plant

__all__ = [
    "HEALTHY",
    "NO_MISMATCH",
    "PYBAMM_AVAILABLE",
    "REALISTIC_MISMATCH",
    "MismatchModel",
    "PlantResult",
    "PlantStabilityError",
    "PyBaMMFault",
    "PyBaMMPlant",
    "PyBaMMUnavailableError",
    "contact_resistance",
    "pybamm_pseudo_ocv",
    "require_pybamm",
    "simulate_pybamm_cell",
    "slow_cathode",
    "simulate_plant",
]
