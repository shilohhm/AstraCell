"""Current profiles. The excitation is half of the identifiability answer."""

from astracell.duty.profiles import (
    DutyProfile,
    constant_current,
    pulse_train,
    random_walk,
    rest_then_pulse,
)

__all__ = [
    "DutyProfile",
    "constant_current",
    "pulse_train",
    "random_walk",
    "rest_then_pulse",
]
