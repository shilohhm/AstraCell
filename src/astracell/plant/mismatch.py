"""What the plant has that the observer does not.

Everything else in AstraCell computes the Cramer-Rao bound, which answers: *given that
my model is right, how well can any unbiased estimator pin down its parameters?* The
answer has been flattering. It is also conditional on a premise that is never true.

A real observer runs a coarser model than the plant. When it fits, it does not converge
on the true parameter ``theta*``. It converges on the **pseudo-true** parameter

    theta_0 = argmin_theta || g(theta*, u) - f(theta, u) ||^2_{Sigma^-1}

where ``g`` is the plant and ``f`` the observer's ECM. The gap ``b = theta_0 - theta*`` is
**structural bias**: it does not shrink with more samples, more sensors, or harder
excitation. See ``astracell.observability.bias``.

This module defines the extra structure. Four knobs, every one defaulting to zero, and
with all four at zero ``simulate_plant`` reduces to ``pack.simulate`` **bit for bit**.
That exact reduction is the whole reason this is a custom plant and not PyBaMM: a
mismatch experiment you cannot switch off is a mismatch experiment you cannot falsify.
It is asserted in ``tests/test_mismatch.py::test_zero_mismatch_reproduces_the_observer_exactly``.

What "true parameter" even means
--------------------------------

Under mismatch, ``theta*`` is a *convention*, because the plant has no parameter called
"the cell's R0" -- it has a function ``R0(SOC, T)``. Every knob here is therefore defined
to **vanish at the initial operating point** ``(soc0, coolant temperature)``:

* the SOC dependence of ``R0`` is centred on ``soc0``, so the factor is exactly 1 at t=0;
* the second RC branch starts empty, so its voltage is exactly 0 at t=0;
* core, surface and sensor temperatures all start equal.

So at ``t = 0`` the plant and the observer agree exactly, ``theta*`` is unambiguous, and
the residual is exactly zero. Mismatch then *accumulates* as the experiment runs. This is
a choice, it is stated, and it is tested
(``test_the_structural_residual_is_exactly_zero_at_the_initial_sample``).

The four knobs
--------------

``r0_soc_slope``
    Real ohmic resistance depends on state of charge; the ECM's does not. Under pulsed
    excitation this residual is pulse-shaped -- the same shape as the ``R0`` sensitivity --
    so it projects almost entirely onto ``R0``.

``second_rc_r_ohm`` / ``second_rc_tau_s``
    Real cells have a slow diffusion (Warburg-ish) branch, tens to hundreds of seconds.
    The ECM has one RC pair at ~2 s and cannot represent it. A slowly growing voltage drop
    looks exactly like SOC falling faster than it should, which is to say: **like a smaller
    capacity.** This is the classic contamination of short-window capacity estimation by
    slow polarisation, and it is not a modelling curiosity, it is why BMS capacity
    estimates drift.

``core_surface_resistance_k_per_w`` / ``core_heat_fraction``
    The cell is not isothermal. Heat is generated in the core; the thermocouple is glued to
    the surface. The ECM has one lumped node, so it reads a surface temperature and infers
    a core resistance from it.

``temp_sensor_tau_s``
    An NTC bonded to a cell has its own thermal mass. The measured temperature lags the
    surface. The observer, having no lag in its model, must explain the delay with the only
    thermal parameter it owns: ``hA``.

Magnitudes are order-of-magnitude representative of a mid-size pouch cell. They are
**chosen, not fitted**, exactly like the OCV curves in ``cell.ocv``. See ``LIMITATIONS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["MismatchModel", "NO_MISMATCH", "REALISTIC_MISMATCH"]

# Knobs that scale the *strength* of the mismatch, i.e. those that must reach zero for the
# plant to collapse onto the observer. The others set the shape of the extra dynamics and
# are meaningless once their partner strength is zero.
_STRENGTH_FIELDS = (
    "r0_soc_slope",
    "second_rc_r_ohm",
    "core_surface_resistance_k_per_w",
    "temp_sensor_tau_s",
)


@dataclass(frozen=True)
class MismatchModel:
    """Structure the plant has and the observer lacks. All-zero == the observer's ECM."""

    r0_soc_slope: float = 0.0
    """dR0/dSOC as a fraction of R0 per unit SOC, centred on ``soc0``."""

    second_rc_r_ohm: float = 0.0
    """Resistance of a slow polarisation branch the ECM does not model."""

    second_rc_tau_s: float = 220.0
    """Its time constant. Shape, not strength: irrelevant when the resistance is zero."""

    core_surface_resistance_k_per_w: float = 0.0
    """Core-to-surface thermal resistance. Zero means an isothermal (lumped) cell."""

    core_heat_fraction: float = 0.7
    """Fraction of the cell's heat capacity in the core node. Shape, not strength."""

    temp_sensor_tau_s: float = 0.0
    """First-order lag of the temperature sensor. Zero means instantaneous."""

    def __post_init__(self) -> None:
        if self.second_rc_r_ohm < 0.0:
            raise ValueError("second_rc_r_ohm must be non-negative")
        if self.second_rc_tau_s <= 0.0:
            raise ValueError("second_rc_tau_s must be positive")
        if self.core_surface_resistance_k_per_w < 0.0:
            raise ValueError("core_surface_resistance_k_per_w must be non-negative")
        if not 0.0 < self.core_heat_fraction < 1.0:
            raise ValueError("core_heat_fraction must lie strictly in (0, 1)")
        if self.temp_sensor_tau_s < 0.0:
            raise ValueError("temp_sensor_tau_s must be non-negative")

    @property
    def is_exact(self) -> bool:
        """True when the plant *is* the observer's model, so the residual must vanish."""
        return all(getattr(self, name) == 0.0 for name in _STRENGTH_FIELDS)

    def scaled(self, factor: float) -> MismatchModel:
        """Scale every strength knob, leaving the shape knobs alone.

        ``scaled(0.0).is_exact`` is True, which makes ``factor`` a clean dial from "the
        observer is right" to "the observer is as wrong as we think it really is". Bias is
        first-order in this factor for small values, and that is a test, not a hope.
        """
        if factor < 0.0:
            raise ValueError("factor must be non-negative")
        return replace(self, **{name: getattr(self, name) * factor for name in _STRENGTH_FIELDS})


NO_MISMATCH = MismatchModel()
"""The observer is right. Used to prove the harness measures mismatch and not its own bugs."""

REALISTIC_MISMATCH = MismatchModel(
    r0_soc_slope=0.35,  # R0 rises ~3.5% per 10% SOC drop
    second_rc_r_ohm=4.0e-4,  # 0.4 mOhm, vs R0 = 1.5 mOhm
    second_rc_tau_s=220.0,  # slow diffusion branch
    core_surface_resistance_k_per_w=0.40,  # k_cs = 2.5 W/K
    core_heat_fraction=0.7,
    temp_sensor_tau_s=15.0,  # NTC bonded to the can
)
"""Order-of-magnitude representative of a mid-size pouch cell. Chosen, not fitted."""
