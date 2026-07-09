"""Open-circuit voltage curves.

HONESTY NOTE — read before using any number this module produces
----------------------------------------------------------------
Neither curve below is fitted to measured data for a specific commercial cell.

``NMC_LIKE``
    Uses the functional form and published coefficients of Chen & Rincon-Mora
    (2006), which were fitted to a lithium-polymer cell. Its *shape* is broadly
    graphite/NMC-like: a steep knee below ~10% SOC and a gentle slope above.
    We use it as a stand-in because it is smooth, analytically differentiable,
    and cited. It is not an NMC cell.

``LFP_LIKE``
    Hand-constructed. It reproduces the *qualitative* problem LFP poses for
    SOC-based diagnostics -- a plateau with dOCV/dSOC of order 0.5 mV per
    percent SOC -- but it is not quantitatively faithful to any real cell.

The entropic coefficients (dOCV/dT) are illustrative, order 0.1 mV/K, with a
SOC dependence that is plausible in sign and magnitude but not measured.

Replace both with tables exported from PyBaMM, or from measured pseudo-OCV
sweeps, before drawing any quantitative conclusion about a real battery. The
identifiability machinery in ``astracell.observability`` is only as trustworthy
as the sensitivities it differentiates, and those come from here.

Reference temperature for all curves is 298.15 K.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

T_REF_K: float = 298.15

# SOC is clipped into this interval before evaluation. Both curves have
# exponential terms that blow up outside [0, 1]; clipping keeps derivatives
# finite, but a simulation that hits these bounds is telling you the duty cycle
# is wrong, not that the model is fine. `astracell.pack.simulate` raises instead.
_SOC_LO: float = 1e-3
_SOC_HI: float = 1.0 - 1e-3

FloatArray = NDArray[np.float64]
Scalarish = float | FloatArray


@dataclass(frozen=True)
class OcvCurve:
    """An OCV curve with analytic first derivatives.

    All callables accept and return arrays broadcast over cells.
    """

    name: str
    chemistry: str
    _ocv_ref: Callable[[FloatArray], FloatArray]
    _docv_dsoc: Callable[[FloatArray], FloatArray]
    _docv_dtemp: Callable[[FloatArray], FloatArray]

    def ocv(self, soc: Scalarish, temp_k: Scalarish = T_REF_K) -> FloatArray:
        """Open-circuit voltage [V] at the given SOC and temperature."""
        z = self._clip(soc)
        return self._ocv_ref(z) + (np.asarray(temp_k, dtype=float) - T_REF_K) * self._docv_dtemp(z)

    def docv_dsoc(self, soc: Scalarish) -> FloatArray:
        """dOCV/dSOC [V per unit SOC]. Multiply by 0.01 for V per percent."""
        return self._docv_dsoc(self._clip(soc))

    def docv_dtemp(self, soc: Scalarish) -> FloatArray:
        """Entropic coefficient dOCV/dT [V/K]. Small, signed, SOC-dependent."""
        return self._docv_dtemp(self._clip(soc))

    @staticmethod
    def _clip(soc: Scalarish) -> FloatArray:
        return np.clip(np.asarray(soc, dtype=float), _SOC_LO, _SOC_HI)


# --------------------------------------------------------------------------
# NMC-like: Chen & Rincon-Mora (2006) functional form.
# --------------------------------------------------------------------------
def _nmc_ocv(z: FloatArray) -> FloatArray:
    return -1.031 * np.exp(-35.0 * z) + 3.685 + 0.2156 * z - 0.1178 * z**2 + 0.3201 * z**3


def _nmc_docv_dsoc(z: FloatArray) -> FloatArray:
    return 35.0 * 1.031 * np.exp(-35.0 * z) + 0.2156 - 0.2356 * z + 0.9603 * z**2


def _nmc_docv_dtemp(z: FloatArray) -> FloatArray:
    # Illustrative. Order 0.1 mV/K, negative across most of the SOC window,
    # which makes discharge reversibly exothermic. See thermal.reversible_heat.
    return 2.0e-4 * (-0.5 + 1.2 * z - 0.9 * z**2)


NMC_LIKE = OcvCurve(
    name="nmc_like_chen2006_form",
    chemistry="graphite/NMC-like (stand-in, not fitted)",
    _ocv_ref=_nmc_ocv,
    _docv_dsoc=_nmc_docv_dsoc,
    _docv_dtemp=_nmc_docv_dtemp,
)


# --------------------------------------------------------------------------
# LFP-like: hand-constructed flat plateau. Exists to make the hard case visible.
# --------------------------------------------------------------------------
def _lfp_ocv(z: FloatArray) -> FloatArray:
    return 3.30 + 0.055 * z + 0.10 * np.tanh(10.0 * (z - 0.93)) - 0.45 * np.exp(-45.0 * z)


def _lfp_docv_dsoc(z: FloatArray) -> FloatArray:
    sech2 = 1.0 - np.tanh(10.0 * (z - 0.93)) ** 2
    return 0.055 + 1.0 * sech2 + 20.25 * np.exp(-45.0 * z)


def _lfp_docv_dtemp(z: FloatArray) -> FloatArray:
    return np.full_like(z, -5.0e-5)


LFP_LIKE = OcvCurve(
    name="lfp_like_synthetic",
    chemistry="LFP-like (synthetic, qualitative plateau only)",
    _ocv_ref=_lfp_ocv,
    _docv_dsoc=_lfp_docv_dsoc,
    _docv_dtemp=_lfp_docv_dtemp,
)


CURVES: dict[str, OcvCurve] = {c.name: c for c in (NMC_LIKE, LFP_LIKE)}


def ocv_from_table(
    soc: FloatArray,
    ocv: FloatArray,
    *,
    name: str,
    chemistry: str,
    docv_dtemp: float = 0.0,
) -> OcvCurve:
    """Build an ``OcvCurve`` from a sampled OCV(SOC) table.

    This is the extension point the module docstring promises: a pseudo-OCV swept slowly from an
    external cell model (PyBaMM) or a bench, turned into the analytic-ish curve the sensitivity
    machinery needs. The table must be sorted by ``soc`` and cover the operating window.

    The derivative ``dOCV/dSOC`` is taken from the table by central differences (``np.gradient``)
    and interpolated, rather than differentiating an interpolating spline: the sensitivities that
    consume it are already first-order finite differences, so a matched, monotone,
    non-oscillatory derivative is worth more than a formally smoother one that rings between knots.

    ``docv_dtemp`` is a single entropic coefficient [V/K], defaulting to zero -- an external
    voltage-only sweep does not measure it, and pretending otherwise would be exactly the kind of
    invented number this repository exists to avoid. Supply one only if it was measured.
    """
    soc = np.asarray(soc, dtype=float)
    ocv = np.asarray(ocv, dtype=float)
    if soc.ndim != 1 or soc.shape != ocv.shape:
        raise ValueError(
            f"soc and ocv must be matching 1-D arrays, got {soc.shape} and {ocv.shape}"
        )
    if soc.size < 4:
        raise ValueError("need at least 4 points to build an OCV table")
    order = np.argsort(soc)
    soc, ocv = soc[order], ocv[order]
    if np.any(np.diff(soc) <= 0.0):
        raise ValueError("soc values must be strictly increasing after sorting (no duplicates)")

    slope = np.gradient(ocv, soc)

    def _ocv_ref(z: FloatArray) -> FloatArray:
        return np.interp(z, soc, ocv)

    def _docv_dsoc(z: FloatArray) -> FloatArray:
        return np.interp(z, soc, slope)

    def _docv_dtemp(z: FloatArray) -> FloatArray:
        return np.full_like(z, docv_dtemp)

    return OcvCurve(
        name=name,
        chemistry=chemistry,
        _ocv_ref=_ocv_ref,
        _docv_dsoc=_docv_dsoc,
        _docv_dtemp=_docv_dtemp,
    )
