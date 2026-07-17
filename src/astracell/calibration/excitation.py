"""Plan the excitation, not just the estimate: score a current profile before commanding it.

``examples/03`` ranks the tests a BMS could run for a *thermal* fault, by how much each would
sharpen the hypothesis. v0.9 raises the identical question for a different fault -- the fast RC
branch ``(R1, C1)`` the 4-parameter paired fit now estimates -- and this module is the one
primitive that answers it: the VIF of each fitted parameter under a candidate excitation.

The whole point is that VIF is *data-independent*. It is read off the Fisher information at the
healthy nominal, so it depends only on the observer, the current profile, and the fitted specs --
never on a measured voltage. That is what lets it plan: you can score a pulse train the cell has
never seen and know whether it would make ``R1`` identifiable, before you spend a second commanding
it. See ``docs/REAL_CELL.md`` (Fit the dynamics) and ``WHAT_DID_NOT_WORK.md`` 6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astracell.calibration.external import external_scenario
from astracell.observability.estimator import build_context
from astracell.observability.sensitivity import ParameterSpec
from astracell.pack.params import PackParams

FloatArray = NDArray[np.float64]


def vif_under_excitation(
    observer: PackParams,
    current_a: FloatArray,
    *,
    soc0: float,
    specs: tuple[ParameterSpec, ...],
    dt_s: float = 1.0,
) -> FloatArray:
    """Variance-inflation factor of each fitted parameter under a candidate excitation.

    VIF is the separability gate: how much a parameter's variance is inflated by its collinearity
    with the others under this excitation. It comes straight off the FIM at the healthy nominal
    (:func:`~astracell.observability.estimator.build_context`), so it depends only on the observer,
    the current profile, and the fitted ``specs`` -- *not* on any measured voltage. That
    data-independence is what makes it a planning tool: score a current profile a BMS could command
    before commanding it, exactly as ``examples/03`` ranks tests for a thermal fault. Returns one
    VIF per spec, in spec order.

    v0.9's use: the fast RC branch is identifiable only under rich excitation. Under a 1C discharge
    ``dV/dR1`` collapses onto ``dV/dR0`` and ``VIF(R1) >> 10``; a pulse train re-opens the transient
    and drops it below 10. Capacity's VIF, by contrast, barely responds to the excitation -- because
    capacity's refusal on a real cell is model bias (moving OCV), not collinearity, and no
    excitation removes bias, it only routes it (``examples/04``, ``WHAT_DID_NOT_WORK.md`` 6). So a
    pulse earns the *dynamics*, never the *capacity* verdict.
    """
    scenario = external_scenario(
        name="excitation_plan",
        observer=observer,
        current_a=np.asarray(current_a, dtype=float),
        soc0=soc0,
        dt_s=dt_s,
        specs=specs,
    )
    ctx = build_context(
        scenario.params,
        scenario.specs,
        scenario.current_a,
        scenario.dt_s,
        scenario.topology,
        scenario.noise,
        soc0=scenario.soc0,
        temp0_k=scenario.temp0_k,
    )
    return np.asarray(ctx.vif, dtype=float)
