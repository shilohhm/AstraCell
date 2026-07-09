"""One repeatable calibration scenario: a known truth, run many times.

A ``Scenario`` fixes everything a Monte Carlo needs to hold constant while only the noise draw
varies: the pack, the parameters being fitted, which of them carries the fault and how large it
is, the excitation, the sensor topology, the noise model, the plant's structural mismatch, and
whether the observer is allowed to use its own bias estimate to gate the verdict.

The truth is *known*, because we injected it. That is the entire methodological move of v0.2:
we cannot check calibration against real batteries (we have none), but we can check whether the
decisions are self-consistent -- whether an interval claimed to cover 90% of the time does, over
draws where we know the answer. What that proves, and what it cannot, is set out in
``docs/CALIBRATION.md``: it is a test of internal honesty, not of external truth.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from astracell.observability.sensitivity import (
    ParameterSpec,
    ParamKind,
    all_specs,
    local_specs,
    with_current_bias,
)
from astracell.pack.params import PackParams, nominal_pack
from astracell.pack.topology import PackTopology
from astracell.plant.mismatch import NO_MISMATCH, MismatchModel
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import SensorTopology, realistic_topology

FloatArray = NDArray[np.float64]

#: Nominal fault magnitudes the README hunts for, by kind. Signed and fractional: capacity
#: and cooling *fall*, resistance *rises*. ``0.0`` is always available as the null hypothesis.
NOMINAL_FAULT: dict[ParamKind, float] = {
    ParamKind.R0: 0.20,
    ParamKind.CAPACITY: -0.05,
    ParamKind.HA: -0.40,
}


@dataclass(frozen=True)
class Scenario:
    """A fully-specified, repeatable experiment with a known injected fault."""

    name: str
    params: PackParams
    specs: tuple[ParameterSpec, ...]
    target_index: int
    fault_magnitude: float
    current_a: FloatArray
    dt_s: float
    topology: SensorTopology
    noise: NoiseModel
    mismatch: MismatchModel = NO_MISMATCH
    use_bias_gate: bool = True
    soc0: float | FloatArray = 0.75
    temp0_k: float | FloatArray | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.target_index < len(self.specs):
            raise ValueError(f"target_index {self.target_index} out of range for {len(self.specs)}")
        if self.target_spec.is_global:
            raise ValueError("the fault target must be a per-cell physical parameter")
        if self.fault_magnitude <= -1.0:
            raise ValueError("fault_magnitude must exceed -1 (cannot scale a parameter below 0)")

    @property
    def target_spec(self) -> ParameterSpec:
        return self.specs[self.target_index]

    @property
    def is_null(self) -> bool:
        """No fault: the target parameter is at its healthy value."""
        return self.fault_magnitude == 0.0

    @property
    def delta_true(self) -> FloatArray:
        """The true parameter-deviation vector: the fault at the target, zero elsewhere."""
        truth = np.zeros(len(self.specs))
        truth[self.target_index] = self.fault_magnitude
        return truth

    @property
    def plant_params(self) -> PackParams:
        """The pack the plant actually runs: healthy nominal with the fault applied."""
        if self.is_null:
            return self.params
        spec = self.target_spec
        assert spec.cell is not None  # per-cell, enforced in __post_init__
        return self.params.scale_cell(spec.kind.value, spec.cell, 1.0 + self.fault_magnitude)

    def relabel(self, name: str) -> Scenario:
        return replace(self, name=name)

    def with_fault(self, magnitude: float) -> Scenario:
        return replace(self, fault_magnitude=magnitude, name=f"{self.name}@{magnitude:+.0%}")

    def matched(self) -> Scenario:
        """The same scenario with the plant collapsed onto the observer (no mismatch)."""
        return replace(self, mismatch=NO_MISMATCH, name=f"{self.name}/matched")


def build_scenario(
    *,
    name: str,
    fault_kind: ParamKind,
    fault_magnitude: float | None = None,
    target_cell: int,
    n_modules: int = 2,
    cells_per_module: int = 2,
    n_temp_sensors: int = 1,
    seed: int = 0,
    current_a: FloatArray | None = None,
    dt_s: float = 1.0,
    noise: NoiseModel | None = None,
    mismatch: MismatchModel = NO_MISMATCH,
    use_bias_gate: bool = True,
    include_current_bias: bool = False,
    use_all_specs: bool = False,
) -> Scenario:
    """Assemble a single-target calibration scenario on a small pack.

    Defaults to a 2x2 pack with one thermocouple -- small enough for thousands of Monte Carlo
    fits, large enough to carry a sensed and an unsensed cell. ``fault_magnitude=None`` uses the
    nominal magnitude for that fault kind; pass ``0.0`` for the null hypothesis.

    ``include_current_bias`` marginalises the shunt offset; ``use_all_specs`` fits every cell's
    parameters rather than just the target cell's. The two together under constant current are
    what make the pack genuinely confounded (VIF far above 10), which is how ``REFUSE_CONFOUNDED``
    becomes reachable -- the same mechanism README §5 describes.
    """
    pack = PackTopology(n_modules=n_modules, cells_per_module=cells_per_module)
    params = nominal_pack(pack, seed=seed)
    topology = realistic_topology(pack, n_temp_sensors=n_temp_sensors)
    specs = all_specs(pack.n_cells) if use_all_specs else local_specs(target_cell)
    if include_current_bias:
        specs = with_current_bias(specs)
    target_index = next(
        i for i, s in enumerate(specs) if s.kind is fault_kind and s.cell == target_cell
    )
    magnitude = NOMINAL_FAULT[fault_kind] if fault_magnitude is None else fault_magnitude

    if current_a is None:
        from astracell.duty import pulse_train

        duty = pulse_train(600.0, dt_s, mean_c_rate=0.2, pulse_c_rate=1.0)
        current_a = duty.current_a

    return Scenario(
        name=name,
        params=params,
        specs=specs,
        target_index=target_index,
        fault_magnitude=magnitude,
        current_a=np.asarray(current_a, dtype=float),
        dt_s=dt_s,
        topology=topology,
        noise=noise or NoiseModel(),
        mismatch=mismatch,
        use_bias_gate=use_bias_gate,
    )
