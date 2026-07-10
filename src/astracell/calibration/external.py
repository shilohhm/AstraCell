"""Fit AstraCell's ECM observer against a plant it did not write, and reuse the v0.2 machinery.

The design constraint here is *do not build a second estimator*. v0.2 already has one, and a whole
apparatus -- ``build_context``, ``run_trials``, coverage, abstention -- built on the contract that
a scenario yields a ``ScenarioContext`` with a ``plant_output`` tensor and a ``structural_bias``.
This module supplies exactly that contract from an external voltage trace instead of from
``simulate_plant``. Everything downstream is unchanged; the only new thing is where the numbers
the observer is fitted against come from.

Two deliberate choices make the comparison fair rather than rigged:

* **Voltage only, one cell, and the observer shares the plant's OCV.** The specs fitted are
  ``(R0, capacity)`` -- ``hA`` is dropped because an isothermal voltage-only cell cannot see it,
  and pretending to fit it would inject a spurious singular direction. With the static OCV shared
  (see ``pybamm_pseudo_ocv``), the residual under load is the *dynamics* the first-order ECM
  cannot express, not a curve-shape disagreement we could have removed.

* **The truth is a healthy cell.** ``fault_magnitude`` defaults to ``0.0``: the external cell is
  not degraded, so the honest capacity deviation is zero. Any nonzero estimate the observer
  reports is the model-mismatch bias masquerading as a fault -- which is the entire point. The
  variance-only interval that excludes zero is diagnosing a fault that is not there.

This module has no PyBaMM dependency: it takes a voltage array. That is what lets the
self-consistency control -- an ECM-generated trace fed through this same path -- prove the harness
adds no bias of its own before PyBaMM is ever involved. See ``docs/EXTERNAL_PLANT.md``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astracell.calibration.montecarlo import ScenarioContext
from astracell.calibration.scenario import Scenario
from astracell.cell.ocv import OcvCurve
from astracell.observability.bias import parameter_bias
from astracell.observability.estimator import build_context, output_tensor
from astracell.observability.sensitivity import ParameterSpec, ParamKind
from astracell.pack.params import PackParams
from astracell.pack.simulate import simulate
from astracell.pack.topology import PackTopology
from astracell.sensors.noise import NoiseModel
from astracell.sensors.topology import SensorTopology

FloatArray = NDArray[np.float64]

# A plausible first-order ECM for a ~5 Ah cell. R0 and capacity are *fitted*, so their nominals
# only set the linearisation point; R1/C1 are the fixed RC branch -- the one relaxation timescale
# the observer is allowed to represent. The capacity bias the observer incurs against PyBaMM is
# NOT robust to this choice: ``docs/EXTERNAL_PLANT.md`` measures it running from -94% (tau 8 s) to
# +5% (tau 300 s) as the branch is retuned toward the electrolyte diffusion timescale. What does
# not depend on the tuning is that the bias always exceeds the CRLB sigma (~0.15%) by more than
# 30x, so the variance-only interval overclaims at every tuning. The robust result is the
# overclaim and the refusal, never the magnitude of the phantom fault.
DEFAULT_OBSERVER_R0_OHM: float = 0.025
DEFAULT_OBSERVER_R1_OHM: float = 0.010
DEFAULT_OBSERVER_C1_FARAD: float = 3000.0

_VOLTAGE_ONLY = SensorTopology(n_cells=1, voltage_cells=(0,), temp_cells=())


def constant_profile(
    capacity_ah: float, c_rate: float, n_time: int, dt_s: float = 1.0
) -> FloatArray:
    """A constant discharge current [A] at ``c_rate``, sized to the cell's capacity."""
    return np.full(n_time, c_rate * capacity_ah, dtype=float)


def pulse_profile(
    capacity_ah: float,
    *,
    mean_c_rate: float,
    pulse_c_rate: float,
    n_time: int,
    dt_s: float = 1.0,
    period_s: float = 60.0,
    duty: float = 0.25,
) -> FloatArray:
    """A repeating discharge pulse train [A] with a preserved mean, sized to the capacity.

    The square wave is zero-mean by construction (``+ (1 - duty)`` on, ``- duty`` off), so the
    net current is exactly ``mean_c_rate``. The pulses are what keep the electrolyte gradient in
    perpetual transient -- the regime a single RC branch cannot track, and the one that turns a
    healthy cell into a confident false capacity fault.
    """
    t = np.arange(n_time, dtype=float) * dt_s
    on = (np.mod(t, period_s) / period_s) < duty
    square = np.where(on, 1.0 - duty, -duty)
    return (mean_c_rate + pulse_c_rate * square) * capacity_ah


def build_external_observer(
    curve: OcvCurve,
    capacity_ah: float,
    *,
    r0_ohm: float = DEFAULT_OBSERVER_R0_OHM,
    r1_ohm: float = DEFAULT_OBSERVER_R1_OHM,
    c1_farad: float = DEFAULT_OBSERVER_C1_FARAD,
) -> PackParams:
    """A one-cell ECM observer with the plant's OCV and ``ea_over_r_k=0`` (isothermal voltage).

    Setting ``ea_over_r_k=0`` makes terminal voltage independent of the observer's own thermal
    state, matching PyBaMM's default isothermal cell and removing temperature as a confounder --
    which is also why ``hA`` is not among the fitted specs: it has no path to the voltage.
    """
    one = lambda value: np.array([value], dtype=float)  # noqa: E731
    return PackParams(
        topology=PackTopology(n_modules=1, cells_per_module=1),
        curve=curve,
        capacity_ah=one(capacity_ah),
        r0_ohm=one(r0_ohm),
        r1_ohm=one(r1_ohm),
        c1_farad=one(c1_farad),
        heat_capacity_j_per_k=one(900.0),
        ha_w_per_k=one(2.5),
        ea_over_r_k=0.0,
    )


def external_specs() -> tuple[ParameterSpec, ...]:
    """The fitted parameters: series resistance, then capacity (the target, at index 1)."""
    return (ParameterSpec(0, ParamKind.R0), ParameterSpec(0, ParamKind.CAPACITY))


#: Indices into ``external_specs()``. The fault target is one of these two.
R0_TARGET: int = 0
CAPACITY_TARGET: int = 1


def external_scenario(
    *,
    name: str,
    observer: PackParams,
    current_a: FloatArray,
    soc0: float,
    dt_s: float = 1.0,
    noise: NoiseModel | None = None,
    use_bias_gate: bool = True,
    fault_magnitude: float = 0.0,
    target_index: int = CAPACITY_TARGET,
    temp0_k: float = 298.15,
) -> Scenario:
    """A one-cell, voltage-only scenario whose plant is supplied externally.

    ``fault_magnitude`` is the *true* deviation of the targeted parameter. It defaults to ``0.0``
    -- v0.3's external cell is healthy -- so coverage asks whether the interval contains zero, and
    a harmful overclaim is the observer diagnosing a fault on a sound cell. v0.4 injects a real
    degradation and sets it accordingly; see ``calibration.positive_control``.

    ``target_index`` selects which of ``external_specs()`` carries the hypothesis: ``R0_TARGET``
    for a series-resistance fault, ``CAPACITY_TARGET`` (the default, and v0.3's) for capacity.
    """
    specs = external_specs()
    return Scenario(
        name=name,
        params=observer,
        specs=specs,
        target_index=target_index,
        fault_magnitude=fault_magnitude,
        current_a=np.asarray(current_a, dtype=float),
        dt_s=dt_s,
        topology=_VOLTAGE_ONLY,
        noise=noise or NoiseModel(),
        use_bias_gate=use_bias_gate,
        soc0=soc0,
        temp0_k=temp0_k,
    )


def observer_voltage(scenario: Scenario) -> FloatArray:
    """The observer's own terminal voltage on the scenario's excitation -- the nominal fit point."""
    result = simulate(
        scenario.params,
        scenario.current_a,
        scenario.dt_s,
        soc0=scenario.soc0,
        temp0_k=scenario.temp0_k,
    )
    return np.asarray(result.voltage_v[:, 0], dtype=float)


def prepare_external(
    scenario: Scenario,
    plant_voltage: FloatArray,
    *,
    baseline_voltage: FloatArray | None = None,
) -> ScenarioContext:
    """Build the v0.2 ``ScenarioContext`` from an external voltage trace.

    ``plant_voltage`` is what some richer plant produced on ``scenario.current_a`` (PyBaMM, or the
    ECM itself for the self-consistency control). The structural bias is the omitted-dynamics
    residual projected onto the fitted parameters -- exactly ``observability.bias.parameter_bias``,
    the same estimator v0.2 uses, with the residual now coming from the external plant rather than
    from a hand-built mismatch model.

    **The bias must be measured on a healthy plant, and by default it is not.**

    ``parameter_bias`` of a residual is, by construction, the linear fit to that residual. Omit
    ``baseline_voltage`` and this function projects the *same* trace the estimator will be fitted
    to, so ``structural_bias == E[delta_hat]`` exactly and the credibility gate's

        SNR_total = |delta_hat| / sqrt(sigma^2 + b^2)  ==  1     for every dataset,

    yielding ``REFUSE_MODEL_BIAS`` unconditionally -- for a phantom fault, a real fault, or a
    residual made of sine waves. v0.3 used it this way, correctly, because it only ever ran a
    healthy cell, whose entire estimate *is* bias. It cannot be used that way on a plant that
    might be degraded: it would call the degradation "bias" and refuse to see it. This is the
    trap ``docs/POSITIVE_CONTROL.md`` §2 exists to document, and
    ``tests/test_positive_control.py::test_the_v03_external_gate_refuses_unconditionally`` pins it.

    Pass ``baseline_voltage`` -- a trace of the *healthy* plant on the same excitation -- and the
    bias is projected from that instead, restoring v0.2's convention that the residual is
    evaluated at the healthy nominal while the fault lives in ``plant_output``. The gate then
    discriminates rather than refuses.
    """
    plant_voltage = np.asarray(plant_voltage, dtype=float)
    n_time = scenario.current_a.size
    if plant_voltage.shape != (n_time,):
        raise ValueError(f"plant_voltage must be shape ({n_time},), got {plant_voltage.shape}")
    if baseline_voltage is not None:
        baseline_voltage = np.asarray(baseline_voltage, dtype=float)
        if baseline_voltage.shape != (n_time,):
            raise ValueError(
                f"baseline_voltage must be shape ({n_time},), got {baseline_voltage.shape}"
            )

    fit_ctx = build_context(
        scenario.params,
        scenario.specs,
        scenario.current_a,
        scenario.dt_s,
        scenario.topology,
        scenario.noise,
        soc0=scenario.soc0,
        temp0_k=scenario.temp0_k,
    )

    nominal = simulate(
        scenario.params,
        scenario.current_a,
        scenario.dt_s,
        soc0=scenario.soc0,
        temp0_k=scenario.temp0_k,
    )
    # Residual on the instrumented voltage channel; the temperature channel is uninstrumented and
    # never read, so it stays at the nominal (zero residual there).
    plant_output = output_tensor(plant_voltage[:, None], nominal.temp_k)
    nominal_output = output_tensor(nominal.voltage_v, nominal.temp_k)
    bias_source = plant_voltage if baseline_voltage is None else baseline_voltage
    residual = output_tensor(bias_source[:, None], nominal.temp_k) - nominal_output

    bias = parameter_bias(fit_ctx.sens, residual, scenario.topology, scenario.noise, scenario.specs)
    return ScenarioContext(
        scenario=scenario,
        fit_ctx=fit_ctx,
        plant_output=plant_output,
        structural_bias=bias,
        delta_true=scenario.delta_true,
    )
