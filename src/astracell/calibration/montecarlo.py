"""Repeat a scenario under many noise draws and record what the observer decided.

This is the machine that turns a single convincing example into a frequency statement. For a
fixed, known truth it draws ``n_trials`` independent measurement realisations, fits the fault
each time, and records the estimate and the verdict. Everything downstream -- coverage,
abstention rates, the money plot -- is aggregation over the arrays this returns.

Determinism is a hard requirement: the whole apparatus exists to make honest claims, and a
claim you cannot reproduce is not one. A single seeded ``Generator`` drives every draw, so the
same ``(scenario, n_trials, seed)`` yields bit-identical arrays. ``tests/test_calibration.py``
pins that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astracell.calibration.scenario import Scenario
from astracell.observability.bias import parameter_bias, structural_residual
from astracell.observability.decision import VerdictKind
from astracell.observability.estimator import (
    DEFAULT_MAX_ITER,
    DEFAULT_MAX_STEP,
    DEFAULT_STEP_TOL,
    FitContext,
    build_context,
    fit_gauss_newton,
    fit_linear,
    output_tensor,
    verdict_from_estimate,
)
from astracell.plant.simulate import simulate_plant
from astracell.sensors.sampling import sample_measurement_noise

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ScenarioContext:
    """The noise-independent derived objects for one scenario, built once."""

    scenario: Scenario
    fit_ctx: FitContext
    plant_output: FloatArray  # (t, cell, 2): plant voltage + measured surface temp, faulted
    structural_bias: FloatArray  # (n_params,) at the healthy nominal
    delta_true: FloatArray

    @property
    def target_index(self) -> int:
        return self.scenario.target_index

    @property
    def target_sigma(self) -> float:
        return float(self.fit_ctx.crlb_std[self.target_index])

    @property
    def target_bias(self) -> float:
        return float(self.structural_bias[self.target_index])


def prepare(scenario: Scenario) -> ScenarioContext:
    """Precompute the observer design, the faulted plant output, and the structural bias."""
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
    plant = simulate_plant(
        scenario.plant_params,
        scenario.current_a,
        scenario.dt_s,
        scenario.mismatch,
        soc0=scenario.soc0,
        temp0_k=scenario.temp0_k,
        check_bounds=False,
    )
    plant_output = output_tensor(plant.voltage_v, plant.temp_measured_k)

    # The structural bias is the omitted-variable bias of the *healthy* pack, by the convention
    # of ``observability.bias``: the residual is evaluated at the nominal, not at the fault.
    residual = structural_residual(
        scenario.params,
        scenario.mismatch,
        scenario.current_a,
        scenario.dt_s,
        soc0=scenario.soc0,
        temp0_k=scenario.temp0_k,
    )
    bias = parameter_bias(fit_ctx.sens, residual, scenario.topology, scenario.noise, scenario.specs)
    return ScenarioContext(scenario, fit_ctx, plant_output, bias, scenario.delta_true)


@dataclass(frozen=True)
class TrialResults:
    """Per-trial estimates and verdicts for one scenario, plus the constants they share."""

    scenario_name: str
    target_label: str
    delta_true: float
    crlb_std: float
    bias: float
    used_bias_gate: bool
    delta_hat: FloatArray  # (n_trials,)
    snr_var: FloatArray  # (n_trials,) |delta_hat| / sigma
    snr_bias: FloatArray  # (n_trials,) |delta_hat| / sqrt(sigma^2 + b^2)
    verdicts: tuple[VerdictKind, ...]  # (n_trials,)

    @property
    def n_trials(self) -> int:
        return self.delta_hat.size


def run_trials(
    scenario: Scenario,
    n_trials: int,
    *,
    seed: int,
    estimator: str = "linear",
    estimator_options: dict[str, float] | None = None,
    prepared: ScenarioContext | None = None,
) -> TrialResults:
    """Draw ``n_trials`` noise realisations, fit the fault each time, decide each time.

    ``estimator='linear'`` (default) is the exact linear-Gaussian fit -- fast, general in the
    nuisance, and the right workhorse for the magnitude sweeps. ``estimator='gauss_newton'``
    fits the nonlinear observer instead, at physical-only specs; it is the one that *attains*
    the CRLB under a matched model, so it is what the coverage curves use. ``estimator_options``
    forwards to ``fit_gauss_newton`` (e.g. ``{'max_iter': 10}``) -- the mismatch coverage is
    insensitive to the cap because the structural bias dwarfs any sub-sigma convergence detail,
    so a low cap keeps the example fast without changing the result.

    Pass ``prepared`` to reuse a context across ``run_trials`` calls on the same scenario.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if estimator not in ("linear", "gauss_newton"):
        raise ValueError(f"unknown estimator {estimator!r}")
    options = estimator_options or {}

    sctx = prepared if prepared is not None else prepare(scenario)
    ctx = sctx.fit_ctx
    ti = scenario.target_index
    sigma = sctx.target_sigma
    bias_target = sctx.target_bias
    total = float(np.hypot(sigma, bias_target))
    gate_bias = bias_target if scenario.use_bias_gate else None

    max_step = float(options.get("max_step", DEFAULT_MAX_STEP))
    max_iter = int(options.get("max_iter", DEFAULT_MAX_ITER))
    step_tol = float(options.get("step_tol", DEFAULT_STEP_TOL))

    rng = np.random.default_rng(seed)
    n_time = scenario.current_a.size

    delta_hat = np.empty(n_trials)
    snr_var = np.empty(n_trials)
    snr_bias = np.empty(n_trials)
    verdicts: list[VerdictKind] = []

    for k in range(n_trials):
        noise = sample_measurement_noise(scenario.topology, scenario.noise, n_time, rng)
        measurement = sctx.plant_output + noise
        if estimator == "linear":
            delta = fit_linear(ctx, measurement)
        else:
            delta = fit_gauss_newton(
                ctx, measurement, max_step=max_step, max_iter=max_iter, step_tol=step_tol
            ).delta

        estimate = float(delta[ti])
        magnitude = abs(estimate)
        delta_hat[k] = estimate
        snr_var[k] = magnitude / sigma if np.isfinite(sigma) and sigma > 0.0 else 0.0
        snr_bias[k] = magnitude / total if np.isfinite(total) and total > 0.0 else 0.0
        verdicts.append(verdict_from_estimate(ctx, ti, delta, bias=gate_bias).kind)

    return TrialResults(
        scenario_name=scenario.name,
        target_label=scenario.target_spec.label(),
        delta_true=scenario.fault_magnitude,
        crlb_std=sigma,
        bias=bias_target,
        used_bias_gate=scenario.use_bias_gate,
        delta_hat=delta_hat,
        snr_var=snr_var,
        snr_bias=snr_bias,
        verdicts=tuple(verdicts),
    )
