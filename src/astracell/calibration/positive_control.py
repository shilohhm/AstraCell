"""The positive control: a real fault in the external plant, and whether AstraCell finds it.

v0.3 established that a *healthy* PyBaMM cell makes AstraCell's ECM observer invent a 67.6%
capacity fault, and that the bias gate refuses it. That is a negative control, and a negative
control on its own proves nothing about a diagnostic: **a system that refuses everything passes
it.** v0.3's external gate did exactly that, and did not know. This module supplies the missing
half -- inject a fault we chose, and ask whether it comes back.

Why the v0.3 gate could not have found one
------------------------------------------

``prepare_external`` projects the trace it is about to fit onto the parameter directions and
calls the answer "structural bias". Since a linear fit to a residual *is* that projection, the
bias equals the expected estimate, ``SNR_total`` is pinned at 1, and ``REFUSE_MODEL_BIAS`` is
returned for any data at all. Fed a healthy cell it says "your fault is model error", which is
true. Fed a broken cell it says the same thing, which is not. See
``observability.bias`` and ``docs/POSITIVE_CONTROL.md`` §2.

The fix, and why the two obvious fixes are one fix
--------------------------------------------------

A bias must be estimated on a plant known to be healthy. Given a healthy baseline trace ``g_h``
and a faulted trace ``g_f`` on the same excitation, two corrections suggest themselves:

* **subtract the baseline bias** -- fit ``g_f``, then subtract ``b_h = FIM^-1 S^T Sigma^-1
  (g_h - f(theta*))``;
* **fit the paired difference** -- fit ``g_f - g_h`` directly.

They are the same estimator. The projection is linear, so

    FIM^-1 S^T Sigma^-1 (g_f - f) - FIM^-1 S^T Sigma^-1 (g_h - f)
        == FIM^-1 S^T Sigma^-1 (g_f - g_h),

exactly, and ``tests/test_positive_control.py`` asserts it to 1e-12. There was never a choice to
make. What the identity does buy is the *honest noise*: two measured traces carry two independent
noise draws, so the differential measurement has covariance ``2 Sigma``. ``paired_noise`` doubles
the variance exactly -- not by scaling ``voltage_sigma_v`` by sqrt(2), which would leave the
quantisation term behind -- and the CRLB widens to ``sqrt(2) sigma``. Subtraction is not free.

What gates the corrected estimate
---------------------------------

Not ``parameter_bias``: on the differential residual it is again the estimate, and refuses
everything again. Not zero either -- that would assert the ECM is right about how a fault looks,
which is the sin this repository exists to prevent. What is left is the part of the differential
the observer *cannot reproduce at any parameter setting*: ``observability.bias.lack_of_fit_bias``.
It is truth-free, invariant to the noise scale, exactly zero when the change is expressible, and
large when it is not. It is a **screen and not a bound**; read that module's final section, and
§5 of ``docs/POSITIVE_CONTROL.md``, before quoting it.

The two contexts this module builds, both fed to v0.2's unchanged ``run_trials``:

``prepare_raw``
    v0.3's path, done properly: fit ``g_f``, gate with ``b_h`` from the healthy baseline. The
    estimate carries the phantom, so it is wrong by ``b_h`` and its interval never covers.

``prepare_paired``
    Fit ``g_f - g_h`` at doubled noise, gate with the differential lack-of-fit. The phantom is
    gone by construction; what remains is the fault, and whatever the fault's own shape does to
    a model that cannot express it.

The baseline is the assumption, and here it is unrealistically perfect
---------------------------------------------------------------------

``g_h`` is the *identical simulation* with the fault parameter at its healthy value: same solver,
same grid, same initial SOC. A real beginning-of-life fingerprint is none of those things -- it is
a different day, a different temperature, a different SOC window, a cell that has since aged in
ways unrelated to the fault. Every number this module produces is therefore an **upper bound on
what any real baseline could deliver**, and the paired estimator's nominal coverage is a statement
about the arithmetic, not a promise about a workshop. ``LIMITATIONS.md`` §15.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from astracell.calibration.external import prepare_external
from astracell.calibration.montecarlo import ScenarioContext
from astracell.calibration.scenario import Scenario
from astracell.observability.bias import lack_of_fit_bias, lack_of_fit_norm
from astracell.observability.estimator import build_context, fit_linear, output_tensor
from astracell.pack.simulate import simulate
from astracell.sensors.noise import NoiseModel

FloatArray = NDArray[np.float64]

__all__ = [
    "PairedEvidence",
    "build_evidence",
    "differential_voltage",
    "paired_noise",
    "paired_scenario",
    "prepare_paired",
    "prepare_raw",
]


def paired_noise(noise: NoiseModel) -> NoiseModel:
    """The noise model of a *difference* of two independently measured traces.

    ``Cov(e_f - e_h) = 2 Sigma`` when the draws are independent, and an AR(1) difference is again
    AR(1) with the same ``rho`` -- the autocorrelation function scales, its shape does not. So the
    only change is a doubled stationary variance.

    Scaling ``voltage_sigma_v`` by ``sqrt(2)`` would *not* do that: ``voltage_variance`` also
    carries the quantisation term ``lsb^2/12``, which the sqrt(2) would miss, and the sampled
    noise would then have a different covariance from the one the Fisher information inverts.
    Solve for the sigma that doubles the *total* instead. The FIM and the sampler both read
    ``voltage_variance`` through ``channel_slices``, so they stay consistent by construction.
    """
    doubled = 2.0 * noise.voltage_variance
    sigma = doubled - noise.voltage_lsb_v**2 / 12.0
    if sigma <= 0.0:  # pragma: no cover - would need lsb^2/12 > 2*variance, i.e. lsb > variance
        raise ValueError("quantisation dominates the voltage variance; cannot double it")
    return replace(noise, voltage_sigma_v=float(np.sqrt(sigma)))


def paired_scenario(scenario: Scenario) -> Scenario:
    """The same experiment, run as a healthy/faulty difference: doubled noise, same everything.

    ``fault_magnitude`` is unchanged -- the truth the differential estimator is chasing is the
    same fault -- but the CRLB it is measured against widens by ``sqrt(2)``, which is the price of
    needing a baseline.
    """
    return replace(scenario, noise=paired_noise(scenario.noise), name=f"{scenario.name}/paired")


def differential_voltage(healthy_v: FloatArray, faulty_v: FloatArray) -> FloatArray:
    """``g_f - g_h``: what the injected degradation did to the terminal voltage, and nothing else.

    Both traces come from the same solver on the same grid under the same excitation, so every
    common-mode modelling artefact -- the electrolyte droop the ECM cannot express, the solver's
    own discretisation -- cancels. What survives is the fault's signature.
    """
    healthy_v = np.asarray(healthy_v, dtype=float)
    faulty_v = np.asarray(faulty_v, dtype=float)
    if healthy_v.shape != faulty_v.shape:
        raise ValueError(f"trace shapes disagree: {healthy_v.shape} vs {faulty_v.shape}")
    return faulty_v - healthy_v


def _voltage_residual_tensor(scenario: Scenario, voltage_residual: FloatArray) -> FloatArray:
    """A ``(t, cell, channel)`` residual carrying ``voltage_residual`` and no temperature signal."""
    n_time = scenario.current_a.size
    tensor = np.zeros((n_time, scenario.topology.n_cells, 2))
    tensor[:, 0, 0] = voltage_residual
    return tensor


def prepare_raw(scenario: Scenario, healthy_v: FloatArray, faulty_v: FloatArray) -> ScenarioContext:
    """Fit the faulted trace; gate with the phantom measured on the healthy baseline.

    This is v0.3's path with its degeneracy removed: ``plant_output`` is the faulted plant, the
    structural bias is ``parameter_bias`` of the *healthy* residual, and the two are no longer the
    same number. The estimate still carries the phantom -- there is no correction here, only an
    honest gate -- so its interval misses the truth by ``b_h``. That is the baseline against which
    ``prepare_paired`` is worth its doubled variance.
    """
    return prepare_external(scenario, faulty_v, baseline_voltage=healthy_v)


def prepare_paired(
    paired: Scenario, healthy_v: FloatArray, faulty_v: FloatArray
) -> ScenarioContext:
    """Fit the healthy/faulty difference; gate with what the observer cannot reproduce.

    ``paired`` must already carry the doubled noise (``paired_scenario``), or the reported CRLB
    understates the differential's variance by ``sqrt(2)`` and every coverage number is wrong.

    The measurement handed to ``run_trials`` is ``f(theta*) + (g_f - g_h)``. Adding the observer's
    own nominal back is not a fudge: ``fit_linear`` subtracts exactly that before projecting, so
    the residual it sees is the differential itself, and ``delta_hat = FIM^-1 S^T Sigma^-1
    (g_f - g_h)``. One noise draw at the doubled variance then reproduces the paired experiment's
    statistics exactly, rather than approximately.

    The structural bias is ``lack_of_fit_bias`` of the **noiseless** differential, matching
    ``structural_residual``'s convention that ``b`` is a property of the experiment and not of a
    draw. A deployed estimator, holding only noisy traces, must debias the norm by its ``chi^2``
    degrees of freedom; the two agree, and the test says by how much.
    """
    healthy_v = np.asarray(healthy_v, dtype=float)
    faulty_v = np.asarray(faulty_v, dtype=float)
    n_time = paired.current_a.size
    if healthy_v.shape != (n_time,) or faulty_v.shape != (n_time,):
        raise ValueError(f"both traces must have shape ({n_time},)")

    fit_ctx = build_context(
        paired.params,
        paired.specs,
        paired.current_a,
        paired.dt_s,
        paired.topology,
        paired.noise,
        soc0=paired.soc0,
        temp0_k=paired.temp0_k,
    )
    nominal = simulate(
        paired.params, paired.current_a, paired.dt_s, soc0=paired.soc0, temp0_k=paired.temp0_k
    )

    diff = differential_voltage(healthy_v, faulty_v)
    plant_output = output_tensor((nominal.voltage_v[:, 0] + diff)[:, None], nominal.temp_k)
    residual = _voltage_residual_tensor(paired, diff)
    bias = lack_of_fit_bias(fit_ctx.sens, residual, paired.topology, paired.noise, paired.specs)

    return ScenarioContext(
        scenario=paired,
        fit_ctx=fit_ctx,
        plant_output=plant_output,
        structural_bias=bias,
        delta_true=paired.delta_true,
    )


@dataclass(frozen=True)
class PairedEvidence:
    """Both readings of one injected fault, plus the diagnostics that explain their disagreement.

    ``misfit_paired`` is ``|| (I - P) r~ ||`` on the differential: zero when the observer can
    reproduce the change exactly, hundreds when it cannot. It is the single number that separates
    the recoverable fault from the confounded one, and it is computable knowing neither.

    The three estimates satisfy ``raw_estimate - phantom == paired_estimate`` exactly, which is
    the identity that collapses "baseline subtraction" and "paired comparison" into one method.
    """

    name: str
    delta_true: float
    raw: ScenarioContext
    paired: ScenarioContext
    raw_estimate: float  # E[delta_hat] fitting the faulted trace: the fault plus the phantom
    paired_estimate: float  # E[delta_hat] fitting the difference: the fault alone
    misfit_raw: float
    misfit_paired: float

    @property
    def target_index(self) -> int:
        return self.paired.target_index

    @property
    def phantom(self) -> float:
        """The healthy-baseline bias on the target: what v0.3's healthy cell mistook for a fault."""
        return self.raw.target_bias

    @property
    def lack_of_fit(self) -> float:
        """The differential lack-of-fit bias on the target, in parameter units."""
        return self.paired.target_bias

    @property
    def raw_lack_of_fit(self) -> float:
        """The same screen applied to the *undifferenced* residual, in parameter units.

        Not what ``prepare_raw`` gates on -- that uses the healthy baseline's ``parameter_bias``,
        which is the stronger statement when a baseline exists. This is what the gate would say
        with no baseline at all, and it is how ``docs/POSITIVE_CONTROL.md`` §2a re-scores v0.3's
        healthy cell: the number to compare against the phantom it is trying to warn about.
        """
        return self.misfit_raw * self.raw.target_sigma


def _expected_estimate(context: ScenarioContext) -> float:
    """``E[delta_hat]`` on the target: the noiseless fit, before any measurement draw."""
    return float(fit_linear(context.fit_ctx, context.plant_output)[context.target_index])


def build_evidence(
    scenario: Scenario,
    healthy_v: FloatArray,
    faulty_v: FloatArray,
    *,
    name: str | None = None,
) -> PairedEvidence:
    """Assemble both contexts, both expected estimates, and both misfits for one injected fault.

    Takes voltage arrays, not a ``PyBaMMPlant``, so the whole positive-control apparatus can be
    exercised on an ECM-generated pair with no optional dependency installed -- the same discipline
    that lets ``prepare_external`` carry a self-consistency control.
    """
    paired = paired_scenario(scenario)
    raw_ctx = prepare_raw(scenario, healthy_v, faulty_v)
    paired_ctx = prepare_paired(paired, healthy_v, faulty_v)

    nominal = simulate(
        scenario.params,
        scenario.current_a,
        scenario.dt_s,
        soc0=scenario.soc0,
        temp0_k=scenario.temp0_k,
    )
    raw_residual = _voltage_residual_tensor(scenario, faulty_v - nominal.voltage_v[:, 0])
    diff_residual = _voltage_residual_tensor(scenario, differential_voltage(healthy_v, faulty_v))

    return PairedEvidence(
        name=name or scenario.name,
        delta_true=scenario.fault_magnitude,
        raw=raw_ctx,
        paired=paired_ctx,
        raw_estimate=_expected_estimate(raw_ctx),
        paired_estimate=_expected_estimate(paired_ctx),
        misfit_raw=lack_of_fit_norm(
            raw_ctx.fit_ctx.sens, raw_residual, scenario.topology, scenario.noise, scenario.specs
        ),
        misfit_paired=lack_of_fit_norm(
            paired_ctx.fit_ctx.sens, diff_residual, paired.topology, paired.noise, paired.specs
        ),
    )
