"""Identifiability analysis: what could this data possibly tell you?

The pipeline is:

    sensitivities  ->  design matrix  ->  Fisher information  ->  Cramer-Rao bound
                             ^                     ^
                             |                     |
                    sensor topology + noise    nuisance priors

and then a fault magnitude turns a Cramer-Rao bound into a detection SNR, and an
SNR threshold turns that into ``OBSERVABLE`` / ``WEAK`` / ``UNOBSERVABLE``.

Nothing in here is a heuristic. The grey cells on the pack map are not painted by a
distance-to-nearest-sensor rule; they fall out of the Cramer-Rao bound, which knows
about thermal mass, conduction anisotropy, excitation, and noise.

When the answer is "unobservable", ``experiment`` and ``mask.recommend_temp_sensor``
say what would change it: excite differently, or instrument differently.
"""

from astracell.observability.decision import (
    Verdict,
    VerdictKind,
    assess,
    decide,
    sensor_recommendation,
)
from astracell.observability.detectability import HeatmapResult, detectability_heatmap
from astracell.observability.experiment import (
    CandidateTest,
    TestScore,
    default_test_library,
    rank_tests,
    render_ranking,
)
from astracell.observability.fisher import (
    condition_number,
    crlb,
    crlb_std,
    design_matrix,
    fisher_information,
    gaussian_entropy,
    information_gain,
    prior_information,
    variance_inflation,
    whiten_ar1,
)
from astracell.observability.mask import (
    DEFAULT_STRONG_SIGMA,
    DEFAULT_WEAK_SIGMA,
    GreyCellMap,
    Observability,
    classify,
    detection_snr,
    grey_cell_map,
    recommend_temp_sensor,
)
from astracell.observability.sensitivity import (
    CELL_PARAM_KINDS,
    CURRENT_BIAS_SPEC,
    GLOBAL_PARAM_KINDS,
    ParameterSpec,
    ParamKind,
    all_specs,
    local_specs,
    sensitivities,
    with_current_bias,
)

__all__ = [
    "CELL_PARAM_KINDS",
    "CURRENT_BIAS_SPEC",
    "DEFAULT_STRONG_SIGMA",
    "DEFAULT_WEAK_SIGMA",
    "GLOBAL_PARAM_KINDS",
    "CandidateTest",
    "GreyCellMap",
    "HeatmapResult",
    "Observability",
    "ParamKind",
    "ParameterSpec",
    "TestScore",
    "Verdict",
    "VerdictKind",
    "all_specs",
    "assess",
    "classify",
    "condition_number",
    "crlb",
    "crlb_std",
    "decide",
    "default_test_library",
    "design_matrix",
    "detectability_heatmap",
    "detection_snr",
    "fisher_information",
    "gaussian_entropy",
    "grey_cell_map",
    "information_gain",
    "local_specs",
    "prior_information",
    "rank_tests",
    "recommend_temp_sensor",
    "render_ranking",
    "sensitivities",
    "sensor_recommendation",
    "variance_inflation",
    "whiten_ar1",
    "with_current_bias",
]
