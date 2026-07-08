"""Identifiability analysis: what could this data possibly tell you?

The pipeline is:

    sensitivities  ->  design matrix  ->  Fisher information  ->  Cramer-Rao bound
                             ^
                             |
                    sensor topology + noise

and then a fault magnitude turns a Cramer-Rao bound into a detection SNR, and an
SNR threshold turns that into ``OBSERVABLE`` / ``WEAK`` / ``UNOBSERVABLE``.

Nothing in here is a heuristic. The grey cells on the pack map are not painted by
a distance-to-nearest-sensor rule; they fall out of the Cramer-Rao bound, which
knows about thermal mass, conduction anisotropy, excitation, and noise.
"""

from astracell.observability.decision import (
    Verdict,
    VerdictKind,
    assess,
    decide,
    sensor_recommendation,
)
from astracell.observability.detectability import HeatmapResult, detectability_heatmap
from astracell.observability.fisher import (
    condition_number,
    crlb,
    crlb_std,
    design_matrix,
    fisher_information,
    information_gain,
    variance_inflation,
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
    ParameterSpec,
    ParamKind,
    all_specs,
    local_specs,
    sensitivities,
)

__all__ = [
    "DEFAULT_STRONG_SIGMA",
    "DEFAULT_WEAK_SIGMA",
    "GreyCellMap",
    "HeatmapResult",
    "Observability",
    "ParamKind",
    "ParameterSpec",
    "Verdict",
    "VerdictKind",
    "all_specs",
    "assess",
    "classify",
    "condition_number",
    "crlb",
    "crlb_std",
    "decide",
    "design_matrix",
    "detectability_heatmap",
    "detection_snr",
    "fisher_information",
    "grey_cell_map",
    "information_gain",
    "local_specs",
    "recommend_temp_sensor",
    "sensor_recommendation",
    "sensitivities",
    "variance_inflation",
]
