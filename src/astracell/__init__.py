"""AstraCell: observability-aware fault identifiability analysis for battery packs.

The central question this package answers is *not* "what fault is present?" but
"given this battery model, this sensor topology, this measurement noise, and the
excitation actually present in the data, which faults could I possibly resolve?"

Everything else is downstream of that question.
"""

__version__ = "0.9.0"

__all__ = ["__version__"]
