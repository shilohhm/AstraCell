"""The Oxford Battery Degradation Dataset as a data source for AstraCell's ECM observer (v0.6).

This is AstraCell's first contact with a *measured* cell. Every tier of validation before this
one was synthetic: the ECM against itself (Tier 1a), a hand-built mismatch (Tier 1b), and PyBaMM,
an electrochemical simulator we did not implement (Tier 2). PyBaMM is a better stand-in than the
ECM, but it is still a model. The Oxford data is eight real Kokam pouch cells, cycled to end of
life, and it is the only thing here that has ever touched a battery.

**What this module is, and is not.** It is a *data source*: it turns the dataset's ``.mat`` file
into clean arrays -- a measured pseudo-OCV table, a 1C-discharge voltage trace at each
characterisation age, and the measured capacity at each age. It imports only numpy and
``cell.ocv``; it has no opinion about fitting. The example (``examples/08_real_cell.py``) wires
these arrays into the *unchanged* v0.4 paired estimator (``calibration.positive_control``), exactly
as ``examples/07`` wires PyBaMM in. The point of v0.3-v0.4 was to build that estimator once; this
adds a real plant, not a second estimator.

**The honest expectation.** A real cell mismatches the first-order ECM at least as badly as PyBaMM
did -- it has hysteresis, path dependence, and a temperature history the isothermal observer
ignores. So the likely result, continuing the v0.3->v0.4 arc, is that AstraCell *refuses capacity
even harder* on a real cell, and the lack-of-fit gate fires. That is not a failure; a diagnostic
that abstains on data it cannot trust is the entire thesis. Whatever the numbers turn out to be,
they must come from a real run against the downloaded data -- see ``docs/REAL_CELL.md`` -- and are
never hard-coded here.

**Provenance and licence.** Oxford Battery Degradation Dataset 1, Howey & Birkl (2017), University
of Oxford, DOI ``10.5287/bodleian:KO2kdmYGg``, released under the ODC Open Database License
(ODC-ODbL). The data is ~254 MB and is **not committed to this repository**; fetch it with
``scripts/fetch_oxford.py``. The optional ``[oxford]`` extra (``scipy`` / ``h5py``) is used only by
:func:`load_cell`; every pure function below runs on plain arrays with no extra installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from astracell.cell.ocv import OcvCurve, ocv_from_table

FloatArray = NDArray[np.float64]
# cell -> cyc -> measurement -> field: the nested-dict shape the .mat reader returns.
_Nested = dict[str, dict[str, dict[str, FloatArray]]]

# The dataset's own struct, confirmed from its Readme:
#   CellN.cycNNNN.{C1ch, C1dc, OCVch, OCVdc}.{t, v, q, T}
# We read the discharge halves: OCVdc (the slow pseudo-OCV) and C1dc (the 1C load to fit against).
_OCV_KEY = "OCVdc"
_LOAD_KEY = "C1dc"

# --------------------------------------------------------------------------------------------
# Provenance of the data this module reads -- every field below was verified against the Oxford
# Research Archive record, not asserted from memory. scripts/fetch_oxford.py prints them before it
# downloads anything; docs/REAL_CELL.md repeats the citation. The data is ODC-ODbL and is never
# committed to this repository.
# --------------------------------------------------------------------------------------------
DATASET_DOI = "10.5287/bodleian:KO2kdmYGg"
DATASET_RECORD_URL = "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac"
DATASET_URL = (
    "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac"
    "/files/m5ac36a1e2073852e4f1f7dee647909a7"
)
DATASET_FILENAME = "Oxford_Battery_Degradation_Dataset_1.mat"
DATASET_SIZE_HUMAN = "266 MB (253.8 MiB, as ORA lists it)"
DATASET_LICENSE = "ODC Open Database License (ODC-ODbL) v1.0"
DATASET_LICENSE_URL = "https://opendatacommons.org/licenses/odbl/1.0/index.html"
DATASET_CITATION = (
    "Howey, D., & Birkl, C. (2017). Oxford Battery Degradation Dataset 1. "
    "University of Oxford. https://doi.org/10.5287/bodleian:KO2kdmYGg"
)
#: The dataset Readme asks that the underlying thesis be cited alongside the DOI. Honour that.
DATASET_CITATION_THESIS = (
    "Birkl, C. R. (2017). Diagnosis and Prognosis of Degradation in Lithium-Ion Batteries. "
    "PhD thesis, Department of Engineering Science, University of Oxford."
)
#: ORA publishes no SHA-256 for the file and this repo never ships it, so there is no trusted digest
#: to assert here. scripts/fetch_oxford.py computes and prints the digest of what it actually
#: downloaded; record it yourself. A hard-coded hash we never verified would be the exact kind of
#: unmeasured number this project refuses to invent.
DATASET_SHA256: str | None = None

_ENV_VAR = "ASTRACELL_OXFORD_MAT"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def default_mat_path() -> Path:
    """Where the Oxford ``.mat`` is expected on disk.

    The ``ASTRACELL_OXFORD_MAT`` environment variable wins if set (point it anywhere, e.g. a shared
    cache); otherwise a gitignored ``data/oxford/`` under the repository root. ``scripts/
    fetch_oxford.py`` writes here and ``examples/08_real_cell.py`` and the ``oxford``-marked tests
    read from here, so the one location lives in one place.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override)
    return _REPO_ROOT / "data" / "oxford" / DATASET_FILENAME


@dataclass(frozen=True)
class OxfordAge:
    """One characterisation age of one cell, as clean unit-normalised arrays.

    Everything downstream consumes *this*, not the ``.mat`` file, so the fitting logic is testable
    on synthetic ``OxfordAge`` objects with no dataset present. Conventions:

    * ``soc`` / ``ocv_v`` -- the pseudo-OCV table, SOC in [0, 1] (dimensionless, so unit-robust).
    * ``time_s`` / ``current_a`` / ``voltage_v`` -- the 1C discharge, discharge current *positive*
      (matching ``pack.simulate``: SOC falls under positive current).
    * ``capacity_ah`` -- the charge the 1C discharge removed, the age's measured capacity. This is
      the Tier-3 ground truth against which the observer's estimate is scored.
    """

    cyc: int
    soc: FloatArray
    ocv_v: FloatArray
    time_s: FloatArray
    current_a: FloatArray
    voltage_v: FloatArray
    capacity_ah: float


def pseudo_ocv_curve(age: OxfordAge, *, name: str | None = None) -> OcvCurve:
    """Turn an age's measured pseudo-OCV into the analytic-ish curve the sensitivities need.

    A *measured* OCV table is exactly the input ``cell.ocv.ocv_from_table`` was written for (its
    docstring names "a bench"). The entropic coefficient is left at zero: a room-temperature
    pseudo-OCV sweep does not measure ``dOCV/dT``, and inventing one is the kind of unmeasured
    number this repository exists to avoid.

    The raw table is not strictly monotone: a ~C/18 sweep sampled at 1 Hz repeats the charge
    counter -- and hence SOC -- wherever it holds between samples, so consecutive rows share a SOC.
    Sort and keep the first row at each distinct SOC, the strictly-increasing table that
    ``ocv_from_table`` (and the interpolation the sensitivities ride on) requires. This is data
    conditioning, not fabrication: no SOC or voltage is invented, only exact duplicates dropped.
    """
    order = np.argsort(age.soc, kind="stable")
    soc = age.soc[order]
    ocv = age.ocv_v[order]
    keep = np.concatenate(([True], np.diff(soc) > 0.0))
    return ocv_from_table(
        soc[keep],
        ocv[keep],
        name=name or f"oxford_pseudo_ocv_cyc{age.cyc:04d}",
        chemistry="measured Kokam pouch (Oxford Battery Degradation Dataset)",
    )


def measured_fade(baseline: OxfordAge, aged: OxfordAge) -> float:
    """The measured relative capacity deviation of ``aged`` from ``baseline`` (negative for fade).

    This is a *measurement*, from the two full 1C discharges, not a parameter we set -- which is
    precisely what makes it a Tier-3 ground truth. It matches ``Scenario.fault_magnitude``'s
    convention (capacity falls, so the number is negative) and is the value the observer's
    short-window estimate is scored against.
    """
    if baseline.capacity_ah <= 0.0:
        raise ValueError("baseline capacity must be positive")
    return (aged.capacity_ah - baseline.capacity_ah) / baseline.capacity_ah


def aligned_pair(
    baseline: OxfordAge,
    aged: OxfordAge,
    *,
    window_s: float,
    dt_s: float = 1.0,
    soc0: float = 0.98,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Resample both ages' 1C discharges onto one common window, so a paired fit is well-posed.

    The paired estimator needs both traces on the *same* excitation and grid, but an aged cell's
    1C discharge is shorter than a fresh one -- less capacity, emptied sooner. Both discharges
    start from a full, freshly-charged cell, so we take a fixed ``window_s`` from each discharge's
    start (where the current is a near-constant 1C) and resample onto a shared ``dt_s`` grid.

    The capacity difference then expresses itself the way the observer can actually see it: the
    aged cell's voltage droops faster for the same removed charge, and the ECM, fitting capacity,
    reads that as a lower ``Q``. The shared current is the baseline's (both are ~1C).

    ``soc0`` is where the *observer* re-integrates the replay, and it is bounded by the ECM, not by
    the cell: ``pack.simulate`` represents SOC only within ``[~0.02, 0.98]``, so the replay must
    start at or below ``0.98`` -- the default. The real full-charge discharge begins above that, so
    the top ~2% of charge is not represented and ``window_s`` must be chosen to keep the replay
    above ~0.02 as well (a longer window at 1C drives SOC down ``window_s / 3600`` of full scale; if
    it crosses the floor, ``simulate`` raises, loudly, rather than fitting nonsense). That the
    first-order ECM cannot express the curve's ends is itself a real source of lack-of-fit on a real
    cell -- expected, and quantified in ``docs/REAL_CELL.md``, not hidden.
    Returns ``(current_a, baseline_v, aged_v, soc0)``.

    This fixed-window alignment is the simplest defensible choice and it is where "real data is
    messier than a simulator" lives: the two currents are not bit-identical, and the two cells were
    not measured on the same day. ``docs/REAL_CELL.md`` states what that costs.
    """
    if window_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("window_s and dt_s must be positive")
    n = int(round(window_s / dt_s))
    if n < 4:
        raise ValueError("window is too short to fit (need >= 4 samples)")
    grid = np.arange(n, dtype=float) * dt_s

    def _resample(age: OxfordAge, field: FloatArray) -> FloatArray:
        t0 = age.time_s - age.time_s[0]
        if t0[-1] < window_s:
            raise ValueError(
                f"cyc{age.cyc:04d} discharge is only {t0[-1]:.0f} s, shorter than the "
                f"{window_s:.0f} s window"
            )
        return np.interp(grid, t0, field)

    current_a = _resample(baseline, baseline.current_a)
    baseline_v = _resample(baseline, baseline.voltage_v)
    aged_v = _resample(aged, aged.voltage_v)
    return current_a, baseline_v, aged_v, soc0


# --------------------------------------------------------------------------------------------
# The thin .mat shell. Everything above runs on arrays; this is the only part that needs the
# download and the optional [oxford] extra, and it is the only part whose unit assumptions cannot
# be verified without the data. Tests skip it (the `oxford` marker); the pure core above does not.
# --------------------------------------------------------------------------------------------

#: Unit scales for the .mat fields. ``q`` (cumulative charge) is in **mAh**, as the Readme states.
#: ``t``, however, is a **MATLAB datenum in days**, NOT the "seconds" the Readme claims: the real
#: file's first C1dc sample reads 735954.82 -- day 735954 is 2015-01-08, exactly the Readme's own
#: "Start date of tests" -- sampled every 1.157e-5 days (= 1.0000 s). The end-to-end run caught
#: this; the Readme did not, which is why the run, not the Readme, is the authority (LIMITATIONS
#: §16c). The ECM coulomb count (``cell.ecm.soc_step``: ``dz/dt = -I/(Q_ah*3600)``) wants capacity
#: in Ah and current in A, so ``q`` -> Ah and ``t`` -> seconds here, and the current derived from
#: them is scaled to amps in ``_age_from_fields``. Overridable so a future dataset needs no logic.
_CHARGE_TO_AH: float = 1e-3  # mAh -> Ah
_TIME_TO_S: float = 86400.0  # datenum days -> seconds (Readme says "seconds"; the file disagrees)


def _age_from_fields(
    cyc: int,
    ocvdc: dict[str, FloatArray],
    c1dc: dict[str, FloatArray],
    *,
    charge_to_ah: float = _CHARGE_TO_AH,
    time_to_s: float = _TIME_TO_S,
) -> OxfordAge:
    """Assemble an :class:`OxfordAge` from the two discharge structs' ``t/v/q`` arrays.

    SOC is a charge *ratio*, so it is immune to the ``q`` unit; capacity and current are not, hence
    the explicit scales. This function is pure and unit-tested on synthetic dicts; only its callers
    touch the real file.
    """
    q_ocv = np.asarray(ocvdc["q"], dtype=float)
    v_ocv = np.asarray(ocvdc["v"], dtype=float)
    removed = q_ocv - q_ocv.min()
    span = removed.max()
    if span <= 0.0:
        raise ValueError(f"cyc{cyc:04d} OCVdc has no charge span")
    soc = 1.0 - removed / span  # full -> empty over the slow discharge

    # ``t`` is a datenum in days with a huge epoch offset (~735954), so subtract the first sample
    # *before* scaling to seconds -- keeping elapsed time small and exact rather than 6.4e10 s.
    t_raw = np.asarray(c1dc["t"], dtype=float)
    t = (t_raw - t_raw[0]) * time_to_s  # -> elapsed seconds
    v = np.asarray(c1dc["v"], dtype=float)
    q = np.asarray(c1dc["q"], dtype=float) * charge_to_ah  # -> Ah
    capacity_ah = float(np.abs(q.max() - q.min()))
    # Discharge current in amps, positive, from the charge derivative: dq/dt is Ah/s, and 1 Ah/s is
    # 3600 A, so x3600 converts to the amps the ECM's coulomb count expects. Near-constant at 1C.
    current_a = np.abs(np.gradient(q, t)) * 3600.0

    return OxfordAge(
        cyc=cyc,
        soc=soc,
        ocv_v=v_ocv,
        time_s=t,
        current_a=current_a,
        voltage_v=v,
        capacity_ah=capacity_ah,
    )


def load_cell(path: str, cell: str = "Cell1") -> dict[int, OxfordAge]:
    """Load one cell's characterisation ages from the Oxford ``.mat`` file.

    Requires the optional ``[oxford]`` extra and the downloaded data (``scripts/fetch_oxford.py``).
    The 254 MB file is MATLAB v7.3 (HDF5) in practice, so ``scipy.io.loadmat`` will refuse it and we
    fall back to ``h5py``; both paths are handled. Navigates ``cell -> cycNNNN -> {OCVdc, C1dc}``.

    The unit assumptions in ``_age_from_fields`` (``q`` in mAh, ``t`` in seconds) are confirmed
    against the dataset Readme; the end-to-end run against the real file is the final check, and the
    scales are overridable there should a future dataset differ. This function is deliberately not
    imported at module top level so the pure core needs no optional dependency.
    """
    struct = _read_mat_cell(path, cell)
    ages: dict[int, OxfordAge] = {}
    for key, cyc_struct in struct.items():
        if not key.startswith("cyc"):
            continue
        try:
            cyc = int(key[3:])
        except ValueError:
            continue
        if _OCV_KEY not in cyc_struct or _LOAD_KEY not in cyc_struct:
            continue
        ages[cyc] = _age_from_fields(cyc, cyc_struct[_OCV_KEY], cyc_struct[_LOAD_KEY])
    if not ages:
        raise ValueError(f"no usable characterisation cycles found for {cell} in {path}")
    return dict(sorted(ages.items()))


def _read_mat_cell(path: str, cell: str) -> _Nested:
    """Read ``cell -> cyc -> field -> {t,v,q,T}`` as nested dicts, via scipy or (v7.3) h5py.

    Isolated so the two MATLAB-format code paths -- and their differing struct-access idioms -- are
    the only place that depends on the file format. Not exercised by the offline test suite.
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - guarded by the `oxford` marker
        raise ImportError(
            "the Oxford loader needs the optional [oxford] extra (scipy, h5py)"
        ) from exc

    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        return _walk_scipy_struct(mat[cell])
    except NotImplementedError:  # v7.3 files are HDF5; scipy cannot read them
        import h5py

        with h5py.File(path, "r") as handle:
            return _walk_h5_struct(handle, cell)


def _walk_scipy_struct(cell_obj: object) -> _Nested:
    """Flatten a scipy ``mat_struct`` (``cyc -> {C1dc,OCVdc} -> {t,v,q,T}``) into plain dicts.

    Covered by ``tests/test_oxford.py`` on a synthetic old-format .mat. The real 254 MB file is
    v7.3/HDF5 and takes the :func:`_walk_h5_struct` path instead, but the navigation is identical.
    """
    out: _Nested = {}
    for cyc in getattr(cell_obj, "_fieldnames", []):
        cyc_obj = getattr(cell_obj, cyc)
        measurements: dict[str, dict[str, FloatArray]] = {}
        for meas in getattr(cyc_obj, "_fieldnames", []):
            meas_obj = getattr(cyc_obj, meas)
            measurements[meas] = {
                f: np.asarray(getattr(meas_obj, f), dtype=float)
                for f in getattr(meas_obj, "_fieldnames", [])
            }
        out[cyc] = measurements
    return out


def _walk_h5_struct(handle: object, cell: str) -> _Nested:  # pragma: no cover - needs the real file
    """Flatten a v7.3 (HDF5) group tree for one cell into plain dicts.

    MATLAB writes v7.3 structs as HDF5 groups; iterating a group yields its member names and the
    array leaves need raveling. Kept minimal and behind the same contract as the scipy path.
    """
    root = handle[cell]  # type: ignore[index]
    out: _Nested = {}
    for cyc in root:
        measurements: dict[str, dict[str, FloatArray]] = {}
        for meas in root[cyc]:
            measurements[meas] = {
                field: np.asarray(root[cyc][meas][field], dtype=float).ravel()
                for field in root[cyc][meas]
            }
        out[cyc] = measurements
    return out
