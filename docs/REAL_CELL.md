# Real-cell contact: the Oxford Battery Degradation Dataset (Tier 3, v0.6)

Every result in this repository before v0.6 is synthetic. Tier 1 tests the ECM against itself and
against a mismatch we hand-wrote; Tier 2 tests it against PyBaMM, an electrochemical simulator we
did not implement. Both are models. This document covers the machinery that lets AstraCell's
observer run against **eight real lithium-ion cells** — and, just as importantly, everything that
machinery still does *not* establish.

> **v0.6 is contact, not validation.** The dataset is a licensed ~266 MB download that is **never
> committed** to this repository. The [Results](#results-first-run-cell1-2026-07) below come from a
> real run — one cell of eight, a first-order ECM, and a refusal on every age. You reproduce them by
> fetching the data and re-running `examples/08`; the offline test suite exercises the whole pipeline
> on *synthetic* Oxford-format data so it stays green without the download. This establishes that
> AstraCell knows when it cannot trust the ECM on a real cell — and nothing broader. See
> [CLAIMS.md](CLAIMS.md) C19 and [LIMITATIONS.md](../LIMITATIONS.md) §16.

## Why this is the right next step, and what makes it different

The honest next step every prior version pointed at is a **measured** pseudo-OCV and pulse response,
which no simulation can stand in for. A public degradation dataset supplies three things a simulator
cannot:

1. **A real ground truth.** The dataset measures each cell's capacity directly, from the charge its
   own 1C discharge delivered at each age. So for the first time AstraCell's capacity estimate can
   be scored against a number **nobody chose** — `measured_fade` — rather than against an injected
   truth (Tier 1) or a simulator's parameter (Tier 2).
2. **A real, imperfect baseline.** The earliest characterisation age is a genuine beginning-of-life
   fingerprint — a different day and temperature history from every later age, exactly the messiness
   [LIMITATIONS §15d](../LIMITATIONS.md#15-the-refusal-now-discriminates--but-only-because-v04-went-looking-for-the-bug)
   named as the impossible-to-supply assumption behind the v0.4 positive control.
3. **A truth-free screen that needs neither.** The differential lack-of-fit
   (`observability.bias.lack_of_fit_norm`) reads only what the observer cannot reproduce at any
   parameter setting. It requires no ground truth at all, and it is what should fire hardest on a
   real cell.

**The honest expectation, stated before any run.** Continuing the v0.3 → v0.4 arc — where a
*healthier* external plant made the ECM look *worse* and the gate refuse harder — a real cell should
mismatch the first-order ECM at least as badly as PyBaMM did. The likely outcome is that AstraCell
**refuses capacity even harder** on a real cell. That is not a failure. A diagnostic that abstains
on data it cannot trust is the entire thesis.

## The dataset

Oxford Battery Degradation Dataset 1 — eight Kokam SLPB533459H4 740 mAh pouch cells, cycled in a
40 °C chamber on an urban Artemis drive-cycle discharge, characterised every 100 cycles with a 1C
discharge (`C1dc`) and a slow ~C/18 pseudo-OCV sweep (`OCVdc`) until end of life (~20–30 % fade).

- **DOI** `10.5287/bodleian:KO2kdmYGg` · [ORA record](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac)
- **Licence** ODC Open Database License (ODC-ODbL) v1.0; individual contents under the DbCL. The
  data is yours to accept those terms for — AstraCell does not accept them for you, does not
  redistribute the data, and never commits it.
- **Cite** Howey, D., & Birkl, C. (2017). *Oxford Battery Degradation Dataset 1.* University of
  Oxford. The dataset Readme also asks that you cite Birkl, C. R. (2017), *Diagnosis and Prognosis
  of Degradation in Lithium-Ion Batteries*, PhD thesis, University of Oxford.
- **Structure** (confirmed against the Readme) `CellN.cycNNNN.{C1ch,C1dc,OCVch,OCVdc}.{t,v,q,T}`,
  with `t` in seconds, `v` in volts, `q` cumulative charge in **mAh**, `T` in °C. The main file is
  MATLAB v7.3 (HDF5), so it loads via `h5py`, not `scipy.io`.

## How to run

```bash
pip install -e ".[oxford]"          # adds scipy + h5py, used only by the loader
python scripts/fetch_oxford.py      # prints the ODbL notice; accept it; ~254 MB to data/oxford/
python examples/08_real_cell.py     # runs the observer against Cell1, writes real_cell_capacity.png
```

`fetch_oxford.py` streams one file from the archive, refuses to proceed without licence acceptance,
guards against downloading an error page (by size), and **prints the file's SHA-256 for you to
compare against the archive** — it asserts no checksum this repository never verified. If the direct
URL ever rots, download by hand from the record page and pass `--from <path>` (or set
`ASTRACELL_OXFORD_MAT`). Without the data, `examples/08` skips cleanly, exactly as 06–07 skip
without PyBaMM.

## The method

`src/astracell/plant/oxford.py` is a **data source**, nothing more: it turns the `.mat` into clean
arrays and has no opinion about fitting. The example wires those arrays into the *unchanged* v0.4
paired estimator, exactly as `examples/07` wires PyBaMM in.

- **`pseudo_ocv_curve(age)`** builds the observer's OCV from the age's *measured* `OCVdc` table — the
  first non-stand-in OCV curve in the project (contrast [LIMITATIONS §1](../LIMITATIONS.md#1-the-models-are-stand-ins-not-cells)).
  The entropic coefficient is left at zero: a room-temperature sweep does not measure `dOCV/dT`.
- **`measured_fade(baseline, aged)`** is the signed relative capacity deviation from the two 1C
  discharges — a *measurement*, and the Tier-3 ground truth the estimate is scored against.
- **`aligned_pair(baseline, aged, window_s)`** resamples both discharges onto one grid so a paired
  fit is well-posed. An aged cell's 1C discharge is shorter (less capacity, emptied sooner), so a
  fixed window is taken from each. The observer re-integrates from SOC 0.98 — the ECM's ceiling, not
  the cell's — so the top ~2 % of charge is not represented, and the window is chosen to keep the
  replay above ~SOC 0.03 as well. Both bounds are read from the data.

`examples/08` runs two observers against each aged cell:

- **shared-OCV (deployable).** Calibrate the pseudo-OCV once on the fresh cell and track. A real
  cell's OCV *moves* as it ages; the first-order ECM cannot express that motion, so the lack-of-fit
  screen should fire and capacity should be refused. This is the honest field setting.
- **per-age OCV (control).** Re-measure the pseudo-OCV at every age, removing the OCV drift the ECM
  cannot model. Not deployable — you cannot re-characterise a pack in a car — but it isolates how
  much of the mismatch was the moving OCV versus everything else a real cell does.

## What this establishes, and what it does not

- It **establishes** that AstraCell can ingest a real, measured cell and score its own capacity
  estimate against a real fade — and it exposes the machinery to whatever a real cell does to a
  first-order ECM. Whether the answer is refusal or diagnosis, it is the first answer in this
  repository that is not about a model.
- It does **not** establish that the ECM is *right* about a real cell. Tier 3 here is **contact, not
  validation**: one cell of eight, a first-order Thevenin observer, an isothermal fit that ignores
  the 40 °C thermal history, a shared baseline that is a different day, and a paired window that
  discards the ends of the SOC range the ECM cannot represent. Every reason to distrust a confident
  diagnosis is present — which is exactly why a refusal, if that is what prints, is the point.
- No physical *fault* is injected or detected. The dataset's ageing is real capacity fade, scored
  against its own measurement; it is not a labelled fault-detection benchmark.

See [LIMITATIONS.md §16](../LIMITATIONS.md#16-the-real-cell-is-contact-not-validation) for the full
account of what a real-cell result would and would not mean.

## Results (first run: Cell1, 2026-07)

From `examples/08_real_cell.py` on **Oxford Cell1** — 78 characterisation ages, 13 scored evenly
across life, `SEED=0`. Data fetched 2026-07 (266 MB; SHA-256 `a8f0b928…cf26781`). Measured, not
asserted: regenerate these and the figure by fetching the data and re-running the two commands above.

**The cell degrades; the first-order ECM says it gains capacity.**

| quantity | fresh (cyc0000) | end of life (cyc8200) |
|---|---|---|
| measured 1C capacity | 0.7391 Ah | 0.5606 Ah |
| **measured fade** (ground truth) | 0% | **−24.2%** |
| ECM capacity estimate — shared OCV | — | **+10.5%** |
| ECM capacity estimate — per-age OCV | — | **+17.1%** |
| variance-only interval (1σ) | — | **±0.03%** |
| differential lack-of-fit | — | 815 |
| verdict | — | **REFUSE_MODEL_BIAS** |

Across all 13 scored ages the naive ECM capacity estimate is not merely wrong in magnitude but
**wrong in sign**: it reports a +1.9% → +10.5% capacity *gain* (shared OCV) while the cell loses
3.7% → 24.2%. The variance-only interval is **±0.03%**, so the +10.5% estimate sits roughly
**1150σ** from the −24.2% truth — exquisitely precise and catastrophically wrong, the
confident-wrongness this whole project is built against. The differential lack-of-fit climbs
91 → 815 (its contract: hundreds when the observer cannot reproduce a change), coverage of the
measured fade is **0 / 13**, and AstraCell returns **REFUSE_MODEL_BIAS on every age, in both OCV
modes (13/13)**. Re-measuring the pseudo-OCV at each age (the per-age control) does **not** rescue
the estimate — it makes it worse (+17.1% at end of life), because the dominant mismatch is the
first-order *dynamics*, not the moving OCV. One per-age point (cyc3200) collapses to a near-zero
estimate at vanishing σ — a real-data conditioning quirk, visible in the figure and still refused.

This is the honest expectation, stated in advance and now measured: a real cell mismatches the ECM
harder than PyBaMM did (§14), so AstraCell refuses capacity harder. What it establishes is exactly
[LIMITATIONS §16b](../LIMITATIONS.md)'s narrow claim — that AstraCell *knows when it cannot trust the
ECM on a real cell* — and nothing broader. It is one cell of eight; it is not validation of the ECM,
and no fault was detected. **The refusal is the result.**

## A note on units: the run corrected the Readme

The end-to-end run did what no synthetic test could — it caught a unit the Readme states wrongly. The
Readme says time `t` is in **seconds**; the file actually stores a **MATLAB datenum in days** —
Cell1's first sample reads `735954.82`, and day 735954 is 2015-01-08, *exactly* the Readme's own
"Start date of tests". Left as seconds, the derived 1C current came out near 64000 A and every window
collapsed to zero. The scale is now `_TIME_TO_S = 86400` (days → seconds), verified against that
embedded date and pinned in `tests/test_oxford.py`; the `q`-in-mAh → Ah scaling and the `dq/dt` →
amps (`×3600`) conversion for `dz/dt = −I/(Q·3600)` are pinned alongside it. Two further real-file
surprises the run handled: the file is an old-format `.mat` that scipy reads directly (not the
v7.3/HDF5 the Readme implies), and the slow pseudo-OCV repeats SOC where its 1 Hz charge counter
holds, so `pseudo_ocv_curve` collapses exact duplicates before building the table. The Readme is a
starting point; the run is the authority — which is the whole reason to run it.
