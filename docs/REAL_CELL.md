# Real-cell contact: the Oxford Battery Degradation Dataset (Tier 3, v0.6–v0.7)

Every result in this repository before v0.6 is synthetic. Tier 1 tests the ECM against itself and
against a mismatch we hand-wrote; Tier 2 tests it against PyBaMM, an electrochemical simulator we
did not implement. Both are models. This document covers the machinery that lets AstraCell's
observer run against **eight real lithium-ion cells** — and, just as importantly, everything that
machinery still does *not* establish.

> **Contact, not validation.** The dataset is a licensed ~266 MB download that is **never
> committed** to this repository. The [Results](#results-all-eight-cells-2026-07) below come from a
> real run — v0.7 across **all eight cells**, a first-order ECM, and a refusal on every one of 208
> evaluations. You reproduce them by fetching the data and re-running `examples/08`; the offline test
> suite exercises the whole pipeline on *synthetic* Oxford-format data so it stays green without the
> download. This establishes that AstraCell knows when it cannot trust the ECM on a real cell — and
> nothing broader. See [CLAIMS.md](CLAIMS.md) C19 and [LIMITATIONS.md](../LIMITATIONS.md) §16.

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

- It **establishes** that AstraCell can ingest eight real, measured cells and score its own capacity
  estimate against each one's real fade — and it exposes the machinery to whatever a real cell does to
  a first-order ECM. The answer, on all eight, is refusal; either way it is the first answer in this
  repository that is not about a model.
- It does **not** establish that the ECM is *right* about a real cell. Tier 3 here is **contact, not
  validation**: eight cells but one chemistry, a first-order Thevenin observer, an isothermal fit that
  ignores the 40 °C thermal history, a shared baseline that is a different day, and a paired window
  that discards the ends of the SOC range the ECM cannot represent. Breadth across the eight cells
  does not touch any of those — it only shows the phantom and the refusal are not a Cell1 accident.
  Every reason to distrust a confident diagnosis is present — which is exactly why the refusal, which
  is what prints, is the point.
- No physical *fault* is injected or detected. The dataset's ageing is real capacity fade, scored
  against its own measurement; it is not a labelled fault-detection benchmark.

See [LIMITATIONS.md §16](../LIMITATIONS.md#16-the-real-cell-is-contact-not-validation) for the full
account of what a real-cell result would and would not mean.

## Results (all eight cells, 2026-07)

From `examples/08_real_cell.py` on **all eight Oxford cells** — 46–78 characterisation ages each,
13 aged ages scored per cell (**104 in total**), `SEED=0`. Data fetched 2026-07 (266 MB; SHA-256
`a8f0b928…cf26781`). Measured, not asserted: regenerate these and the figure by fetching the data
and re-running the two commands above.

**Every cell fades; the first-order ECM reports a gain — and AstraCell refuses all 208 evaluations.**

The worked example is **Cell1**, unchanged from v0.6's first run — one cell of the eight:

| quantity | fresh (cyc0000) | end of life (cyc8200) |
|---|---|---|
| measured 1C capacity | 0.7391 Ah | 0.5606 Ah |
| **measured fade** (ground truth) | 0% | **−24.2%** |
| ECM capacity estimate — shared OCV | — | **+10.5%** |
| ECM capacity estimate — per-age OCV | — | **+17.1%** |
| variance-only interval (1σ) | — | **±0.03%** |
| differential lack-of-fit | — | 815 |
| verdict | — | **REFUSE_MODEL_BIAS** |

On Cell1 the estimate is **wrong in sign**: +1.9% → +10.5% capacity *gain* (shared OCV) while the
cell loses 3.7% → 24.2%. The ±0.03% interval puts the +10.5% estimate roughly **1150σ** from the
−24.2% truth — exquisitely precise and catastrophically wrong, the confident-wrongness this project
is built against — and the differential lack-of-fit climbs 91 → 815 (its contract: hundreds when the
observer cannot reproduce a change).

v0.7 runs the same pipeline across all eight and asks whether that is a property of Cell1 or of the
observer meeting a real cell. Two readings a breadth run is for:

**The refusal distribution.** Across 8 cells × 13 ages × 2 OCV modes = **208 evaluations**, the
verdict is **`REFUSE_MODEL_BIAS` every single time**: shared-OCV **104/104**, per-age-OCV **104/104**,
**8/8 cells refuse every scored age**, and no other verdict kind (no WEAK, no DIAGNOSE) ever appears.
Coverage of the measured fade is **0/104** in both modes, and the variance-only interval is a median
**±0.03%** wide — so those are not near-misses; the estimate and the truth do not overlap.

**The phantom-gain spread.** Every cell loses a fifth to a third of its capacity: measured EOL fade
**−20.0% (Cell7) to −38.0% (Cell5)**, median −23.1%. The deployable shared-OCV estimate lands **−0.2%
to +14.3%** (median +10.7%): **7 of 8 cells report an outright capacity *gain*** at end of life, wrong
in sign against a real loss, and across all 104 shared-OCV ages **95%** of estimates read positive.
The one exception is **Cell5**, the most-faded of the eight (−38.0%), whose shared estimate collapses
to **≈0%** at a vanishing σ — still tens of points from its measured fade, still the highest
lack-of-fit of any cell (1151), and still refused. Re-measuring the pseudo-OCV at every age (the
per-age control) does **not** rescue the estimate — it makes it worse, **+13.3% to +18.9%** (all eight
a gain) — because the dominant mismatch is the first-order *dynamics*, not the moving OCV.

![all eight Oxford cells: every cell fades, the ECM reports a gain, AstraCell refuses every age](../reports/figures/real_cell_capacity.png)

This is the honest expectation, stated in advance and now measured on eight cells rather than one: a
real cell mismatches the first-order ECM harder than PyBaMM did (§14), so AstraCell refuses capacity
harder — and it does so on *every* cell, which makes the phantom the observer's failure to represent
a real cell, not a quirk of Cell1. What it establishes is exactly [LIMITATIONS §16b](../LIMITATIONS.md)'s
narrow claim — that AstraCell *knows when it cannot trust the ECM on a real cell* — now across the
whole dataset. It does **not** upgrade the tier: still one first-order ECM, still isothermal, still a
shared baseline a different day, still **no fault detected** and no validation of the ECM. **The
refusal is the result — eight cells over, not one.**

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
