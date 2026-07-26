# Contributing to AstraCell

Thank you for helping make AstraCell more rigorous. This repository values evidence over
novelty: a smaller claim with a reproducible test is preferable to a broader claim that the
current validation cannot support.

## Development setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,notebook]"  # Windows
# .venv/bin/python -m pip install -e ".[dev,notebook]"   # Linux / macOS
```

Run the same gates used by CI:

```bash
make check
```

Without `make`, run the equivalent commands from [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
PyBaMM and Oxford-data tests use optional dependencies and skip cleanly when their prerequisites
are unavailable.

## What a change should include

- Add or update tests for behavior changes. Numerical findings should have a regression test;
  general behavior should prefer an invariant or property-based test.
- Keep claims traceable. If a documented result changes, update
  [CLAIMS.md](docs/CLAIMS.md), its supporting example, and the relevant limitation together.
- Document negative results. A falsified hypothesis belongs in
  [WHAT_DID_NOT_WORK.md](docs/WHAT_DID_NOT_WORK.md), not in a deleted branch.
- Preserve optional-dependency boundaries. The core package must remain usable without PyBaMM,
  SciPy, h5py, or the Oxford dataset.
- Do not hand-edit generated notebooks. Edit `scripts/build_notebook.py`, then run
  `make notebook && make notebook-run`.

## Pull-request checklist

- [ ] Ruff lint and format checks pass.
- [ ] Mypy passes.
- [ ] The relevant tests pass, including optional suites when the change touches them.
- [ ] New public behavior is documented.
- [ ] Every changed headline number is reproducible from a committed example or test.
- [ ] Limitations and withdrawn claims remain explicit.

For the full fresh-clone and release process, see
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
