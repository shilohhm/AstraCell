"""Fetch the Oxford Battery Degradation Dataset 1 for AstraCell's Tier-3 real-cell work.

Run this yourself. Nothing in AstraCell downloads data automatically, and the 253.8 MB file is
**never committed** to the repository: it is released under the ODC Open Database License
(ODC-ODbL), whose terms are yours to accept, not this project's to accept on your behalf. The file
lands in a gitignored ``data/oxford/`` directory (or ``--dest``), where ``examples/08_real_cell.py``
and the ``oxford``-marked tests look for it via :func:`astracell.plant.oxford.default_mat_path`.

    python scripts/fetch_oxford.py --accept-license

What it does, and does not do. It streams one file from the Oxford Research Archive, checks the
result is plausibly the dataset and not an error page (by size), and prints the file's SHA-256 so
you can compare it against the archive -- it does **not** assert a checksum this repository never
verified. If the direct URL ever rots, download the file by hand from the record page and hand it
to the loader with ``--from <path>`` (or ``ASTRACELL_OXFORD_MAT``).

    Howey, D., & Birkl, C. (2017). Oxford Battery Degradation Dataset 1. University of Oxford.
    DOI 10.5287/bodleian:KO2kdmYGg. Licence: ODC-ODbL.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from astracell.plant.oxford import (
    DATASET_CITATION,
    DATASET_CITATION_THESIS,
    DATASET_LICENSE,
    DATASET_LICENSE_URL,
    DATASET_RECORD_URL,
    DATASET_SIZE_HUMAN,
    DATASET_URL,
    default_mat_path,
)

# The real file is ~254 MB; anything much smaller is an HTML error page, not the dataset.
_MIN_PLAUSIBLE_BYTES = 200_000_000
_CHUNK = 1 << 20  # 1 MiB


def _note(message: str = "") -> None:
    """Human-facing output goes to stderr; only the final dataset path goes to stdout."""
    print(message, file=sys.stderr)


def _license_notice() -> str:
    rule = "=" * 78
    return "\n".join(
        [
            rule,
            "Oxford Battery Degradation Dataset 1",
            f"  {DATASET_CITATION}",
            f"  Cite also: {DATASET_CITATION_THESIS}",
            f"  Licence: {DATASET_LICENSE}",
            f"           {DATASET_LICENSE_URL}",
            f"  Record:  {DATASET_RECORD_URL}",
            "",
            "This data is licensed under the ODC-ODbL. By downloading it you accept that licence's",
            "terms. AstraCell does not accept them for you, does not redistribute the data, and",
            "never commits it to git.",
            rule,
        ]
    )


def _acknowledged(accept_flag: bool) -> bool:
    """Show the licence and require acceptance -- a flag for scripts, a prompt for humans."""
    _note(_license_notice())
    if accept_flag:
        _note("Licence accepted via --accept-license.")
        return True
    if not sys.stdin.isatty():
        _note("Refusing to download without acceptance. Re-run with --accept-license.")
        return False
    try:
        reply = input("Type 'yes' to accept the ODC-ODbL and download: ")
    except EOFError:
        _note("\nNo input available (non-interactive). Re-run with --accept-license.")
        return False
    return reply.strip().lower() in {"yes", "y"}


def _progress(done: int, total: int) -> None:
    if total > 0:
        detail = f"{done / 1e6:8.1f} / {total / 1e6:.1f} MB ({100.0 * done / total:5.1f}%)"
    else:
        detail = f"{done / 1e6:8.1f} MB"
    print(f"\r  downloading: {detail}", end="", file=sys.stderr, flush=True)


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` via a ``.part`` file, renamed only on a complete read."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / (dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "astracell-fetch-oxford/0.6"})
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        with part.open("wb") as handle:
            while chunk := response.read(_CHUNK):
                handle.write(chunk)
                done += len(chunk)
                _progress(done, total)
    _note()
    part.replace(dest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    default_dest = default_mat_path()
    parser = argparse.ArgumentParser(description="Fetch the Oxford Battery Degradation Dataset 1.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=default_dest,
        help=f"where to write the .mat (default: {default_dest})",
    )
    parser.add_argument("--url", default=DATASET_URL, help="override the download URL")
    parser.add_argument(
        "--from",
        dest="local",
        type=Path,
        default=None,
        help="copy from a file you downloaded by hand instead of fetching over the network",
    )
    parser.add_argument(
        "--accept-license", action="store_true", help="accept the ODC-ODbL non-interactively"
    )
    parser.add_argument("--force", action="store_true", help="re-download even if --dest exists")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dest: Path = args.dest

    if dest.exists() and not args.force:
        _note(f"Already present: {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
        _note(f"  SHA-256: {_sha256(dest)}")
        _note("  (pass --force to re-download)")
        print(dest)
        return 0

    if not _acknowledged(args.accept_license):
        return 1

    if args.local is not None:
        if not args.local.exists():
            _note(f"No such file: {args.local}")
            return 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.local, dest)
        _note(f"Copied {args.local} -> {dest}")
    else:
        try:
            _download(args.url, dest)
        except (urllib.error.URLError, OSError) as exc:
            _note(f"\nDownload failed: {exc}")
            _note(f"Fetch it by hand from {DATASET_RECORD_URL} and re-run with --from <path>.")
            return 1

    size = dest.stat().st_size
    if size < _MIN_PLAUSIBLE_BYTES:
        _note(
            f"\nGot only {size / 1e6:.1f} MB -- expected ~{DATASET_SIZE_HUMAN}. This is almost "
            "certainly an error page, not the dataset. Removing it."
        )
        dest.unlink(missing_ok=True)
        return 1

    _note(f"\nSaved {dest}  ({size / 1e6:.1f} MB, expected ~{DATASET_SIZE_HUMAN})")
    _note(f"  SHA-256: {_sha256(dest)}")
    _note("  Compare this digest against the Oxford Research Archive record before trusting it.")
    _note(f"\nNext: python examples/08_real_cell.py   (reads {dest})")
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
