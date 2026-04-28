#!/usr/bin/env python3
"""SOSCF validation suite runner.

Reads test/SOSCF/expected.csv, copies each <mol>/<exc>/dftb_in.hsd plus
geometry into a scratch dir, runs build/app/dftb+/dftb+, parses
autotest.tag for the converged total (Mermin) energy, and compares to
the reference within `ENERGY_TOL_H`.  Also gates on a per-test maximum
SOSCF iter count to catch performance regressions.

The dftb_in.hsd templates use @SKDIR@ as a placeholder for the
Slater-Koster directory.  By default it resolves to
external/slakos/origin/mio-1-1 (standard DFTB+ layout); override via the
SKDIR env var when the SK files live elsewhere.

Stand-alone:
    python3 test/SOSCF/run_tests.py
    SKDIR=/path/to/mio-1-1 python3 test/SOSCF/run_tests.py
"""

import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTDIR = Path(__file__).resolve().parent
DFTBP = ROOT / "build" / "app" / "dftb+" / "dftb+"
EXPECTED = TESTDIR / "expected.csv"

ENERGY_TOL_H = 1e-6

# dftb_in.hsd files use @SKDIR@ as a placeholder for the Slater-Koster
# directory (mio-1-1).  Default location follows the standard DFTB+ layout;
# overridable via the SKDIR env var.
SKDIR = os.environ.get("SKDIR") or str(ROOT / "external/slakos/origin/mio-1-1")


def parse_autotest_tag(path: Path) -> float | None:
    """Extract mermin_energy (Hartree) from autotest.tag, or None if missing."""
    if not path.exists():
        return None
    text = path.read_text().splitlines()
    for i, line in enumerate(text):
        if line.startswith("mermin_energy"):
            return float(text[i + 1].split()[0])
    for i, line in enumerate(text):
        if line.startswith("total_energy"):
            return float(text[i + 1].split()[0])
    return None


def count_soscf_iters(stdout_log: Path) -> int:
    """Count SOSCF outer iterations from stdout.

    The header line is `iSCC Total electronic ...` and each iter is a row
    starting with whitespace then an integer index.  Counts the lines after
    the LAST occurrence of the header (i.e., the SOSCF outer-loop block).
    """
    if not stdout_log.exists():
        return -1
    text = stdout_log.read_text().splitlines()
    last_hdr = -1
    for i, line in enumerate(text):
        if "iSCC" in line and "Total electronic" in line:
            last_hdr = i
    if last_hdr < 0:
        return -1
    n = 0
    for line in text[last_hdr + 1 :]:
        if re.match(r"^\s+\d+\s+-?\d", line):
            n += 1
        elif line.strip() and not re.match(r"^\s+\d", line):
            break
    return n


def converged(stdout_log: Path) -> bool:
    """Inspect stdout for explicit non-convergence warning from runSoscfLoop."""
    if not stdout_log.exists():
        return False
    text = stdout_log.read_text()
    if "SOSCF: outer loop did not converge" in text:
        return False
    if "Delta-SCF (pre-SOSCF): handoff" in text and "not reached" in text:
        return False
    return True


def run_one(
    name: str, expected_e: float, max_iters: int, work_root: Path
) -> tuple[str, str, float | None, int]:
    src = TESTDIR / name
    if not (src / "dftb_in.hsd").exists():
        return ("SKIP", f"no dftb_in.hsd at {src}", None, 0)

    sandbox_label = name.replace("/", "__").replace(",", "_")
    sb = work_root / sandbox_label
    sb.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        dst = sb / f.name
        if f.name == "dftb_in.hsd":
            dst.write_text(f.read_text().replace("@SKDIR@", SKDIR))
        else:
            shutil.copy(f, dst)

    stdout_log = sb / "stdout.log"
    proc = subprocess.run(
        [str(DFTBP)],
        cwd=sb,
        env={"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1"},
        stdout=stdout_log.open("w"),
        stderr=subprocess.STDOUT,
    )

    if proc.returncode != 0:
        return ("FAIL", f"dftb+ exit {proc.returncode}", None, 0)

    actual_e = parse_autotest_tag(sb / "autotest.tag")
    if actual_e is None:
        return ("FAIL", "no autotest.tag energy", None, 0)
    if not converged(stdout_log):
        return ("FAIL", "not converged", actual_e, count_soscf_iters(stdout_log))
    n_iters = count_soscf_iters(stdout_log)

    diff = abs(actual_e - expected_e)
    if diff >= ENERGY_TOL_H:
        return (
            "FAIL",
            f"energy diff {diff:.2e} > {ENERGY_TOL_H:.0e} (expected {expected_e}, got {actual_e})",
            actual_e,
            n_iters,
        )
    if n_iters > max_iters:
        return ("FAIL", f"iter regression: {n_iters} > {max_iters}", actual_e, n_iters)
    return ("PASS", "", actual_e, n_iters)


def main() -> int:
    if not DFTBP.exists():
        sys.stderr.write(f"ERROR: {DFTBP} not found — build first.\n")
        return 2
    if not EXPECTED.exists():
        sys.stderr.write(f"ERROR: {EXPECTED} not found.\n")
        return 2
    if not Path(SKDIR).is_dir():
        sys.stderr.write(
            f"ERROR: SKDIR={SKDIR} is not a directory "
            "(set SKDIR env var or place mio-1-1 at external/slakos/origin/mio-1-1).\n"
        )
        return 2

    rows = []
    with EXPECTED.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    work_root = Path(tempfile.mkdtemp(prefix="soscf_validation_"))
    try:
        passed = 0
        failed = 0
        results: list[tuple[str, str, str, str, str]] = []
        for r in rows:
            name = r["test"]
            expected_e = float(r["total_energy_h"])
            max_iters = int(r["max_iters"])
            status, reason, actual_e, n_iters = run_one(
                name, expected_e, max_iters, work_root
            )
            e_str = f"{actual_e:.10f}" if actual_e is not None else "?"
            results.append((status, name, e_str, str(n_iters), reason))
            if status == "PASS":
                passed += 1
            else:
                failed += 1

        for status, name, e_str, iters, reason in results:
            mark = "PASS" if status == "PASS" else status
            print(f"  {mark:<5} {name:<28} E={e_str:<16} iters={iters:<4} {reason}")

        print()
        print(f"=== SOSCF validation: {passed} passed, {failed} failed ===")
        return 0 if failed == 0 else 1
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
