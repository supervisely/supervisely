"""Differential replay: the Smart Tool direct-mask reproducer on the baseline and on HEAD.

Runs ``smart_tool_direct_mask_repro.py`` twice with the same inputs: once against the
sources of the baseline commit (``VERIFY_BASE_SHA``, or the first argument) and once
against the sources of this checkout. The baseline must fail by downloading the annotation
of the edited figure, this checkout must serve the very same request from its ``mask``.

Usage: ``python tests/unit/smart_tool_baseline_diff.py [baseline-commit]`` from the
repository root. Materializing a baseline may require ``git fetch`` and therefore network
access when that commit is not already local. The paired replay builds its test environment
by installing packages, which likewise needs network when that environment is not already
present. Exits 0 only if HEAD serves the request and the materialized baseline fails on the
annotation download; exits 1 for either replay failure, and 2 when the baseline sources
cannot be materialized. A skipped stage never exits 0.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
REPRO = TESTS_DIR / "smart_tool_direct_mask_repro.py"
HARNESS = TESTS_DIR / "smart_tool_harness.py"
# The baseline fails inside the deprecated figure-id download; both markers must show up.
BASELINE_MARKERS = ("download_init_mask", "must not be downloaded")
# The master commit this branch is based on and was reproduced on, used when none is given.
DEFAULT_BASELINE = "efde8255cb54e4dccd69391f7b6210e621e9dc30"


def run_repro(repo_root: Path) -> subprocess.CompletedProcess:
    """Runs the reproducer against the sources of ``repo_root``."""
    print(f"+ {sys.executable} tests/unit/smart_tool_direct_mask_repro.py  (cwd={repo_root})")
    return subprocess.run(
        [sys.executable, "tests/unit/smart_tool_direct_mask_repro.py"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=600,
    )


def materialize_baseline(commit: str, target: Path):
    """Writes the sources of ``commit`` into ``target``, or returns the reason it cannot."""
    archive_command = ["git", "archive", "--format=tar", commit]
    archive = subprocess.run(
        archive_command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        timeout=600,
    )
    if archive.returncode != 0:
        fetch_command = ["git", "fetch", "--depth=1", "origin", commit]
        print(f"+ {' '.join(fetch_command)}  (cwd={REPO_ROOT})")
        fetch = subprocess.run(
            fetch_command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            timeout=300,
        )
        print(f"--- git fetch --depth=1 origin {commit}: exit={fetch.returncode}")
        if fetch.stdout:
            print(fetch.stdout.decode("utf-8", "replace").strip())
        if fetch.stderr:
            print(fetch.stderr.decode("utf-8", "replace").strip())
        archive = subprocess.run(
            archive_command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            timeout=600,
        )
    if archive.returncode != 0:
        return archive.stderr.decode("utf-8", "replace").strip() or "git archive failed"
    target.mkdir(parents=True, exist_ok=True)
    extract = subprocess.run(
        ["tar", "-x", "-C", str(target)], input=archive.stdout, capture_output=True, timeout=600
    )
    if extract.returncode != 0:
        return extract.stderr.decode("utf-8", "replace").strip() or "tar failed"
    # The reproducer and its harness are inputs of the replay, not baseline sources.
    (target / "tests" / "unit").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPRO, target / "tests" / "unit" / REPRO.name)
    shutil.copy(HARNESS, target / "tests" / "unit" / HARNESS.name)
    return None


def report(title: str, result: subprocess.CompletedProcess):
    print(f"--- {title}: exit={result.returncode}")
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())


def main(argv) -> int:
    commit = argv[1] if len(argv) > 1 else os.environ.get("VERIFY_BASE_SHA") or DEFAULT_BASELINE
    failures = []
    baseline_skipped = False

    head = run_repro(REPO_ROOT)
    report("HEAD", head)
    if head.returncode != 0:
        failures.append("HEAD must serve the direct-mask request without a figure id")

    if not commit:
        print("--- BASELINE: no baseline commit resolved")
        failures.append("baseline commit is required to compare against")
    else:
        with tempfile.TemporaryDirectory(prefix="smart-tool-baseline-") as tmp:
            baseline_root = Path(tmp) / "baseline"
            problem = materialize_baseline(commit, baseline_root)
            if problem:
                baseline_skipped = True
                print(
                    f"[diff] SKIPPED: baseline {commit} unavailable in this checkout: {problem}; "
                    "only the HEAD assertions above were checked"
                )
            else:
                baseline = run_repro(baseline_root)
                report(f"BASELINE {commit}", baseline)
                if baseline.returncode == 0:
                    failures.append(
                        f"baseline {commit} unexpectedly served the direct-mask request"
                    )
                output = baseline.stdout + baseline.stderr
                missing = [marker for marker in BASELINE_MARKERS if marker not in output]
                if missing:
                    failures.append(
                        f"baseline {commit} did not fail on the annotation download: "
                        f"missing {missing}"
                    )

    if failures:
        print(f"[diff] FAILED: {failures}")
        return 1
    if baseline_skipped:
        return 2
    print("[diff] OK: baseline downloads the annotation, HEAD serves the request from the mask")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
