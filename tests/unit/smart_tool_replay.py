"""Self-contained replay of the Smart Tool direct-mask evidence in the Python 3.11 SDK profile.

The test image ships a bare interpreter, so this bootstrap (standard library only) builds the
documented SDK test environment - a virtual environment with ``pytest`` and this checkout
installed as ``.[apps]``, exactly like the SDK profile check does - and then runs one stage:

* ``pytest``  - ``tests/unit/test_smart_tool_init_mask.py``, the offline regressions;
* ``repro``   - ``tests/unit/smart_tool_direct_mask_repro.py``, the issue reproducer at HEAD;
* ``diff``    - ``tests/unit/smart_tool_baseline_diff.py``, the same reproducer on the baseline
  commit (``VERIFY_BASE_SHA``) and on HEAD in one run.

Usage: ``python tests/unit/smart_tool_replay.py [stage]`` from the repository root; with no
stage every stage runs. The environment is cached between runs, exits nonzero if any stage
fails, and prints only the tail of each stage so replay output stays small.
"""

import os
import shutil
import site
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_DIR = Path(
    os.environ.get("SMART_TOOL_REPLAY_VENV", Path(tempfile.gettempdir()) / "smart-tool-replay-venv")
)
STAGES = {
    "pytest": [
        "-m",
        "pytest",
        "tests/unit/test_smart_tool_init_mask.py",
        "-q",
        "--no-header",
        "-p",
        "no:warnings",
    ],
    "repro": ["tests/unit/smart_tool_direct_mask_repro.py"],
    "diff": ["tests/unit/smart_tool_baseline_diff.py"],
}
TAIL_LINES = {"pytest": 6, "repro": 24, "diff": 26}


def venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(argv, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800, **kwargs
    )


def usable(python: Path) -> bool:
    if not python.is_file():
        return False
    probe = run([str(python), "-c", "import supervisely, numpy, cv2, pytest, fastapi"])
    return probe.returncode == 0


def install(python: Path, args, env) -> None:
    uv = "/stack/uv" if Path("/stack/uv").is_file() else shutil.which("uv")
    if uv:
        argv = [uv, "pip", "install", "--python", str(python), "-q", *args]
    else:
        argv = [str(python), "-m", "pip", "install", "-q", *args]
    result = run(argv, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"installing {args} failed with exit {result.returncode}:\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )


def build_env() -> Path:
    """Creates the SDK test environment the profile check documents, or reuses a cached one."""
    python = venv_python(VENV_DIR)
    if usable(python):
        print(f"[replay] reusing the test environment at {VENV_DIR}")
        return python
    shutil.rmtree(VENV_DIR, ignore_errors=True)
    print(
        f"[replay] building the Python {sys.version_info.major}.{sys.version_info.minor} SDK test environment at {VENV_DIR}"
    )
    uv = "/stack/uv" if Path("/stack/uv").is_file() else shutil.which("uv")
    if uv:
        create = run(
            [uv, "venv", "--python", sys.executable, "--system-site-packages", str(VENV_DIR)]
        )
    else:
        create = run([sys.executable, "-m", "venv", "--system-site-packages", str(VENV_DIR)])
    if create.returncode != 0:
        raise RuntimeError(
            f"creating the virtual environment failed:\n{create.stdout[-2000:]}\n{create.stderr[-2000:]}"
        )
    python = venv_python(VENV_DIR)
    # A venv made from an image interpreter that lives in another venv does not inherit its
    # packages; add the image paths behind the new environment's own, as the profile check does.
    target = Path(
        run(
            [str(python), "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))']
        ).stdout.strip()
    )
    extra = [p for p in site.getsitepackages() if p != str(target) and Path(p).is_dir()]
    if extra and target.is_dir():
        (target / "agent-image-packages.pth").write_text("".join(p + "\n" for p in extra))
    env = dict(os.environ, RELEASE_VERSION="0.0.0.dev0", PIP_DISABLE_PIP_VERSION_CHECK="1")
    install(python, ["pytest", "setuptools", "wheel", "requests"], env)
    # setup.py only detects an installed headless OpenCV without build isolation.
    install(python, ["opencv-python-headless<5"], env)
    install(python, ["--no-build-isolation", ".[apps]"], env)
    if not usable(python):
        raise RuntimeError("the test environment is still missing the SDK dependencies")
    return python


def run_stage(name: str, python: Path) -> bool:
    argv = [str(python), *STAGES[name]]
    print(f"[replay] === {name}: {' '.join(argv[1:])}")
    result = run(argv)
    output = (result.stdout + result.stderr).strip().splitlines()
    tail = TAIL_LINES[name] if result.returncode == 0 else 60
    if len(output) > tail:
        print(f"[replay] ... {len(output) - tail} earlier lines omitted")
    print("\n".join(output[-tail:]))
    print(f"[replay] === {name}: exit={result.returncode}")
    return result.returncode == 0


def main(argv) -> int:
    requested = argv[1:] or list(STAGES)
    unknown = [stage for stage in requested if stage not in STAGES]
    if unknown:
        print(f"[replay] unknown stage(s): {unknown}; known stages: {list(STAGES)}")
        return 2
    print(f"[replay] repository: {REPO_ROOT}")
    print(
        f"[replay] baseline commit for the diff stage: {os.environ.get('VERIFY_BASE_SHA', '(default in smart_tool_baseline_diff.py)')}"
    )
    try:
        python = build_env()
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"[replay] FAILED to prepare the test environment: {error}")
        return 1
    failed = [stage for stage in requested if not run_stage(stage, python)]
    if failed:
        print(f"[replay] FAILED stages: {failed}")
        return 1
    print(f"[replay] OK: {requested}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
