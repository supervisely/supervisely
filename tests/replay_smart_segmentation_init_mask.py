"""Replay of the smart tool init mask regression, at the current checkout and at its base.

Runs ``tests/unit/test_smart_segmentation_init_mask.py`` twice: against the
working checkout, where it must pass, and against the base commit
(``VERIFY_BASE_SHA`` or the first argument), where the same file must fail
because the ``mask`` field of the request context is ignored there.

Usage: python tests/replay_smart_segmentation_init_mask.py [base commit]
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEST = "tests/unit/test_smart_segmentation_init_mask.py"
PYTEST = ["-m", "pytest", TEST, "-p", "no:warnings", "--no-header", "-q"]


def run(argv, cwd=None, env=None):
    print("+ " + " ".join(str(a) for a in argv), flush=True)
    return subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=600)


def tail(completed, lines=12):
    output = (completed.stdout + completed.stderr).strip().splitlines()
    return "\n".join(output[-lines:])


def build_env(repo: Path, venv: Path) -> str:
    """Install the test dependencies of the SDK into a throwaway virtualenv."""
    uv = shutil.which("uv") or "/stack/uv"
    env = dict(os.environ, RELEASE_VERSION="0.0.0.dev0", PIP_DISABLE_PIP_VERSION_CHECK="1")
    python = str(venv / "bin/python")
    steps = [
        [uv, "venv", "--python", sys.executable, "--system-site-packages", str(venv)],
        [uv, "pip", "install", "-q", "--python", python, "pytest", "setuptools", "wheel", "requests"],
        [uv, "pip", "install", "-q", "--python", python, "opencv-python-headless<5"],
        [uv, "pip", "install", "-q", "--python", python, "--no-build-isolation", ".[apps]"],
    ]
    for step in steps:
        done = run(step, cwd=repo, env=env)
        if done.returncode != 0:
            sys.exit(f"Could not prepare the test environment:\n{tail(done)}")
    return python


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VERIFY_BASE_SHA")
    if not base:
        sys.exit("Base commit is required: pass it as an argument or set VERIFY_BASE_SHA")

    failures = []
    with tempfile.TemporaryDirectory(prefix="init-mask-replay-") as tmp:
        python = build_env(repo, Path(tmp, "venv"))

        head = run([python, *PYTEST], cwd=repo)
        print(f"--- current checkout ---\n{tail(head)}", flush=True)
        if head.returncode != 0:
            failures.append("the regression test does not pass at the current checkout")

        worktree = Path(tmp, "base")
        checkout = run(["git", "worktree", "add", "--detach", str(worktree), base], cwd=repo)
        if checkout.returncode != 0:
            sys.exit(f"Could not check out base commit {base}:\n{tail(checkout)}")
        try:
            shutil.copyfile(repo / TEST, worktree / TEST)
            old = run([python, *PYTEST], cwd=worktree)
            print(f"--- base commit {base} ---\n{tail(old)}", flush=True)
            if old.returncode == 0:
                failures.append(f"the regression test also passes at base commit {base}")
        finally:
            run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo)

    for failure in failures:
        print(f"REPLAY FAILED: {failure}", flush=True)
    if failures:
        return 1
    print("REPLAY OK: fails at the base commit, passes at the current checkout", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
