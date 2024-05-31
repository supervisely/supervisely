import torch

try:
    import fast_reid
except ImportError:
    import sys
    from pathlib import Path

    fast_reid_repo_url = "https://github.com/JDAI-CV/fast-reid.git"
    fast_reid_parent_path = Path(__file__).parent
    fast_reid_path = fast_reid_parent_path.joinpath("fast_reid")
    if not fast_reid_path.exists():
        import subprocess

        subprocess.run(["git", "clone", fast_reid_repo_url, str(fast_reid_path.resolve())])

    sys.path.insert(0, str(fast_reid_parent_path.resolve()))
    sys.path.append(str(fast_reid_path.resolve()))

    import fast_reid

from .sly_tracker import BoTTracker
