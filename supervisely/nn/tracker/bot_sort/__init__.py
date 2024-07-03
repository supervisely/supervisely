import torch  # pylint: disable=import-error

try:
    import fastreid
except ImportError:
    import sys
    from pathlib import Path

    fast_reid_repo_url = "https://github.com/supervisely-ecosystem/fast-reid.git"
    fast_reid_parent_path = Path(__file__).parent
    fast_reid_path = fast_reid_parent_path.joinpath("fast_reid")
    if not fast_reid_path.exists():
        import subprocess

        subprocess.run(["git", "clone", fast_reid_repo_url, str(fast_reid_path.resolve())])

    sys.path.insert(0, str(fast_reid_path.resolve()))

    import fastreid

from supervisely.nn.tracker.bot_sort.sly_tracker import BoTTracker
