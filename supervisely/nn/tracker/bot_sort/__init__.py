import torch

try:
    import fast_reid
except ImportError:
    import sys
    from pathlib import Path

    fast_reid_parent_path = Path(__file__).parent
    fast_reid_path = fast_reid_parent_path.joinpath("fast_reid")
    sys.path.insert(0, str(fast_reid_parent_path.resolve()))
    sys.path.append(str(fast_reid_path.resolve()))

    import fast_reid

from .sly_tracker import BoTTracker
