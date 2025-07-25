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

    # Import fastreid first
    import fastreid
    
    # Import all backbone modules to ensure proper registration
    try:
        from fastreid.modeling.backbones import (
            build_backbone,
            build_resnet_backbone,
            build_resnet_backbone_NL,
            build_osnet_backbone,
            build_efficientnet_backbone,
            build_mobilenetv2_backbone,
            build_mobilenetv3_backbone,
            build_regnet_backbone,
            build_resnest_backbone,  # This is the missing one!
        )
    except ImportError as e:
        # If specific imports fail, try to import the module directly
        try:
            import fastreid.modeling.backbones.resnest as resnest_module
        except ImportError:
            print(f"Warning: Could not import ResNest backbone: {e}")
    
    # Ensure all backbone modules are imported and registered
    try:
        import fastreid.modeling.backbones.resnet
        import fastreid.modeling.backbones.osnet
        import fastreid.modeling.backbones.resnest
        import fastreid.modeling.backbones.regnet
        import fastreid.modeling.backbones.efficientnet
        import fastreid.modeling.backbones.mobilenet
    except ImportError as e:
        print(f"Warning: Some backbone modules could not be imported: {e}")

from supervisely.nn.tracker.bot_sort_legacy.sly_tracker import BoTTracker