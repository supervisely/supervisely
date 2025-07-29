import pip

# install clip
pip.main(["install", "git+https://github.com/supervisely-ecosystem/depends-CLIP.git"])

from .sly_tracker import DeepSortTracker
