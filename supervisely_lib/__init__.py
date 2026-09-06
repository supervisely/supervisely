import sys

import supervisely
from supervisely import *

sys.modules['supervisely_lib'] = supervisely

for module_name in list(sys.modules.keys()):
    if module_name.startswith("supervisely."):
        new_name = module_name.replace("supervisely.", "supervisely_lib.", 1)
        sys.modules[new_name] = sys.modules[module_name]
