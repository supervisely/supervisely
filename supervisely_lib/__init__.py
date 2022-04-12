import sys
import supervisely
from supervisely import *

sys.modules['supervisely_lib'] = supervisely
sys.modules['supervisely_lib.api.api'] = supervisely.api
