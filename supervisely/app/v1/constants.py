STATE = "state"
DATA = "data"
CONTEXT = "context"
TEMPLATE = "template"

SHARED_DATA = '/sessions'

STOP_COMMAND = "stop"

IMAGE_ANNOTATION_EVENTS = ["manual_selected_figure_changed"]

# Error message for missing or incompatible protobuf dependencies
PROTOBUF_REQUIRED_ERROR = (
    "protobuf is required for agent/worker/app_v1 functionality. "
    "Please install supervisely with agent extras: pip install 'supervisely[agent]'"
)
