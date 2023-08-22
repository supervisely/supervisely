import numpy as np
import functools
from fastapi import Request, BackgroundTasks
from typing import Any, Dict, List, Optional, Union
import supervisely as sly
from supervisely.nn.inference.tracking.tracker_interface import TrackerInterface
from supervisely.nn.inference import Inference
import supervisely.nn.inference.tracking.functional as F


class Cuboid3DTracking(Inference):
    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[Union[Dict[str, Any], str]] = None,
    ):
        super().__init__(
            model_dir,
            custom_inference_settings,
            sliding_window_mode=None,
            use_gui=False,
        )

        try:
            self.load_on_device(model_dir, "cuda")
        except RuntimeError:
            self.load_on_device(model_dir, "cpu")

    def get_info(self):
        info = super().get_info()
        info["task type"] = "tracking"
        return info

    def serve(self):
        super().serve()
        server = self._app.get_server()

        @server.post("/interpolate_figures_ids")
        def start_track(request: Request, task: BackgroundTasks):
            task.add_task(track, request)
            return {"message": "Track task started."}

        def send_error_data(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                value = None
                try:
                    value = func(*args, **kwargs)
                except Exception as exc:
                    request: Request = args[0]
                    context = request.state.context
                    api: sly.Api = request.state.api
                    track_id = context["trackId"]
                    api.logger.error("An error occured:")
                    api.logger.exception(exc)

                    api.post(
                        "videos.notify-annotation-tool",
                        data={
                            "type": "videos:tracking-error",
                            "data": {
                                "trackId": track_id,
                                "error": {"message": repr(exc)},
                            },
                        },
                    )
                return value

            return wrapper

        @send_error_data
        def track(request: Request = None):
            state = request.state.state
            api: sly.Api = request.state.api
