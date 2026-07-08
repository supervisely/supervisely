from fastapi import FastAPI, HTTPException, Request, Response
import uvicorn
import threading
import asyncio

from typing import TYPE_CHECKING, List

from .request_queue import RequestQueue, RequestType
from .utils import TrainingStoppedException
import supervisely as sly
from supervisely import logger
from supervisely.project.project_type import _MULTIVIEW_TAG_NAME

if TYPE_CHECKING:
    from supervisely.api.image_api import ImageInfo
    from .live_training import LiveTraining


def start_api_server(
    app: sly.Application,
    request_queue: RequestQueue,
    lt: "LiveTraining" = None,
    host: str = "0.0.0.0",
    port: int = 8000
) -> threading.Thread:
    """Start FastAPI server in a daemon thread."""
    server = app.get_server()
    create_api(server, request_queue, lt)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    thread = threading.Thread(target=server.run, daemon=True, name="APIServer")
    thread.start()
    
    logger.debug(f"Live Training API server started: http://{host}:{port}")

    return thread


def create_api(app: FastAPI, request_queue: RequestQueue, lt: "LiveTraining" = None) -> FastAPI:

    @app.post("/start")
    async def start(response: Response):
        """Start the live training process."""
        future = request_queue.put(RequestType.START)
        result = await _wait_for_result(future, response, timeout=None)
        return result
        
    @app.post("/predict")
    async def predict(request: Request, response: Response):
        """Run inference on an image."""
        sly_api = _api_from_request(request)
        state = request.state.state
        img_np = sly_api.image.download_np(state['image_id'])
        future = request_queue.put(
            RequestType.PREDICT,
            {'image': img_np, 'image_id': state['image_id']}
        )
        result = await _wait_for_result(future, response)
        return result
    
    @app.post("/add-sample")
    async def add_sample(request: Request, response: Response):
        """Add a new training sample."""
        sly_api = _api_from_request(request)
        state = request.state.state
        image_infos = _get_training_sample_image_infos(sly_api, state['image_id'], lt)
        result = None
        for image_info in image_infos:
            img_np = sly_api.image.download_np(image_info.id)
            ann_json = sly_api.annotation.download_json(image_info.id)
            future = request_queue.put(
                RequestType.ADD_SAMPLE,
                {
                    'image': img_np,
                    'annotation': ann_json,
                    'image_id': image_info.id,
                    'image_name': image_info.name
                }
            )
            result = await _wait_for_result(future, response)
            if response.status_code is not None and response.status_code >= 400:
                return result

        if isinstance(result, dict):
            result['image_ids'] = [image_info.id for image_info in image_infos]
        return result

    @app.post("/status")
    async def status(response: Response):
        """Check the status of the training process."""
        # In "continue" mode, status requests received after the START signal
        # must wait until the checkpoint is fully restored, otherwise the UI
        # polls status mid-restore (phase still ready_to_start) and re-shows
        # the "Start Live Training" button. Requests before START pass through.
        if (
            lt is not None
            and lt.checkpoint_mode == "continue"
            and lt._start_received.is_set()
            and not lt._continue_checkpoint_loaded.is_set()
        ):
            await asyncio.to_thread(lt._continue_checkpoint_loaded.wait)
        future = request_queue.put(RequestType.STATUS)
        result = await _wait_for_result(future, response)
        return result

    return app


def _get_training_sample_image_infos(
    api: sly.Api,
    image_id: int,
    lt: "LiveTraining" = None,
) -> List["ImageInfo"]:
    image_info = api.image.get_info_by_id(image_id)
    if image_info is None:
        raise ValueError(f"Image {image_id} not found.")

    group_value = _get_multiview_group_value(image_info, api, lt)
    if group_value is None:
        return [image_info]

    group_infos = []
    for candidate in api.image.get_list(image_info.dataset_id):
        if _get_multiview_group_value(candidate, api, lt) == group_value:
            group_infos.append(candidate)

    if not group_infos:
        return [image_info]

    logger.info(
        f"Resolved multiview group '{group_value}' into "
        f"{len(group_infos)} image(s): {[info.id for info in group_infos]}"
    )
    return group_infos


def _get_multiview_group_value(
    image_info: "ImageInfo",
    api: sly.Api,
    lt: "LiveTraining" = None,
):
    project_id = image_info.project_id or getattr(lt, "project_id", None)
    tag_id = _get_multiview_tag_id(api, project_id, lt)
    if tag_id is None:
        return None

    for tag in image_info.tags or []:
        if tag.get("tagId", tag.get("tag_id")) == tag_id:
            return tag.get("value")
    return None


def _get_multiview_tag_id(api: sly.Api, project_id: int, lt: "LiveTraining" = None):
    project_meta = getattr(lt, "project_meta", None)
    tag_meta = project_meta.get_tag_meta(_MULTIVIEW_TAG_NAME) if project_meta is not None else None

    if (tag_meta is None or tag_meta.sly_id is None) and project_id is not None:
        project_meta_json = api.project.get_meta(project_id, with_settings=True)
        tag_meta = sly.ProjectMeta.from_json(project_meta_json).get_tag_meta(_MULTIVIEW_TAG_NAME)

    if tag_meta is not None:
        return tag_meta.sly_id
    return None
async def _wait_for_result(future: asyncio.Future, response: Response, timeout: float = 600.0):
    """Wait for the future to complete with a timeout."""
    try:
        result = await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        # raise HTTPException(503, detail={"error": "Request timeout - training may be busy"})
        response.status_code = 500
        result = _error_response_message("Request timeout - training may be busy")
    except TrainingStoppedException as e:
        response.status_code = 404
        result = _error_response_message(str(e))
    except Exception as e:
        # raise HTTPException(500, detail={"error": str(e)})
        response.status_code = 500
        result = _error_response_message(str(e))
    return result


def _api_from_request(request: Request) -> sly.Api:
    api = None
    try:
        api = request.state.api
    finally:
        if not isinstance(api, sly.Api):
            logger.warning("sly.Api instance not found in request.state.api. Creating API from app's credentials.")
            api = sly.Api()
    return api


def _error_response_message(message: str):
    return {
        "details": {
            "message": message,
            "checkLogs": False
        }
    }
