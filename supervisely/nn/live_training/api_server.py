from fastapi import FastAPI, HTTPException, Request, Response
import uvicorn
import threading
import asyncio

from .request_queue import RequestQueue, RequestType
import supervisely as sly
from supervisely import logger


def start_api_server(
    app: sly.Application,
    request_queue: RequestQueue,
    host: str = "0.0.0.0",
    port: int = 8000
) -> threading.Thread:
    """Start FastAPI server in a daemon thread."""
    server = app.get_server()
    create_api(server, request_queue)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    thread = threading.Thread(target=server.run, daemon=True, name="APIServer")
    thread.start()
    
    logger.debug(f"Live Training API server started: http://{host}:{port}")

    return thread


def create_api(app: FastAPI, request_queue: RequestQueue) -> FastAPI:

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

    @app.post("/predict-batch")
    async def predict(request: Request, response: Response):
        """Run inference on a batch of images."""
        sly_api = _api_from_request(request)
        state = request.state.state
        img_nps = sly_api.image.download_nps(state["image_ids"])
        future = request_queue.put(
            RequestType.PREDICT_BATCH, {"images": img_nps, "image_ids": state["image_ids"]}
        )
        result = await _wait_for_result(future, response)
        return result

    @app.post("/predict-video")
    async def predict_video(request: Request, response: Response):
        """Run inference on a video frame."""
        sly_api = _api_from_request(request)
        state = request.state.state
        video_id = state["video_id"]
        frame_idx = state["frame_idx"]
        frame_np = sly_api.video.frame.download_np(video_id, frame_idx)
        frame_id = f"{video_id}_{frame_idx}"
        future = request_queue.put(RequestType.PREDICT, {"image": frame_np, "image_id": frame_id})
        result = await _wait_for_result(future, response)
        return result

    @app.post("/predict-video-batch")
    async def predict_video_batch(request: Request, response: Response):
        """Run inference on a batch of video frames."""
        sly_api = _api_from_request(request)
        state = request.state.state
        video_id = state["video_id"]
        frame_indices = state["frame_indices"]
        frame_nps = sly_api.video.frame.download_nps(video_id, frame_indices)
        frame_ids = [f"{video_id}_{frame_idx}" for frame_idx in frame_indices]
        future = request_queue.put(
            RequestType.PREDICT_BATCH, {"images": frame_nps, "image_ids": frame_ids}
        )
        result = await _wait_for_result(future, response)
        return result

    @app.post("/add-sample")
    async def add_sample(request: Request, response: Response):
        """Add a new training sample."""
        sly_api = _api_from_request(request)
        state = request.state.state
        img_np = sly_api.image.download_np(state['image_id'])
        ann_json = sly_api.annotation.download_json(state['image_id'])
        img_info = sly_api.image.get_info_by_id(state['image_id'])
        future = request_queue.put(
            RequestType.ADD_SAMPLE,
            {
                'image': img_np,
                'annotation': ann_json,
                'image_id': state['image_id'],
                'image_name': img_info.name
            }
        )
        result = await _wait_for_result(future, response)
        return result

    @app.post("/add-sample-video")
    async def add_sample_video(request: Request, response: Response):
        """Add a new training sample from video."""
        sly_api = _api_from_request(request)
        state = request.state.state
        video_id = state["video_id"]
        frame_idx = state["frame_idx"]
        frame_np = sly_api.video.frame.download_np(video_id, frame_idx)
        video_ann_json = sly_api.video.annotation.download(video_id)
        future = request_queue.put(
            RequestType.ADD_SAMPLE_VIDEO,
            {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "frame_np": frame_np,
                "video_ann_json": video_ann_json,
            },
        )
        result = await _wait_for_result(future, response)
        return result

    @app.post("/add-samples-video")
    async def add_samples_video(request: Request, response: Response):
        """Add several new training samples from video."""
        sly_api = _api_from_request(request)
        state = request.state.state
        video_id = state["video_id"]
        frame_indices = state["frame_indices"]
        frame_nps = sly_api.video.frame.download_nps(video_id, frame_indices)
        video_ann_json = sly_api.video.annotation.download(video_id)
        future = request_queue.put(
            RequestType.ADD_SAMPLE_VIDEO,
            {
                "video_id": video_id,
                "frame_indices": frame_indices,
                "frame_nps": frame_nps,
                "video_ann_json": video_ann_json,
            },
        )
        result = await _wait_for_result(future, response)
        return result

    @app.post("/status")
    async def status(response: Response):
        """Check the status of the training process."""
        future = request_queue.put(RequestType.STATUS)
        result = await _wait_for_result(future, response)
        return result
    return app


async def _wait_for_result(future: asyncio.Future, response: Response, timeout: float = 600.0):
    """Wait for the future to complete with a timeout."""
    try:
        result = await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        # raise HTTPException(503, detail={"error": "Request timeout - training may be busy"})
        response.status_code = 503
        result = _error_response_message("Request timeout - training may be busy")
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
    return {"error": {"details": {"message": message}}}
