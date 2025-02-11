import supervisely as sly
from supervisely.nn.utils import ModelSource, RuntimeType
import time
from tqdm import tqdm
import random
from typing import List
import numpy as np


def get_supported_models() -> List[str]:
    """
    Returns list of supported foundation models
    """
    supported_models = [
        "Grounding DINO",
        "Florence 2",
        "Kosmos 2",
        "Molmo",
        "SAM 2",
    ]
    return supported_models


def get_supported_checkpoints(model_name: str) -> List[str]:
    """
    Returns list of supported checkpoints for given foundation  model
    """
    supported_checkpoints = {
        "Grounding DINO": ["grounding-dino-tiny", "grounding-dino-base"],
        "Florence 2": [
            "Florence-2-base",
            "Florence-2-large",
            "Florence-2-base-ft",
            "Florence-2-large-ft",
        ],
        "Kosmos 2": ["kosmos-2-patch14-224"],
        "Molmo": ["Molmo-7B-O-0924"],
        "SAM 2": [
            "sam2.1_hiera_tiny.pt",
            "sam2.1_hiera_small.pt",
            "sam2.1_hiera_base_plus.pt",
            "sam2.1_hiera_large.pt",
        ],
    }
    return supported_checkpoints[model_name]


def deploy_foundation_model(
    model_name: str, checkpoint_name: str, api: sly.Api, additional_params: dict = None
) -> sly.nn.inference.Session:
    """
    Deploys given foundation  model from api
    """
    # get workspace and agent id
    workspace_id = sly.env.workspace_id()
    agent_id = sly.env.agent_id()

    # get app slug
    if model_name == "Grounding DINO":
        app_slug = "supervisely-ecosystem/serve-grounding-dino"
    elif model_name == "Florence 2":
        app_slug = "supervisely-ecosystem/serve-florence-2"
    elif model_name == "Kosmos 2":
        app_slug = "supervisely-ecosystem/serve-kosmos-2"
    elif model_name == "Molmo":
        app_slug = "supervisely-ecosystem/serve-molmo"
    elif model_name == "SAM 2":
        app_slug = "supervisely-ecosystem/serve-segment-anything-2"
    else:
        supported_models = get_supported_models()
        raise ValueError(
            f"Model {model_name} is not supported, here is list of supported models: {supported_models}"
        )

    supported_checkpoints = get_supported_checkpoints(model_name)
    if checkpoint_name not in supported_checkpoints:
        raise ValueError(
            f"Checkpoint {checkpoint_name} is not supported, here is list of supported checkpoints: {supported_checkpoints}"
        )
    # get app module id
    module_id = api.app.get_ecosystem_module_id(app_slug)

    # start app session
    sly.logger.info("Starting app session...")
    nn_session_info = api.app.start(
        agent_id=agent_id,
        module_id=module_id,
        workspace_id=workspace_id,
        task_name=model_name,
        params={},
    )
    task_id = nn_session_info.task_id

    # wait for the app to start
    is_ready = api.app.is_ready_for_api_calls(task_id)
    if not is_ready:
        api.app.wait_until_ready_for_api_calls(task_id, attempts=100000)
    time.sleep(10)  # still need a time after status changed
    sly.logger.info(f"{model_name} app session started, waiting for the model to deploy...")

    # prepare deploy params
    if model_name in ["Grounding DINO", "Kosmos 2", "Molmo"]:
        checkpoint = "/weights/" + checkpoint_name
    elif model_name == "Florence 2":
        checkpoint = "microsoft/" + checkpoint_name
    elif model_name == "SAM 2":
        checkpoint = "/sam2.1_weights/" + checkpoint_name
    if model_name != "SAM 2":
        model_info = {
            "meta": {
                "model_name": model_name,
                "model_files": {"checkpoint": checkpoint},
            }
        }
        deploy_params = {
            "deploy_params": {
                "model_files": {
                    "checkpoint": checkpoint,
                },
                "model_source": ModelSource.PRETRAINED,
                "device": "cuda",
                "runtime": RuntimeType.PYTORCH,
                "model_info": model_info,
            }
        }
    else:
        deploy_params = {
            "deploy_params": {
                "device": "cuda",
                "from_api": True,
                "model_source": additional_params.get("model_source", "Pretrained models"),
                "weights_path": checkpoint,
                "config": additional_params.get("config"),
                "custom_link": additional_params.get("custom_link"),
            }
        }
    # send deploy request
    api.app.send_request(
        nn_session_info.task_id,
        "deploy_from_api",
        deploy_params,
    )
    nn_session = sly.nn.inference.Session(api, task_id=task_id)
    sly.logger.info(f"Successfully deployed {model_name} model")
    return nn_session


def object_detection(
    project_id: int,
    dataset_ids: List[int],
    nn_session: sly.nn.inference.Session,
    api: sly.Api,
    inference_settings: dict = None,
) -> List[sly.Annotation]:
    """
    Applies object detection foundation model to images project
    """
    # get image ids
    image_ids = []
    for dataset_id in dataset_ids:
        ds_image_infos = api.image.get_list(dataset_id)
        ds_image_ids = [image_info.id for image_info in ds_image_infos]
        image_ids.extend(ds_image_ids)
    # split image ids into batches
    batch_size = inference_settings.get("batch_size", 2)
    image_ids_batched = [
        image_ids[i : i + batch_size] for i in range(0, len(image_ids), batch_size)
    ]
    # set inference settings
    if inference_settings and inference_settings.get("batch_size"):
        del inference_settings["batch_size"]
    nn_session.set_inference_settings(inference_settings)
    # get project meta
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    # apply model to image batches
    anns = []
    for image_ids_batch in tqdm(
        image_ids_batched, desc="Labeling image batches with bounding boxes..."
    ):
        # get batch annotation
        batch_anns = nn_session.inference_image_ids(image_ids_batch)
        # add new obj classes and tags to current project meta
        for ann in batch_anns:
            for label in ann.labels:
                if not project_meta.get_obj_class(label.obj_class.name):
                    project_meta = project_meta.add_obj_class(label.obj_class)
                    api.project.update_meta(project_id, project_meta.to_json())
                for tag in label.tags:
                    if not project_meta.get_tag_meta(tag.name):
                        project_meta = project_meta.add_tag_meta(tag.meta)
                        api.project.update_meta(project_id, project_meta.to_json())
        # upload annotation to the platform
        api.annotation.upload_anns(image_ids_batch, batch_anns)
        anns.extend(batch_anns)
    sly.logger.info("Successfully finished prelabeling process for object detection task")
    return anns


def object_pointing(
    project_id: int,
    dataset_ids: List[int],
    nn_session: sly.nn.inference.Session,
    api: sly.Api,
    inference_settings: dict = None,
) -> List[sly.Annotation]:
    """
    Applies object pointing foundation model to images project
    """
    # get image ids
    image_ids = []
    for dataset_id in dataset_ids:
        ds_image_infos = api.image.get_list(dataset_id)
        ds_image_ids = [image_info.id for image_info in ds_image_infos]
        image_ids.extend(ds_image_ids)
    # set inference settings
    if inference_settings:
        nn_session.set_inference_settings(inference_settings)
    # add point class to if necessary project meta
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    if not project_meta.get_obj_class("point"):
        project_meta = project_meta.add_obj_class(sly.ObjClass("point", sly.Point))
        api.project.update_meta(project_id, project_meta.to_json())
    # apply model to images
    anns = []
    for image_id in tqdm(image_ids, desc="Labeling images with points..."):
        # get image annotation
        ann = nn_session.inference_image_id(image_id)
        # upload annotation to the platform
        api.annotation.upload_ann(image_id, ann)
        anns.append(ann)
    sly.logger.info("Successfully finished prelabeling process for object pointing task")
    return anns


def instance_segmentation(
    project_id: int,
    dataset_ids: List[int],
    nn_session: sly.nn.inference.Session,
    api: sly.Api,
    inference_settings: dict,
) -> List[sly.Annotation]:
    """
    Applies instance segmentation foundation model to images project
    """
    # get project meta
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    # get image ids and annotations (for prompts)
    image_ids = []
    download_anns = False
    if (
        inference_settings.get("mode") == "bbox" and not inference_settings.get("bbox_predictions")
    ) or (
        inference_settings.get("mode") == "points"
        and not inference_settings.get("point_predictions")
    ):
        download_anns = True
        image_anns = []
    else:
        if inference_settings.get("mode") == "bbox":
            image_anns = inference_settings.get("bbox_predictions")
        elif inference_settings.get("mode") == "points":
            image_anns = inference_settings.get("point_predictions")
    for dataset_id in dataset_ids:
        ds_image_infos = api.image.get_list(dataset_id)
        ds_image_ids = [image_info.id for image_info in ds_image_infos]
        image_ids.extend(ds_image_ids)
        if download_anns:
            ds_image_ann_infos = api.annotation.download_batch(dataset_id, ds_image_ids)
            ds_image_anns = [
                sly.Annotation.from_json(info.annotation, project_meta)
                for info in ds_image_ann_infos
            ]
            image_anns.extend(ds_image_anns)
    # apply model to images
    anns = []
    for idx, image_id in enumerate(tqdm(image_ids, desc="Labeling images with masks...")):
        # get prompt annotation
        prompt_ann = image_anns[idx]
        total_image_ann = None
        for label in prompt_ann.labels:
            found_geometry = False
            if (
                label.geometry.geometry_name() == "rectangle"
                and inference_settings.get("mode") == "bbox"
            ):
                class_name = label.obj_class.name.rstrip("_bbox")
                rectangle = label.geometry.to_json()
                nn_session.set_inference_settings(
                    {
                        "input_image_id": image_id,
                        "mode": "bbox",
                        "rectangle": rectangle,
                        "bbox_class_name": class_name,
                    }
                )
                found_geometry = True
            elif (
                label.geometry.geometry_name() == "point"
                and inference_settings.get("mode") == "points"
            ):
                point_coordinates = [[label.geometry.col, label.geometry.row]]
                point_labels = [1]
                nn_session.set_inference_settings(
                    {
                        "input_image_id": image_id,
                        "mode": "points",
                        "point_coordinates": point_coordinates,
                        "point_labels": point_labels,
                    }
                )
                found_geometry = True
            if found_geometry:
                # get image annotation
                ann = nn_session.inference_image_id(image_id)
                # merge with previous annotations
                if not total_image_ann:
                    ann = prompt_ann.merge(ann)
                    total_image_ann = ann
                else:
                    total_image_ann = total_image_ann.merge(ann)
        if total_image_ann:
            # add new obj classes and tags to current project meta
            for label in total_image_ann.labels:
                if not project_meta.get_obj_class(label.obj_class.name):
                    project_meta = project_meta.add_obj_class(label.obj_class)
                    api.project.update_meta(project_id, project_meta.to_json())
            # upload annotation to the platform
            api.annotation.upload_ann(image_id, total_image_ann)
            anns.append(ann)
    sly.logger.info("Successfully finished prelabeling process for instance segmentation task")
    return anns


def preview(
    nn_session: sly.nn.inference.Session,
    api: sly.Api,
    inference_settings: dict = None,
    dataset_id: int = None,
    image_id: int = None,
) -> np.ndarray:
    # get image id if necessary
    if not image_id:
        if not dataset_id:
            raise ValueError("Either dataset_id or image_id should be provided")
        ds_image_infos = api.image.get_list(dataset_id)
        image_id = random.choice(ds_image_infos).id
    # set inference settings
    if inference_settings:
        nn_session.set_inference_settings(inference_settings)
    # get image annotation
    image_ann = nn_session.inference_image_id(image_id)
    bitmap = api.image.download_np(image_id)
    # draw annotation on image
    image_ann.draw(
        bitmap,
        thickness=3,
        color=[255, 0, 0],
        fill_rectangles=False,
        draw_class_names=True,
    )
    return bitmap
