import os
import supervisely as sly
from dotenv import load_dotenv
from supervisely.collection.key_indexed_collection import DuplicateKeyError
from supervisely.pointcloud_annotation.pointcloud_episode_tag_collection import (
    PointcloudEpisodeTagCollection,
)
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api.from_env()


PROJECT_ID = sly.env.project_id(False)
DATASET_ID = sly.env.dataset_id()
PROJECT_META_JSON = api.project.get_meta(PROJECT_ID)
PROJECT_META = sly.ProjectMeta.from_json(data=PROJECT_META_JSON)

key_id_map = sly.KeyIdMap()

pcd_entities = api.pointcloud_episode.get_list(DATASET_ID)
pcd_entity_id = pcd_entities[0][0]  # first entity id
pcd_ep_ann_json = api.pointcloud_episode.annotation.download(DATASET_ID)
pcd_ep_ann = sly.PointcloudEpisodeAnnotation.from_json(
    data=pcd_ep_ann_json, project_meta=PROJECT_META, key_id_map=key_id_map
)

new_tag_meta = sly.TagMeta(
    "Tram",
    sly.TagValueType.ONEOF_STRING,
    applicable_to=sly.TagApplicableTo.OBJECTS_ONLY,
    possible_values=["city", "suburb"],
)

project_classes = PROJECT_META.obj_classes
project_tag_metas = PROJECT_META.tag_metas

try:
    new_tags_collection = project_tag_metas.add(new_tag_meta)
    new_project_meta = sly.ProjectMeta(tag_metas=new_tags_collection, obj_classes=project_classes)
    api.project.update_meta(PROJECT_ID, new_project_meta)
except DuplicateKeyError:
    sly.logger.warning(f"New tag ['{new_tag_meta.name}'] already exists in project metadata")
    new_tag_meta = project_tag_metas.get(new_tag_meta.name)


new_tag = sly.PointcloudEpisodeTag(
    meta=new_tag_meta,
    value="suburb",
    frame_range=[12, 13],  # in case you want to add tag to frames
)
new_tag_collection = PointcloudEpisodeTagCollection([new_tag])
new_objects_list = []

for object in pcd_ep_ann.objects:
    # object_tags = object.tags.items() # in case you want to filter objects with the same tag
    # has_this_tag = any(tag.name == new_tag.name for tag in object_tags) # in case you want to filter objects with the same tag
    if object.obj_class.name == "Tram":
        new_obj = object.clone(tags=new_tag_collection)
        new_objects_list.append(new_obj)

new_pcd_ann = pcd_ep_ann.clone(objects=new_objects_list)

api.pointcloud_episode.tag.append_to_objects(
    entity_id=pcd_entity_id,
    project_id=PROJECT_ID,
    objects=new_pcd_ann.objects,
    key_id_map=key_id_map,
)
