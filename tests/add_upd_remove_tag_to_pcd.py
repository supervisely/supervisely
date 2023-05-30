import os
import supervisely as sly
from dotenv import load_dotenv

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api.from_env()


project_id = sly.env.project_id()
dataset_id = sly.env.dataset_id()

project_meta_json = api.project.get_meta(project_id)
project_meta = sly.ProjectMeta.from_json(data=project_meta_json)
pcd_ids = api.pointcloud.get_list(dataset_id)

key_id_map = sly.KeyIdMap()

pcd_ann_json = api.pointcloud.annotation.download(pcd_ids[0].id)

tag_name = "pcd"
tag_values = ["old", "new"]

if not project_meta.tag_metas.has_key(tag_name):
    new_tag_meta = sly.TagMeta(
        tag_name,
        sly.TagValueType.ONEOF_STRING,
        applicable_to=sly.TagApplicableTo.IMAGES_ONLY,
        possible_values=tag_values,
    )
    new_tags_collection = project_meta.tag_metas.add(new_tag_meta)
    new_project_meta = sly.ProjectMeta(
        tag_metas=new_tags_collection, obj_classes=project_meta.obj_classes
    )
    api.project.update_meta(project_id, new_project_meta)
    new_prject_meta_json = api.project.get_meta(project_id)
    new_project_meta = sly.ProjectMeta.from_json(data=new_prject_meta_json)
    new_tag_meta = new_project_meta.tag_metas.get(new_tag_meta.name)
else:
    new_tag_meta = project_meta.tag_metas.get(tag_name)
    if sorted(new_tag_meta.possible_values) != sorted(tag_values):
        sly.logger.warning(
            f"Tag [{new_tag_meta.name}] already exists, but with another values: {new_tag_meta.possible_values}"
        )

pcd_tag_id = api.pointcloud.tag.add(new_tag_meta.sly_id, pcd_ids[0].id, value="new")

# stop here to check add
api.pointcloud.tag.update(pcd_tag_id, value="old")

# stop here to check update
api.pointcloud.tag.remove(pcd_tag_id)
