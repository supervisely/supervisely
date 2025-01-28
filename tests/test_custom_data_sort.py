import asyncio
import os

import supervisely as sly

api = sly.Api.from_env()
from typing import Any, Dict, List, NamedTuple, Optional

from supervisely.api.api import ApiField
from supervisely.api.image_api import ImageApi
from supervisely.project.download import download_async_or_sync
from supervisely.project.upload import upload

workspace_id = 210
save_dir = os.path.expanduser("~/Work/test_images_download/")
# image_path = os.path.expanduser("~/Work/test_images_download/rmtns.jpg")
# image_path2 = os.path.expanduser("~/Work/test_images_download/rmtns2.jpg")
# m1_image_path = os.path.expanduser("~/Work/test_images_download/image-000001.dcm")
# m2_image_path = os.path.expanduser("~/Work/test_images_download/image-000002.dcm")
# image_np = sly.image.read(image_path)
# link = "https://images.pexels.com/photos/1114688/pexels-photo-1114688.jpeg"
# meta_1 = {"test": "test works 1"}
# meta_2 = {"test": "test works 2"}
# meta_3 = {"test": "test works 3"}
# meta_4 = {"test": "test works 4"}
# project_path = os.path.expanduser("~/test_project_download/project_for_upload")
# project_path = os.path.expanduser("~/test_project_download/ds1")
# project_path = os.path.expanduser("~/test_project_download/pg1")
# project_path = os.path.expanduser("~/test_project_download/project_for_upload_only_infos")

# info_generator = api.image.get_list_generator(1592, sort=ApiField.CUSTOM_SORT)
# gen_list = []
# for i in info_generator:
#     gen_list.extend(i)
# info_list_1 = api.image.get_list(1592, sort=ApiField.CUSTOM_SORT)
# info_list_2 = api.image.get_filtered_list(
#     1592,
#     sort=ApiField.CUSTOM_SORT,
#     filters=[{"type": "images_filename", "data": {"value": "img_0001.jpg"}}],
# )
# info_list_4 = api.image.upload_paths(
#         1592,
#         ["img_0001.jpg", "img_0002.jpg"],
#         [image_path, image_path2],
#         metas=[
#             {"customSort": "img_0001", "test": "test works 1", "image_path1": image_path},
#             {"slyCustomSort": "img_0002", "image_path2": image_path2},
#         ],
#         conflict_resolution="rename",
#     )
# with api.image.add_custom_sort("test"):
# info_list_4 = api.image.upload_paths(
#     1592,
#     ["img_0001.jpg", "img_0002.jpg"],
#     [image_path, image_path2],
#     metas=[
#         {"customSort": "img_0001", "test": "test works 1", "image_path1": image_path},
#         {"slyCustomSort": "img_0002", "image_path2": image_path2},
#     ],
#     conflict_resolution="rename",
# )
# api.image.set_custom_sort(info_list_4[0].id, "setted")
# info_2 = api.image.get_info_by_id(info_list_4[0].id)

# info_3 = api.image.upload_np(1592, "img_00041.jpg", image_np, meta=meta_1)
# info_4 = api.image.upload_link(1592, "img_00042.jpg", link=link, meta=meta_2)
# info_5 = api.image.upload_hash(1592, "img_00043.jpg", info_3.hash, meta=meta_3)
# info_6 = api.image.upload_id(1592, "img_00044.jpg", info_3.id, meta=meta_4)
# info_list_5 = api.image.upload_multiview_images(
#     1595, group_name="test", paths=[image_path, image_path2], metas=[meta_1, meta_2]
# )
# info_list_6 = api.image.upload_medical_images(
#     1596, paths=[m1_image_path, m2_image_path], metas=[meta_3, meta_4]
# )
# upload(project_path, api, workspace_id)

# api.image.set_custom_sort_bulk(
#     [info_list_4[1].id, info_list_4[0].id], ["setted_bulk1", "setted_bulk0"]
# )

# imgs_async_gen = []


# async def test_gen():
#     all_images = api.image.get_list_generator_async(1592, sort=ApiField.CUSTOM_SORT)
#     async for image_batch in all_images:
#         imgs_async_gen.extend(image_batch)


# asyncio.run(test_gen())
# for i in imgs_async_gen:
#     print(i.meta.get(ApiField.CUSTOM_SORT, None))
# img_info = api.image.get_info_by_id(231924)
# proj_meta = api.project.get_meta(45)
# meta = sly.ProjectMeta.from_json(proj_meta)
# brick = meta.get_obj_class("brick")
# label = sly.Label(sly.GraphNodes({}), brick)
# ann = sly.Annotation(img_size=[img_info.height, img_info.width], labels=[label])
# api.annotation.upload_ann(231924, ann)
# sly.fs.clean_dir(save_dir)
# download_async_or_sync(api, 11, save_dir)

meta = api.project.get_meta(11)
meta = sly.ProjectMeta.from_json(meta)
obc_class = meta.get_obj_class("graph")
geometry = sly.GraphNodes({})
label = sly.Label(geometry=geometry, obj_class=obc_class)
ann = sly.Annotation([500, 500])
ann._add_labels_impl(dest=ann.labels, labels=[label])
geometry.to_bbox()
print("done")
