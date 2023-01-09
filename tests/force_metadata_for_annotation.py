import os
from dotenv import load_dotenv
import supervisely as sly


load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()

dataset_id = 54396

dataset_info = api.dataset.get_info_by_id(dataset_id)
project_id = dataset_info.project_id
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

test_class = project_meta.get_obj_class("test-class")
if test_class is None:
    test_class = sly.ObjClass("test-class", sly.Rectangle)
    project_meta = project_meta.add_obj_class(test_class)
    api.project.update_meta(project_id, project_meta)

# case 1
image_id = 18557688
ann_json = api.annotation.download_json(image_id, force_metadata_for_links=False)
ann = sly.Annotation.from_json(ann_json, project_meta)
print(ann.img_size)
ann2 = ann.add_label(sly.Label(sly.Rectangle(0, 0, 10, 10), test_class))
api.annotation.upload_ann(image_id, ann2, skip_bounds_validation=True)


# case 2
image_id = 18557688
ann_info = api.annotation.download(image_id=image_id, force_metadata_for_links=False)
ann_json = ann_info.annotation
ann = sly.Annotation.from_json(ann_json, project_meta)
print(ann.img_size)


# case 0
batch_size = 50

# use bigger batch size for huge project (depends on the number of objects per image)
# select the number that is right for you (compromise between speed / performance / server memory)
# batch_size = 1000 # or 10000 # or 20000
for batch in api.annotation.get_list_generator(
    dataset_id=dataset_id, batch_size=batch_size, force_metadata_for_links=False
):
    for ann_info in batch:
        cur_json = ann_info.annotation
        cur_ann = sly.Annotation.from_json(cur_json, project_meta)
        print(cur_ann.img_size)


# case 1
ann_infos = api.annotation.get_list(dataset_id=dataset_id, force_metadata_for_links=False)
for ann_info in ann_infos:
    cur_json = ann_info.annotation
    cur_ann = sly.Annotation.from_json(cur_json, project_meta)
    print(cur_ann.img_size)
