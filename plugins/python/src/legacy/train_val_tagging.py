import os
import supervisely_lib as sly
from supervisely_lib.nn.dataset import partition_train_val

WORKSPACE_ID = int('%%WORKSPACE_ID%%')
src_project_name = '%%IN_PROJECT_NAME%%'
src_dataset_ids = %%DATASET_IDS:None%%
dst_project_name = '%%OUT_PROJECT_NAME%%'

api = sly.Api(server_address=os.environ['SERVER_ADDRESS'], token=os.environ['API_TOKEN'])

# Which fraction of images to tag as a validation set (remaining are tagged as training set).
validation_fraction = float('%%validation_fraction:0.1%%')

train_tag_name = '%%train_tag_name:train%%'
val_tag_name = '%%val_tag_name:val%%'

#### End settings. ####

src_project = api.project.get_info_by_name(WORKSPACE_ID, src_project_name)
src_meta_json = api.project.get_meta(src_project.id)
src_meta = sly.ProjectMeta.from_json(src_meta_json)

for tag_name in [train_tag_name, val_tag_name]:
    if src_meta.tag_metas.has_key(tag_name):
        raise RuntimeError('Project {!r} already contains tag {!r}. For train/validation tagging, we do not support '
                           'existing tags to avoid interactions with your existing tagging. Please specify unique '
                           'train and validation tag names that are not yet present in the project.'.format(
                                src_project_name, tag_name))

tag_meta_train = sly.TagMeta(train_tag_name, sly.TagValueType.NONE)
tag_meta_val = sly.TagMeta(val_tag_name, sly.TagValueType.NONE)
dst_meta = src_meta.add_tag_metas([tag_meta_train, tag_meta_val])

# Will choose a new name if dst_project_name is already taken.
dst_project = api.project.create(WORKSPACE_ID, dst_project_name, change_name_if_conflict=True)
api.project.update_meta(dst_project.id, dst_meta.to_json())

src_dataset_infos = (
    [api.dataset.get_info_by_id(ds_id) for ds_id in src_dataset_ids] if (src_dataset_ids is not None)
    else api.dataset.get_list(src_project.id))
total_images = sum(ds_info.images_count for ds_info in src_dataset_infos)

if total_images <= 1:
    raise RuntimeError('Need at least 2 images in a project to prepare a training set (at least 1 each for training '
                       'and validation).')
is_train_image = partition_train_val(total_images, validation_fraction)

batch_start_idx = 0
for src_dataset in src_dataset_infos:
    dst_dataset = api.dataset.create(dst_project.id, src_dataset.name, src_dataset.description)
    images = api.image.get_list(src_dataset.id)
    ds_progress = sly.Progress(
        'Tagging dataset: {!r}/{!r}'.format(src_project.name, src_dataset.name), total_cnt=len(images))
    for batch in sly.batched(images):
        image_ids = [image_info.id for image_info in batch]
        image_names = [image_info.name for image_info in batch]

        ann_infos = api.annotation.download_batch(src_dataset.id, image_ids)
        src_anns = [sly.Annotation.from_json(ann_info.annotation, dst_meta) for ann_info in ann_infos]
        anns_tagged = [ann.add_tag(sly.Tag(tag_meta_train) if is_train_image[image_idx] else sly.Tag(tag_meta_val))
                       for image_idx, ann in enumerate(src_anns, start=batch_start_idx)]
        anns_tagged_jsons = [ann.to_json() for ann in anns_tagged]

        dst_images = api.image.upload_ids(dst_dataset.id, image_names, image_ids)
        dst_image_ids = [dst_img_info.id for dst_img_info in dst_images]
        api.annotation.upload_jsons(dst_image_ids, anns_tagged_jsons)

        ds_progress.iters_done_report(len(batch))
        batch_start_idx += len(batch)

sly.logger.info('Project {!r} train/val tagging done. Result project: {!r}'.format(src_project.name, dst_project_name))
