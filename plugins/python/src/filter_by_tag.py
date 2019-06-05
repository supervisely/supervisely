import os
import supervisely_lib as sly

WORKSPACE_ID = %%WORKSPACE_ID%%
src_project_name = '%%IN_PROJECT_NAME%%'
dst_project_name = '%%OUT_PROJECT_NAME%%'

api = sly.Api(server_address=os.environ['SERVER_ADDRESS'], token=os.environ['API_TOKEN'])

# Constants to select filtering mode.
FILTER_OBJECTS = 'objects'
FILTER_IMAGES = 'images'
KNOWN_FILTER_MODES = [FILTER_OBJECTS, FILTER_IMAGES]

filter_mode = '%%filter_mode:images%%'
filtered_tag_name = '%%filtered_tag_name%%'

#### End settings. ####

def _die_unsupported_filter_mode(mode):
    raise ValueError(
        'Unsupported tag filtering mode: {!r}. Only the following methods are supported: {!r}'.format(
            mode, KNOWN_FILTER_MODES))


def _filter_ann_tags(ann, tag_name, mode):
    if mode == FILTER_IMAGES:
        return ann if ann.img_tags.has_key(tag_name) else None
    elif mode == FILTER_OBJECTS:
        filtered_labels = [label for label in ann.labels if label.tags.has_key(tag_name)]
        return ann.clone(labels=filtered_labels)
    else:
        _die_unsupported_filter_mode(mode)


if filter_mode not in KNOWN_FILTER_MODES:
    _die_unsupported_filter_mode(filter_mode)

src_project = api.project.get_info_by_name(WORKSPACE_ID, src_project_name)
src_meta_json = api.project.get_meta(src_project.id)
src_meta = sly.ProjectMeta.from_json(src_meta_json)

# Will choose a new name if dst_project_name is already taken.
dst_project = api.project.create(WORKSPACE_ID, dst_project_name, change_name_if_conflict=True)
api.project.update_meta(dst_project.id, src_meta_json)

for src_dataset in api.dataset.get_list(src_project.id):
    dst_dataset = api.dataset.create(dst_project.id, src_dataset.name, src_dataset.description)
    images = api.image.get_list(src_dataset.id)
    ds_progress = sly.Progress(
        'Filtering dataset: {!r}/{!r}'.format(src_project.name, src_dataset.name), total_cnt=len(images))
    for batch in sly.batched(images):
        image_ids = [image_info.id for image_info in batch]
        image_names = [image_info.name for image_info in batch]

        ann_infos = api.annotation.download_batch(src_dataset.id, image_ids)
        filtered_anns = [
            _filter_ann_tags(sly.Annotation.from_json(ann_info.annotation, src_meta), filtered_tag_name, filter_mode)
            for ann_info in ann_infos]
        filter_passed_indices = [i for i, ann in enumerate(filtered_anns) if ann is not None]

        out_img_ids = [image_ids[i] for i in filter_passed_indices]
        out_img_names = [image_names[i] for i in filter_passed_indices]
        out_ann_jsons = [filtered_anns[i].to_json() for i in filter_passed_indices]

        dst_images = api.image.upload_ids(dst_dataset.id, out_img_names, out_img_ids)

        dst_image_ids = [dst_img_info.id for dst_img_info in dst_images]
        api.annotation.upload_jsons(dst_image_ids, out_ann_jsons)

        ds_progress.iters_done_report(len(batch))

sly.logger.info('Project {!r} filtering done.'.format(src_project.name))