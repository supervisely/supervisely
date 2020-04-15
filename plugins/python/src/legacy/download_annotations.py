import json
import os
import supervisely_lib as sly

WORKSPACE_ID = int('%%WORKSPACE_ID%%')
src_project_name = '%%IN_PROJECT_NAME%%'
src_dataset_ids = %%DATASET_IDS:None%%

api = sly.Api(server_address=os.environ['SERVER_ADDRESS'], token=os.environ['API_TOKEN'])

#### End settings. ####

project = api.project.get_info_by_name(WORKSPACE_ID, src_project_name)
if project is None:
    raise RuntimeError('Project {!r} not found'.format(src_project_name))

dest_dir = os.path.join(sly.TaskPaths.OUT_ARTIFACTS_DIR, src_project_name)
sly.fs.mkdir(dest_dir)

meta_json = api.project.get_meta(project.id)
sly.io.json.dump_json_file(meta_json, os.path.join(dest_dir, 'meta.json'))

total_images = 0
src_dataset_infos = (
    [api.dataset.get_info_by_id(ds_id) for ds_id in src_dataset_ids] if (src_dataset_ids is not None)
    else api.dataset.get_list(project.id))

for dataset in src_dataset_infos:
    ann_dir = os.path.join(dest_dir, dataset.name, 'ann')
    sly.fs.mkdir(ann_dir)

    images = api.image.get_list(dataset.id)
    ds_progress = sly.Progress(
        'Downloading annotations for: {!r}/{!r}'.format(src_project_name, dataset.name),
        total_cnt=len(images))
    for batch in sly.batched(images):
        image_ids = [image_info.id for image_info in batch]
        image_names = [image_info.name for image_info in batch]

        #download annotations in json format
        ann_infos = api.annotation.download_batch(dataset.id, image_ids)
        ann_jsons = [ann_info.annotation for ann_info in ann_infos]

        for image_name, ann_info in zip(image_names, ann_infos):
            sly.io.json.dump_json_file(ann_info.annotation, os.path.join(ann_dir, image_name + '.json'))
        ds_progress.iters_done_report(len(batch))
        total_images += len(batch)

sly.logger.info('Project {!r} has been successfully downloaded'.format(src_project_name))
sly.logger.info('Total number of images: {!r}'.format(total_images))
