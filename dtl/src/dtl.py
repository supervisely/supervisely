# coding: utf-8

import os

import supervisely_lib as sly
from supervisely_lib import logger, EventType

from Net import Net


def check_in_graph():
    helper = sly.DtlHelper()
    net = Net(helper.graph, helper.in_project_metas, helper.paths.results_dir)

    # to ensure validation
    _ = net.get_result_project_meta()
    _ = net.get_final_project_name()
    is_archive = net.is_archive()

    need_download = net.may_require_images()
    return {'download_images': need_download, 'is_archive': is_archive}


def main():
    sly.task_verification(check_in_graph)

    logger.info('DTL started')
    helper = sly.DtlHelper()
    net = Net(helper.graph, helper.in_project_metas, helper.paths.results_dir)
    helper.save_res_meta(net.get_result_project_meta())

    # is_archive = net.is_archive()
    results_counter = 0
    for pr_name, pr_dir in helper.in_project_dirs.items():
        root_path, project_name = sly.ProjectFS.split_dir_project(pr_dir)
        project_fs = sly.ProjectFS.from_disk(root_path, project_name, by_annotations=True)
        progress = sly.progress_counter_dtl(pr_name, project_fs.image_cnt)
        for sample in project_fs:
            try:
                img_desc = sly.ImageDescriptor(sample)
                ann = sly.json_load(sample.ann_path)
                data_el = (img_desc, ann)
                export_output_generator = net.start(data_el)
                for res_export in export_output_generator:
                    logger.trace("image processed", extra={'img_name': res_export[0][0].get_img_name()})
                    results_counter += 1
            except Exception:
                ex = {
                    'project_name': sample.project_name,
                    'ds_name': sample.ds_name,
                    'image_name': sample.image_name
                }
                logger.warn('Image was skipped because some error occured', exc_info=True, extra=ex)
            progress.iter_done_report()

    logger.info('DTL finished', extra={'event_type': EventType.DTL_APPLIED, 'new_proj_size': results_counter})


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(logger, sly.DtlPaths().debug_dir)
    sly.main_wrapper('DTL', main)
