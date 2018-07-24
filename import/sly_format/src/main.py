# coding: utf-8

import os
from copy import deepcopy

import supervisely_lib as sly
from supervisely_lib import logger


class ImporterSlyFormat:
    def __init__(self):
        task_paths = sly.DtlPaths()
        self.in_dir = task_paths.data_dir
        self.out_dir = task_paths.results_dir
        self.settings = sly.json_load(task_paths.settings_path)

    def convert(self):
        # determine dir level by project meta file
        meta_p = sly.ProjectMeta.find_in_dir(self.in_dir)
        if meta_p:
            # meta_file in input dir
            in_project_root_dir, pr_name_stub = sly.ProjectFS.split_dir_project(self.in_dir)
        else:
            # meta file in subdir of input dir
            possible_projects = sly.get_subdirs(self.in_dir)
            if len(possible_projects) != 1:
                raise RuntimeError('Wrong input project structure, or multiple projects are passed.')
            in_project_root_dir, pr_name_stub = self.in_dir, possible_projects[0]

        # read if it's possible
        try:
            in_fs = sly.ProjectFS.from_disk(in_project_root_dir, pr_name_stub)
            in_pr_meta = sly.ProjectMeta.from_dir(in_fs.project_path)
        except Exception:
            logger.error('Unable to read input meta.', exc_info=False)
            raise

        in_possible_datasets = sly.get_subdirs(in_fs.project_path)
        in_datasets = list(in_fs.pr_structure.datasets.keys())
        if len(in_possible_datasets) != len(in_datasets):
            raise RuntimeError('Excess top-level directories without data (wrong img-ann structure?).')
        for ds_name, the_ds in in_fs.pr_structure.datasets.items():
            req_cnt = the_ds.image_cnt
            dataset_path = in_fs.dataset_path(ds_name)
            anns_path = in_fs.dataset_anns_path(dataset_path)
            imgs_path = in_fs.dataset_imgs_path(dataset_path)
            for subdir in (anns_path, imgs_path):
                items_cnt = len(list(os.scandir(subdir)))
                if items_cnt != req_cnt:
                    raise RuntimeError('Excess files or directories in dataset subdirectory.')

        found_exts = {x.ia_data['image_ext'] for x in in_fs}
        if not all(x in sly.ImportImgLister.extensions for x in found_exts):
            raise RuntimeError('Found image(s) with unsupported types (by extension).')
        sample_cnt = in_fs.pr_structure.image_cnt
        if sample_cnt == 0:
            raise RuntimeError('Empty project, no samples.')

        logger.info('Found source structure: Supervisely format, {} sample(s).'.format(sample_cnt))

        out_pr = deepcopy(in_fs.pr_structure)
        out_pr.name = self.settings['res_names']['project']
        out_pr_fs = sly.ProjectFS(self.out_dir, out_pr)
        out_pr_fs.make_dirs()

        in_pr_meta.to_dir(out_pr_fs.project_path)

        progress = sly.progress_counter_import(out_pr.name, out_pr.image_cnt)
        for s in out_pr_fs:
            try:
                src_img_path = in_fs.img_path(s.ds_name, s.image_name)
                sly.copy_file(src_img_path, s.img_path)  # img is ready

                src_ann_path = in_fs.ann_path(s.ds_name, s.image_name)
                packed_ann = sly.json_load(src_ann_path)
                _ = sly.Annotation.from_packed(packed_ann, in_pr_meta)  # to check if it is correct
                sly.copy_file(src_ann_path, s.ann_path)  # ann is ready
            except Exception:
                logger.error('Error occured while processing input sample', exc_info=False, extra={
                             'dataset_name': s.ds_name, 'image_name': s.image_name,
                             })
                raise
            progress.iter_done_report()


def main():
    importer = ImporterSlyFormat()
    importer.convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('SLY_FORMAT_IMPORT', main)
