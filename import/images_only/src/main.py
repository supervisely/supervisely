# coding: utf-8

import os.path as osp

import pyexiv2
from PIL import Image
import supervisely_lib as sly
from supervisely_lib import logger


class ImporterImagesOnly:
    def __init__(self):
        task_paths = sly.DtlPaths()
        self.in_dir = task_paths.data_dir
        self.out_dir = task_paths.results_dir
        self.settings = sly.json_load(task_paths.settings_path)

    def _find_in_datasets(self):

        def flat_images_only(dir_path):
            subdirs = sly.get_subdirs(dir_path)
            img_fnames = sly.ImportImgLister.list_images(dir_path)
            res = len(subdirs) == 0 and len(img_fnames) > 0
            return res

        def collect_ds_names(pr_path):
            subdirs = sly.get_subdirs(pr_path)
            subd_paths = [osp.join(pr_path, x) for x in subdirs]
            if not all(flat_images_only(x) for x in subd_paths):
                return None
            res = list(zip(subdirs, subd_paths))
            return res

        if flat_images_only(self.in_dir):
            logger.info('Input structure: flat set of images.')
            return [('ds', self.in_dir), ]

        in_datasets = collect_ds_names(self.in_dir)
        if in_datasets:
            logger.info('Input structure: set of dirs (datasets).', extra={'ds_cnt': len(in_datasets)})
            return in_datasets

        top_subdirs = sly.get_subdirs(self.in_dir)
        if len(top_subdirs) == 1:
            new_in_dir = osp.join(self.in_dir, top_subdirs[0])
            in_datasets = collect_ds_names(new_in_dir)
            if in_datasets:
                logger.info('Input structure: dir with set of dirs (datasets).', extra={'ds_cnt': len(in_datasets)})
                return in_datasets

        raise RuntimeError('Unable to determine structure of input directory.')

    def convert(self):
        in_datasets = self._find_in_datasets()

        # map input structure to output
        out_pr = sly.ProjectStructure(self.settings['res_names']['project'])

        for ds_name, ds_path in in_datasets:
            img_fnames = sly.ImportImgLister.list_images(ds_path)
            for name_with_ext in img_fnames:
                img_name, img_ext = osp.splitext(name_with_ext)
                src_img_path = osp.join(ds_path, name_with_ext)
                dt = {
                    'src_img_path': src_img_path,
                    'image_ext': img_ext,
                }
                out_pr.add_item(ds_name, img_name, dt)
            logger.info('Found source dataset with raw images: "{}", {} sample(s).'.format(ds_name, len(img_fnames)))

        out_pr_fs = sly.ProjectFS(self.out_dir, out_pr)
        out_pr_fs.make_dirs()

        res_meta = sly.ProjectMeta()  # empty
        res_meta.to_dir(out_pr_fs.project_path)

        progress = sly.progress_counter_import(out_pr.name, out_pr.image_cnt)
        for sample_info in out_pr_fs:
            sample_data = sample_info.ia_data
            src_img_path = sample_data['src_img_path']
            sly.copy_file(src_img_path, sample_info.img_path)  # img is ready

            image = Image.open(sample_info.img_path)
            exif_data = pyexiv2.metadata.ImageMetadata(sample_info.img_path)
            exif_data.read()

            if exif_data.get_orientation() != 1:
                logger.debug('Image with flip/rot EXIF', extra={'orientation': exif_data.get_orientation(),
                                                                'image_path': sample_info.img_path})
                image = sly.image_transpose_exif(image)
                image.save(sample_info.img_path)
                exif_data['Exif.Image.Orientation'] = pyexiv2.ExifTag('Exif.Image.Orientation', 1)
                exif_data.modified = True
                exif_data.write()

            imsize_wh = image.size
            ann = sly.Annotation.new_with_objects(imsize_wh, [])
            sly.json_dump(ann.pack(), sample_info.ann_path)  # ann is ready
            progress.iter_done_report()


def main():
    importer = ImporterImagesOnly()
    importer.convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('SLY_FORMAT_IMPORT', main)
