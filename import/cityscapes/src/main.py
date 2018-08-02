# coding: utf-8

import os
import os.path as osp
import glob

import supervisely_lib as sly
from supervisely_lib import logger


class AnnConvException(Exception):
    pass


class ImporterCityscapes:
    def __init__(self):
        task_paths = sly.DtlPaths()
        self.in_dir = task_paths.data_dir
        self.out_dir = task_paths.results_dir
        self.settings = sly.json_load(task_paths.settings_path)

        self.classes = set()
        self.tags = set()

    @classmethod
    def json_path_to_image_path(cls, json_path):
        img_path = json_path.replace('/gtFine/', '/leftImg8bit/')
        img_path = img_path.replace('_gtFine_polygons.json', '_leftImg8bit.png')
        return img_path

    def _load_citysc_annotation(self, ann_path):
        json_data = sly.json_load(ann_path)
        imsize_wh = json_data['imgWidth'], json_data['imgHeight']
        figures = []
        for obj in json_data['objects']:
            cls = obj['label']
            if cls == 'out of roi':
                polygon = obj['polygon'][:5]
                interiors = [obj['polygon'][5:]]
            else:
                polygon = obj['polygon']
                interiors = []

            self.classes.add(cls)

            to_add = sly.FigurePolygon.from_np_points(cls, imsize_wh, polygon, interiors)
            figures.extend(to_add)

        ann = sly.Annotation.new_with_objects(imsize_wh, figures)
        return ann

    def _convert_sample(self, sample_info):
        sample_data = sample_info.ia_data

        try:
            ann = self._load_citysc_annotation(sample_data['orig_ann_path'])
            ann['tags'].append(sample_data['tag_name'])
            packed_ann = ann.pack()
        except Exception:
            raise AnnConvException()  # ok, may continue work with another sample

        self.tags.add(sample_data['tag_name'])
        sly.json_dump(packed_ann, sample_info.ann_path)  # ann is ready
        sly.copy_file(sample_data['orig_img_path'], sample_info.img_path)  # img is ready

    def convert(self):
        search_fine = osp.join(self.in_dir, "gtFine", "*", "*", "*_gt*_polygons.json")
        files_fine = glob.glob(search_fine)
        files_fine.sort()

        search_imgs = osp.join(self.in_dir, "leftImg8bit", "*", "*", "*_leftImg8bit.png")
        files_imgs = glob.glob(search_imgs)
        files_imgs.sort()

        out_pr = sly.ProjectStructure(self.settings['res_names']['project'])

        for orig_ann_path in files_fine:
            parent_dir, json_fname = osp.split(os.path.abspath(orig_ann_path))
            dataset_name = osp.basename(parent_dir)
            sample_name = json_fname.replace('_gtFine_polygons.json', '')

            orig_img_path = self.json_path_to_image_path(orig_ann_path)

            tag_path = osp.split(parent_dir)[0]
            tag_name = osp.basename(tag_path)  # e.g. train, val, test

            dt = {
                'orig_ann_path': orig_ann_path,
                'orig_img_path': orig_img_path,
                'tag_name': tag_name,
                'image_ext': '.png',  # hard-coded ext (see glob above)
            }

            if all(osp.isfile(x) for x in (orig_img_path, orig_ann_path)):
                out_pr.add_item(dataset_name, sample_name, dt)

        stat_dct = {'samples': out_pr.image_cnt, 'src_ann_cnt': len(files_fine), 'src_img_cnt': len(files_imgs)}
        logger.info('Found img/ann pairs.', extra=stat_dct)
        if stat_dct['samples'] < stat_dct['src_ann_cnt']:
            logger.warn('Found source annotations without corresponding images.', extra=stat_dct)

        out_pr_fs = sly.ProjectFS(self.out_dir, out_pr)
        out_pr_fs.make_dirs()

        progress = sly.progress_counter_import(out_pr.name, out_pr.image_cnt)
        ok_cnt = 0
        for s in out_pr_fs:
            log_dct = s._asdict()  # ok, it's documented
            try:
                self._convert_sample(s)
            except AnnConvException:
                logger.warn('Error occurred while processing input sample annotation.', exc_info=True, extra=log_dct)
            except Exception:
                logger.error('Error occurred while processing input sample.', exc_info=False, extra=log_dct)
                raise
            else:
                ok_cnt += 1
            progress.iter_done_report()

        logger.info('Processed.', extra={'samples': out_pr.image_cnt, 'ok_cnt': ok_cnt})

        res_meta = sly.ProjectMeta()
        for class_name in self.classes:
            res_meta.classes.add({'title': class_name, 'shape': 'polygon', 'color': sly.gen_new_color()})
        res_meta.img_tags.update(self.tags)
        res_meta.to_dir(out_pr_fs.project_path)
        logger.info('Found classes.', extra={'cnt': len(self.classes), 'classes': sorted(list(self.classes))})
        logger.info('Created tags.', extra={'cnt': len(self.tags), 'tags': sorted(list(self.tags))})


def main():
    importer = ImporterCityscapes()
    importer.convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('CITYSCAPES_IMPORT', main)
