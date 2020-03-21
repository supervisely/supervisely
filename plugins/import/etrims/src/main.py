# coding: utf-8

import os, cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file

classes_dict = sly.ObjClassCollection()
color_to_class = {(0, 0, 0): 'Unknown', (128, 0, 0): 'Building', (128, 0, 128): 'Car', (128, 128, 0): 'Door',
                  (128, 128, 128): 'Pavement', (128, 64, 0): 'Road', (0, 128, 128): 'Sky', (0, 128, 0): 'Vegetation',
                  (0, 0, 128): 'Window'}


def read_datasets(all_ann):
    src_datasets = {}
    if not os.path.isdir(all_ann):
        raise RuntimeError('There is no directory {}, but it is necessary'.format(all_ann))
    for dirname in os.listdir(all_ann):
        sample_names = []
        if os.path.isdir(os.path.join(all_ann, dirname)):
            for file in os.listdir(os.path.join(all_ann, dirname)):
                if file.endswith('.png'):
                    sample_names.append(os.path.splitext(file)[0])
                    src_datasets[dirname] = sample_names
            sly.logger.info('Found source dataset "{}" with {} sample(s).'.format(dirname, len(sample_names)))
    return src_datasets


def get_ann(img_path, inst_path, color_to_class):
    global classes_dict
    ann = sly.Annotation.from_img_path(img_path)
    instance_img = sly.image.read(inst_path)
    colored_img = instance_img
    unique_colors = np.unique(instance_img.reshape(-1, instance_img.shape[2]), axis=0)
    for color in unique_colors:
        class_name = color_to_class[tuple(color)]
        mask = np.all(colored_img == color, axis=2)
        mask = mask.astype(np.uint8) * 128
        ret, thresh = cv2.threshold(mask, 1, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            arr = np.array(contours[i], dtype=int)
            mask_temp = np.zeros(instance_img.shape, dtype=np.uint8)
            cv2.fillPoly(mask_temp, [arr], (255, 255, 255))
            mask_temp = cv2.split(mask_temp)[0]
            mask = mask_temp.astype(np.bool)
            bitmap = sly.Bitmap(data=mask)
            if not classes_dict.has_key(class_name):
                obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=list(color))
                classes_dict = classes_dict.add(obj_class)
            ann = ann.add_label(sly.Label(bitmap, classes_dict.get(class_name)))
    return ann


def convert():
    settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']),
                              sly.OpenMode.CREATE)
    all_img = os.path.join(sly.TaskPaths.DATA_DIR, 'etrims-db_v1/images')
    all_ann = os.path.join(sly.TaskPaths.DATA_DIR, 'etrims-db_v1/annotations')
    src_datasets = read_datasets(all_ann)
    for ds_name, sample_names in src_datasets.items():
        ds = out_project.create_dataset(ds_name) #make train -> img, ann
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger
        imgs_dir = os.path.join(all_img, ds_name)
        inst_dir = os.path.join(all_ann, ds_name)
        for name in sample_names:
            src_img_path = os.path.join(imgs_dir, name + '.jpg')
            inst_path = os.path.join(inst_dir, name + '.png')
            if all((os.path.isfile(x) or (x is None) for x in [src_img_path, inst_path])):
                ann = get_ann(src_img_path, inst_path, color_to_class)
                ds.add_item_file(name, src_img_path, ann=ann)
            progress.iter_done_report()
    out_meta = sly.ProjectMeta(obj_classes=classes_dict)
    out_project.set_meta(out_meta)


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('etrims', main)

