# coding: utf-8

import os, cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file

classes_dict = sly.ObjClassCollection()
default_classes_colors = {(0, 0, 0): 'unknown', (255, 0, 0): 'window', (255, 128, 0): 'door', (0, 255, 0): 'ground_floor', (255, 255, 0): 'facade', (128, 255, 255): 'sky', (0, 0, 255): 'roof', (128, 0, 255): 'balkony'}


def read_datasets(all_ann):
    src_datasets = {}
    if not os.path.isdir(all_ann):
        raise RuntimeError('There is no directory {}, but it is necessary'.format(all_ann))
    sample_names = []
    for file in os.listdir(all_ann):
        if file.endswith('.png'):
            sample_names.append(os.path.splitext(file)[0])
            src_datasets['dataset'] = sample_names
    sly.logger.info('Found source dataset with {} sample(s).'.format(len(sample_names)))
    return src_datasets


def get_ann(img_path, inst_path, default_classes_colors):
    global classes_dict
    instance_img = sly.image.read(inst_path)
    colored_img = instance_img
    ann = sly.Annotation.from_img_path(img_path)
    unique_colors = np.unique(instance_img.reshape(-1, instance_img.shape[2]), axis=0)
    for color in unique_colors:
        mask = np.all(colored_img == color, axis=2)
        class_name = default_classes_colors[tuple(color)]
        mask = mask.astype(np.uint8) * 128
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            arr = np.array(contours[i], dtype=int)
            mask_temp = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(mask_temp, [arr], (254, 254, 254))
            mask = mask_temp.astype(np.bool)
            bitmap = sly.Bitmap(data=mask)
            if not classes_dict.has_key(class_name):
                obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color = list(color))
                classes_dict = classes_dict.add(obj_class)
            ann = ann.add_label(sly.Label(bitmap, classes_dict.get(class_name)))
    return ann


def convert():
    sly.fs.clean_dir(sly.TaskPaths.RESULTS_DIR)
    settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
    all_img = os.path.join(sly.TaskPaths.DATA_DIR, 'graz50_facade_dataset/graz50_facade_dataset/images')
    all_ann = os.path.join(sly.TaskPaths.DATA_DIR, 'graz50_facade_dataset/graz50_facade_dataset/labels_full')
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']), sly.OpenMode.CREATE)
    src_datasets = read_datasets(all_ann)
    for ds_name, sample_names in src_datasets.items():
        ds = out_project.create_dataset(ds_name) #make train -> img, ann
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger
        for name in sample_names:
            src_img_path = os.path.join(all_img, name + '.png')
            inst_path = os.path.join(all_ann, name + '.png')

            if all((os.path.isfile(x) or (x is None) for x in [src_img_path, inst_path])):
                ann = get_ann(src_img_path, inst_path, default_classes_colors)
                ds.add_item_file(name, src_img_path, ann=ann)
            progress.iter_done_report()
    out_meta = sly.ProjectMeta(obj_classes=classes_dict)
    out_project.set_meta(out_meta)


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('ParisArt', main)

