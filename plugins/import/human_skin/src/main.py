# coding: utf-8

import os, cv2
import numpy as np

import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file


classes_dict = sly.ObjClassCollection()

def read_datasets(inst_dir):
    src_datasets = {}
    if not os.path.isdir(inst_dir):
        raise RuntimeError('There is no directory {}, but it is necessary'.format(inst_dir))

    for dirname in os.listdir(inst_dir):
        if os.path.isdir(os.path.join(inst_dir, dirname)):
            sample_names = []
            for file in os.listdir(os.path.join(inst_dir, dirname)):
                if file.endswith('.png'):
                    sample_names.append(os.path.splitext(file)[0])
                    src_datasets[dirname[8:]] = sample_names
            sly.logger.info('Found source dataset "{}" with {} sample(s).'.format(dirname, len(sample_names)))
    return src_datasets


def get_ann(img_path, inst_path, default_classes_colors):
    global classes_dict
    ann = sly.Annotation.from_img_path(img_path)
    instance_img = sly.image.read(inst_path)

    img_gray = cv2.cvtColor(instance_img, cv2.COLOR_BGR2GRAY)
    _, mask_foreground = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    mask_background = (img_gray == 0)

    class_name = 'background'
    new_color = default_classes_colors[class_name]
    bitmap = sly.Bitmap(data=mask_background)

    if not classes_dict.has_key(class_name):
        obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=new_color)
        classes_dict = classes_dict.add(obj_class)
    ann = ann.add_label(sly.Label(bitmap, classes_dict.get(class_name)))

    contours, hierarchy = cv2.findContours(mask_foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    class_name = 'skin'
    new_color = default_classes_colors[class_name]
    for i in range(len(contours)):
        arr = np.array(contours[i], dtype=int)
        mask_temp = np.zeros(instance_img.shape, dtype=np.uint8)
        cv2.fillPoly(mask_temp, [np.int32(arr)], (255, 255, 255))
        mask_temp = cv2.split(mask_temp)[0]
        mask = mask_temp.astype(np.bool)
        bitmap = sly.Bitmap(data=mask)

        if not classes_dict.has_key(class_name):
            obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=new_color)
            classes_dict = classes_dict.add(obj_class)
        ann = ann.add_label(sly.Label(bitmap, classes_dict.get(class_name)))
    return ann


def convert():
    settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
    imgs_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'Pratheepan_Dataset')
    inst_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'Ground_Truth')
    default_classes_colors = {'background': [1, 1, 1], 'skin': [255, 255, 255]}
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']), sly.OpenMode.CREATE)
    src_datasets = read_datasets(inst_dir)
    for ds_name, sample_names in src_datasets.items():
        ds = out_project.create_dataset(ds_name)
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger

        img_dir_temp = os.path.join(imgs_dir, ds_name)
        inst_dir_temp = os.path.join(inst_dir, 'GroundT_' + ds_name)
        for name in sample_names:
            src_img_path = os.path.join(img_dir_temp, name + '.jpg')
            inst_path = os.path.join(inst_dir_temp, name + '.png')

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
  sly.main_wrapper('Human_Skin', main)


