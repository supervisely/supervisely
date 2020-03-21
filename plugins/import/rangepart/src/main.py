# coding: utf-8

import scipy.io
import os, cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file


classes_dict = sly.ObjClassCollection()
number_class = {1: 'background', 2: 'head', 3: 'body', 4: 'hand', 5: 'lag'}
pixel_color = {1: (0, 0, 255), 2: (0, 128, 0), 3: (255, 255, 0), 4: (128, 255, 0), 5: (255, 0, 0)}


def read_datasets(all_dir):
    src_datasets = {}
    for dir in os.listdir(all_dir):
        sample_names = []
        watching_dir = os.path.join(all_dir, dir)
        for next_dir in os.listdir(watching_dir):
            for file in os.listdir(os.path.join(watching_dir, next_dir)):
                if file.endswith('.mat'):
                    sample_names.append(os.path.splitext(file)[0])
        src_datasets[dir] = sample_names
    sly.logger.info('Found source {} dataset with {} sample(s).'.format(dir, len(sample_names)))
    return src_datasets


def get_ann(img_path, inst_path, number_class, pixel_color):
    global classes_dict
    ann = sly.Annotation.from_img_path(img_path)
    mat = scipy.io.loadmat(inst_path)
    instance_img = mat['MM'][0][0][0]
    instance_img = instance_img.astype(np.uint8) + 1
    colored_img = instance_img
    unique_pixels = np.unique(instance_img)
    for pixel in unique_pixels:
        color = pixel_color[pixel]
        class_name = number_class[pixel]
        imgray = np.where(colored_img == pixel, colored_img, 0)
        ret, thresh = cv2.threshold(imgray, 1, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            arr = np.array(contours[i], dtype=int)
            mask_temp = np.zeros(instance_img.shape, dtype=np.uint8)
            cv2.fillPoly(mask_temp, [arr], (255, 255, 255))
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
    all_dirs = os.path.join(sly.TaskPaths.DATA_DIR, 'RANGE')
    src_datasets = read_datasets(all_dirs)
    for ds_name, sample_names in src_datasets.items():
        ds = out_project.create_dataset(ds_name) #make train -> img, ann
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger
        for name in sample_names:
            subdir = os.path.join(all_dirs, ds_name)
            img_foto = os.path.join(subdir, 'd_images')
            img_mat = os.path.join(subdir, 'd_masks')
            src_img_path = os.path.join(img_foto, name + '.jpg')
            inst_path = os.path.join(img_mat, name + '.mat')
            if all((os.path.isfile(x) or (x is None) for x in [src_img_path, inst_path])):
                ann = get_ann(src_img_path, inst_path, number_class, pixel_color)
                ds.add_item_file(name, src_img_path, ann=ann)
            progress.iter_done_report()
    out_meta = sly.ProjectMeta(obj_classes=classes_dict)
    out_project.set_meta(out_meta)


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
  sly.main_wrapper('RangePart', main)

