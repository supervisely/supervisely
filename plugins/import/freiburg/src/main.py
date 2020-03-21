# coding: utf-8

import scipy.io
import os, cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file


classes_dict = sly.ObjClassCollection()
number_class = {1: 'background', 2: 'part1', 3: 'part2', 4: 'part3', 5: 'part4', 6: 'part5', 7: 'part6', 8: 'part7', 9: 'part8', 10: 'part9', 11: 'part10', 12: 'part11', 13: 'part12', 14: 'part13', 15: 'part14'}
pixel_color = {1: (139, 251, 246), 2: (229, 39, 123), 3: (239, 248, 31), 4: (39, 49, 229), 5: (6, 207, 60), 6: (248, 193, 171), 7: (107, 102, 2), 8: (118, 138, 128), 9: (184, 54, 246), 10: (78, 20, 126), 11: (31, 230, 187), 12: (117, 216, 34), 13: (234, 103, 8), 14: (151, 1, 64), 15: (141, 247, 139)}


def read_datasets(all_mat):
    src_datasets = {}
    sample_names = []
    for file in os.listdir(all_mat):
        if file.endswith('.mat'):
            sample_names.append(os.path.splitext(file)[0])
        src_datasets['dataset'] = sample_names
    sly.logger.info('Found source dataset with {} sample(s).'.format(len(sample_names)))
    return src_datasets


def get_ann(img_path, inst_path, number_class, pixel_color):
    global classes_dict
    ann = sly.Annotation.from_img_path(img_path)
    mat = scipy.io.loadmat(inst_path)
    instance_img = mat['M']
    instance_img = instance_img.astype(np.uint8) + 1
    colored_img = instance_img
    unique_pixels = np.unique(instance_img)
    for pixel in unique_pixels:
        color = pixel_color[pixel]
        class_name = number_class[pixel]
        if pixel == 1:
            continue
        imgray = np.where(colored_img == pixel, colored_img, 0)
        ret, thresh = cv2.threshold(imgray, 1, 255, 0)
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
    sly.fs.clean_dir(sly.TaskPaths.RESULTS_DIR)
    settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']),
                              sly.OpenMode.CREATE)
    all_img = os.path.join(sly.TaskPaths.DATA_DIR, 'Sitting/img')
    all_mat = os.path.join(sly.TaskPaths.DATA_DIR, 'Sitting/masks')
    src_datasets = read_datasets(all_mat)
    for ds_name, sample_names in src_datasets.items():
        ds = out_project.create_dataset(ds_name) #make train -> img, ann
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger
        for name in sample_names:
            src_img_path = os.path.join(all_img, name + '.jpg')
            inst_path = os.path.join(all_mat, name + '.mat')
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
  sly.main_wrapper('freiburg_sitting', main)

