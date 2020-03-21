# coding: utf-8

import os, cv2
import numpy as np

import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file

classes_dict = sly.ObjClassCollection()


def read_datasets(all_ann):
    src_datasets = {}
    if not os.path.isdir(all_ann):
        raise RuntimeError('There is no directory {}, but it is necessary'.format(all_ann))

    sample_names = []
    for file in os.listdir(all_ann):
        if file.endswith('.txt'):
            sample_names.append(os.path.splitext(file)[0])
            src_datasets['dataset'] = sample_names
    sly.logger.info('Found source dataset with {} sample(s).'.format(len(sample_names)))
    return src_datasets


def get_ann(img_path, inst_path, number_class, pixel_color):
    global classes_dict
    instance_img = []
    with open(inst_path) as file:
        for line in file:
            line = line.split('\n')[0]
            line = line.split(' ')
            instance_img.append(line)
    instance_img = np.array(instance_img, np.uint8)
    instance_img = instance_img + 2
    curr_color_to_class = {}
    temp = np.unique(instance_img)
    for pixel in temp:
        if pixel == 255:
            continue
        curr_color_to_class[pixel] = number_class[pixel]

    ann = sly.Annotation.from_img_path(img_path)

    for color, class_name in curr_color_to_class.items():
        new_color = list(pixel_color[color])
        mask = np.where(instance_img == color, instance_img, 0)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            arr = np.array(contours[i], dtype=int)
            mask_temp = np.zeros(instance_img.shape, dtype=np.uint8)
            cv2.fillPoly(mask_temp, [np.int32(arr)], (254, 254, 254))
            mask = mask_temp.astype(np.bool)
            bitmap = sly.Bitmap(data=mask)

            if not classes_dict.has_key(class_name):
                obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=new_color)
                classes_dict = classes_dict.add(obj_class)
            ann = ann.add_label(sly.Label(bitmap, classes_dict.get(class_name)))
    return ann


def convert():
    sly.fs.clean_dir(sly.TaskPaths.RESULTS_DIR)
    settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
    all_img = os.path.join(sly.TaskPaths.DATA_DIR, 'ParisArtDecoFacadesDataset-master/images')
    all_ann = os.path.join(sly.TaskPaths.DATA_DIR, 'ParisArtDecoFacadesDataset-master/labels')
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']), sly.OpenMode.CREATE)
    src_datasets = read_datasets(all_ann)
    number_class = {2: 'Door',
                    3: 'Shop',
                    4: 'Balcony',
                    5: 'Window',
                    6: 'Wall',
                    7: 'Sky',
                    8: 'Roof',
                    1: 'Unknown'}

    pixel_color = {2: (255, 255, 0), 3: (0, 128, 0), 4: (0, 0, 255), 5: (128, 255, 0), 6: (255, 0, 0), 7: (0, 255, 255),
                   8: (211, 211, 211), 1: (0, 0, 0)}

    for ds_name, sample_names in src_datasets.items():
        ds = out_project.create_dataset(ds_name) #make train -> img, ann
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger
        for name in sample_names:
            src_img_path = os.path.join(all_img, name + '.png')
            inst_path = os.path.join(all_ann, name + '.txt')

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
    sly.main_wrapper('ParisArt', main)
