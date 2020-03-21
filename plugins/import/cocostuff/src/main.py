# coding: utf-8

import os, cv2
import numpy as np

from supervisely_lib.imaging.color import generate_rgb
from supervisely_lib.io.json import load_json_file
import supervisely_lib as sly


classes_dict = sly.ObjClassCollection()


def read_datasets(inst_dir):
    src_datasets = {}
    if not os.path.isdir(inst_dir):
        raise RuntimeError('There is no directory {}, but it is necessary'.format(inst_dir))

    for dirname in os.listdir(inst_dir):
        sample_names = []
        if os.path.isdir(os.path.join(inst_dir, dirname)):
            for file in os.listdir(os.path.join(inst_dir, dirname)):
                if file.endswith('.png'):
                    sample_names.append(os.path.splitext(file)[0])
                    src_datasets[dirname] = sample_names
        sly.logger.info('Found source dataset "{}" with {} sample(s).'.format(dirname, len(sample_names)))
    return src_datasets


def read_colors(labels_file_path):
    number_class = {}
    pixel_color = {}
    if os.path.isfile(labels_file_path):
        sly.logger.info('Generate random color mapping.')

        with open(labels_file_path, "r") as file:
            all_lines = file.readlines()
            for line in all_lines:
                line = line.split('\n')[0].split(':')
                temp = int(line[0])
                if temp == 0:
                    temp = 256
                number_class[temp - 1] = (line[1][1:])

        default_classes_colors, colors = {}, [(0, 0, 0)]
        for class_name in number_class.values():
            new_color = generate_rgb(colors)
            colors.append(new_color)
            default_classes_colors[class_name] = new_color

        for i, j in number_class.items():
            pixel_color[i] = default_classes_colors[j]

        cls2col = default_classes_colors
    else:
        raise RuntimeError('There is no file {}, but it is necessary'.format(labels_file_path))

    sly.logger.info('Determined {} class(es).'.format(len(cls2col)),
                        extra={'classes': list(cls2col.keys())})

    return number_class, pixel_color


def get_ann(img_path, inst_path, number_class, pixel_color):
    global classes_dict
    ann = sly.Annotation.from_img_path(img_path)

    if inst_path is not None:
        instance_img = sly.image.read(inst_path)
        current_color2class = {}
        temp = np.unique(instance_img)
        for pixel in temp:
            if pixel == 255:
                continue
            current_color2class[pixel] = number_class[pixel]

        for pixel, class_name in current_color2class.items():
            instance_img = np.where(instance_img != 0, instance_img, 200)
            new_color = pixel_color[pixel]
            if pixel == 0:
                pixel = 200
            imgray = np.where(instance_img == pixel, instance_img, 0)
            imgray = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 1, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                arr = np.array(contours[i], dtype=int)
                mask_temp = np.zeros(instance_img.shape, dtype=np.uint8)

                cv2.fillPoly(mask_temp, [np.int32(arr)], (255, 255, 255))
                mask_temp = cv2.split(mask_temp)[0]
                mask = mask_temp.astype(np.bool)
                bitmap = sly.Bitmap(data=mask)

                if not classes_dict.has_key(class_name):
                    obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=new_color)
                    classes_dict = classes_dict.add(obj_class)  # make it for meta.json

                ann = ann.add_label(sly.Label(bitmap, classes_dict.get(class_name)))
    return ann


def convert():
    settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
    imgs_dir = sly.TaskPaths.DATA_DIR
    inst_dir = os.path.join(sly.TaskPaths.DATA_DIR, 'stuffthingmaps_trainval2017')
    labels = os.path.join(sly.TaskPaths.DATA_DIR, 'labels.txt')
    number_class, pixel_color = read_colors(labels)
    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']), sly.OpenMode.CREATE)
    src_datasets = read_datasets(inst_dir)
    for ds_name, sample_names in src_datasets.items():
        ds = out_project.create_dataset(ds_name) #make train -> img, ann
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names)) # for logger
        imgs_dir_new = os.path.join(imgs_dir, ds_name)
        inst_dir_new = os.path.join(inst_dir, ds_name)
        for name in sample_names:
            src_img_path = os.path.join(imgs_dir_new, name + '.jpg')
            inst_path = os.path.join(inst_dir_new, name + '.png')

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
  sly.main_wrapper('COCO_stuff', main)
