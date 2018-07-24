import numpy as np
import supervisely_lib as sly


def load_ann(ann_path, labels_mapping, in_pr_meta):
    ann_packed = sly.json_load(ann_path)
    ann = sly.Annotation.from_packed(ann_packed, in_pr_meta)
    # ann.normalize_figures()  # @TODO: enaaaable!

    w, h = ann.image_size_wh
    gt = np.zeros((h, w), dtype=np.uint8)  # default bkg = 0
    for fig in ann['objects']:
        gt_color = labels_mapping.get(fig.class_title, None)
        if gt_color is None:
            err_str = 'Missing class mapping (title to index). Class {}.'.format(fig.class_title)
            print(err_str)  # exception info may be suppressed
            raise RuntimeError(err_str)
        fig.draw(gt, gt_color)

    gt = gt.astype(np.float32)
    return gt


# def get_labels_mapping(classes_json, ignore_class, ignore_label=255):
#     class_names = [x['title'] for x in classes_json]
#     labels_mapping = dict()
#     if ignore_class in class_names:
#         class_names.remove(ignore_class)
#         labels_mapping[ignore_class] = ignore_label
#     class_names = sorted(class_names)
#     labels_mapping.update(dict(zip(class_names, range(1, len(class_names) + 1))))
#
#     return labels_mapping


def get_label_colours(classes_json, labels_mapping, ignore_label=255):
    unsorted_colors = {}
    sorted_classes_json = sorted(classes_json, key=lambda x: x['title'])
    sorted_labels_mapping = sorted(labels_mapping.items(), key=lambda x: x[1])
    for i in range(len(classes_json)):
        class_desc = sorted_classes_json[i]
        label_mapping = sorted_labels_mapping[i]
        if label_mapping[1] == ignore_label:
            continue
        color_code = class_desc['color']
        color_rgb = shared_utils.code2color(color_code)
        unsorted_colors[label_mapping[0]] = color_rgb
    label_colours = [list(color[1]) for color in sorted(unsorted_colors.items(), key=lambda x: x[0])]

    return label_colours


def format_result(result, classes_json):
    objects = []

    for cls in classes_json:
        obj = dict()
        label = cls['internal_idx']
        if label not in result:
            continue
        mask = result == label
        obj['bitmap'] = {"origin": [0, 0], "np": mask}
        obj['type'] = 'bitmap'
        obj['classTitle'] = cls['title']
        obj['description'] = ""
        obj['tags'] = []
        obj['points'] = {"interior": [], "exterior": []}
        objects.append(obj)

    return objects
