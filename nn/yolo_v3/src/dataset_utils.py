import numpy as np
import supervisely_lib as sly


def get_bbox(mask):
    mask_points = np.where(mask == 1)
    return np.min(mask_points[1]), np.min(mask_points[0]), np.max(mask_points[1]), np.max(mask_points[0])


def load_ann(ann_fpath, classes_mapping, project_meta):
    ann_packed = sly.json_load(ann_fpath)
    ann = sly.Annotation.from_packed(ann_packed, project_meta)
    (w, h) = ann.image_size_wh

    gt_boxes, classes_text, classes = [], [], []
    for fig in ann['objects']:
        gt_idx = classes_mapping.get(fig.class_title, None)
        if gt_idx is None:
            raise RuntimeError('Missing class mapping (title to index). Class {}.'.format(fig.class_title))
        rect = fig.get_bbox()
        x = (rect.left + rect.right) / (2*w)
        y = (rect.bottom + rect.top) / (2*h)
        r_width = (rect.right - rect.left) / w
        r_height = (rect.bottom - rect.top) / h
        gt_boxes.extend([gt_idx, x, y, r_width, r_height])
    num_boxes = len(ann['objects'])
    return num_boxes, gt_boxes


def load_dataset(samples, classes_mapping, project_meta):
    img_paths = []
    gts = []
    num_boxes = []
    for descr in samples:
        img_paths.append(descr.img_path.encode('utf-8'))
        num_b, gt_boxes = load_ann(descr.ann_path, classes_mapping, project_meta)
        gts.append(gt_boxes)
        num_boxes.append(num_b)

    gts = [np.array(x).astype(np.float32).tolist() for x in gts]

    return img_paths, gts, num_boxes
