# coding: utf-8

import argparse

import supervisely_lib as sly

from common import construct_detection_classes, TrainConfigRW


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_file', type=str,
        help='Input class lines.', required=True)
    parser.add_argument(
        '--out_dir', type=str,
        help='Dir to save json.', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.in_file) as f:
        lines = f.readlines()
    lines = [ln for ln in (line.strip() for line in lines) if ln]

    out_classes = sly.FigClasses()
    for x in construct_detection_classes(lines):
        out_classes.add(x)
    cls_mapping = {x: i for i, x in enumerate(lines)}
    res_cfg = {
        'settings': {},
        'out_classes': out_classes.py_container,
        'class_title_to_idx': cls_mapping,
    }

    saver = TrainConfigRW(args.out_dir)
    saver.save(res_cfg)
    print('Done: {} -> {}'.format(args.in_file, saver.train_config_fpath))


if __name__ == '__main__':
    main()
