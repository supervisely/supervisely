# coding: utf-8

import argparse
import glob
import os.path as osp
import json

import numpy as np
import pandas as pd
from supervisely_lib import EventType
import matplotlib as mpl
mpl.use('Agg')  # prevent from using X
import matplotlib.pyplot as plt


def training_log_to_df(fpath):
    with open(fpath) as f:
        str_content = f.readlines()

    records = []
    for s in str_content:
        try:
            dct = json.loads(s)
            if dct.get('event_type', None) != str(EventType.METRICS):
                continue
            dct_to_row = {
                'epoch': dct['epoch'],
                'stage_name': dct['type'],
            }
            dct_to_row.update(dct['metrics'])
            records.append(dct_to_row)
        except KeyError:
            pass  # ok, some records may have other structure

    df = pd.DataFrame.from_records(records)
    return df


def drop_before_n_epochs(log_df, last_epochs):
    ser = log_df['epoch']
    thresh = ser.max() - last_epochs
    res_df = log_df[ser > thresh]
    return res_df


def plot_training_log(log_df, ofpath):
    if not ofpath:
        return

    metrics = sorted(list(set(log_df.columns.values) - {'epoch', 'stage_name'}))
    stages = [
        ('train', 'g-', 1),
        ('val', 'ro-', 2),
    ]

    ep_ser = log_df['epoch']
    min_epoch = np.floor(ep_ser.min())
    max_epoch = np.ceil(ep_ser.max())
    to_ticks = np.arange(min_epoch, max_epoch + 1, 1)

    subpl_cnt = len(metrics)
    fig, axs = plt.subplots(subpl_cnt, 1, figsize=(16, 1 + 4 * subpl_cnt))
    for metr, ax in zip(metrics, axs):

        for st_name, fmt, lw in stages:
            mask = log_df['stage_name'] == st_name
            xs = log_df.loc[mask, 'epoch'].values
            ys = log_df.loc[mask, metr].values.astype(np.float)
            ax.plot(xs, ys, fmt, linewidth=lw, label=st_name)

        ax.set_xticks(to_ticks)
        ax.grid(True, linestyle=":", color='gray')
        ax.legend(loc='lower center')
        ax.set_ylabel(metr)
        ax.set_xlabel('Epoch')
        ax.set_title(metr + ':', fontsize=16)

    fig.tight_layout()
    plt.savefig(ofpath, dpi=100)
    fig.clf()
    plt.close(fig)


def find_latest_log(dirpath):
    # get list of items in dir with required ext
    exts = ['*.txt', "*.log"]
    flist = []
    for ext in exts:
        flist.extend(glob.glob(osp.join(dirpath, ext)))

    if len(flist) == 0:
        raise RuntimeError('Unable to find logs in "{}".'.format(dirpath))
    res_fpath = max(flist, key=osp.getctime)
    return res_fpath


def parse_args():
    parser = argparse.ArgumentParser()

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        '--src_file', type=str,
        help='Jsonlines file (training log).')
    src_group.add_argument(
        '--src_dir', type=str,
        help='Dir with jsonlines file (search for newly changed).')

    parser.add_argument(
        '--out_file', type=str,
        help='Path to save plotted chart.',
        required=True)
    parser.add_argument(
        '--last_epochs', type=float,
        help='Number of last epochs to plot.',
        default=None)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.src_dir is not None:
        src_fpath = find_latest_log(args.src_dir)
    else:
        src_fpath = args.src_file

    print('{} -> {}'.format(src_fpath, args.out_file))
    log_df = training_log_to_df(src_fpath)

    if args.last_epochs is not None:
        log_df = drop_before_n_epochs(log_df, args.last_epochs)

    plot_training_log(log_df, args.out_file)


if __name__ == '__main__':
    main()
