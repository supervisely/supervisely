import os
import cv2
import warnings

try:
    import torch
except ImportError:
    warnings.warn(
        "Pytorch is not installed (ignore this warning if you are not going to use semantic segmentation model benchmark)"
    )


class SegmentationLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        pred_dir,
        gt_dir,
        pred_label_map=None,
        gt_label_map=None,
    ):
        self.pred_dir = pred_dir
        self.pred_files = sorted(os.listdir(self.pred_dir))
        self.gt_dir = gt_dir
        self.gt_files = sorted(os.listdir(self.gt_dir))
        if len(self.pred_files) != len(self.gt_files):
            raise RuntimeError(
                f"Number of predictions({len(self.pred_files)}) and ground-truth annotations ({len(self.gt_files)}) do not match!"
            )
        self.pred_label_map = pred_label_map
        self.gt_label_map = gt_label_map

    def __len__(self):
        return len(self.pred_files)

    def __getitem__(self, index):
        pred_path = os.path.join(self.pred_dir, self.pred_files[index])
        gt_path = os.path.join(self.gt_dir, self.gt_files[index])

        pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        if self.pred_label_map:
            pred = self.pred_label_map(pred)

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if self.gt_label_map:
            gt = self.gt_label_map(gt)

        img_name = self.gt_files[index]
        return (pred, gt, img_name)


def build_segmentation_loader(
    pred_dir,
    gt_dir,
    num_workers=0,
    pred_label_map=None,
    gt_label_map=None,
):
    loader = SegmentationLoader(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        pred_label_map=pred_label_map,
        gt_label_map=gt_label_map,
    )
    return torch.utils.data.DataLoader(
        loader,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda l: (l[0][0], l[0][1], l[0][2]),
    )
