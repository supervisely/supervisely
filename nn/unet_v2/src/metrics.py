# coding: utf-8

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from supervisely_lib.utils.pytorch import cuda_variable


def dice_loss(preds, trues, weight=None, is_average=True, ignore_index=None):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)

    if ignore_index is not None:
        ignore_mask = trues.data == ignore_index
        preds = preds.clone()
        preds.data[ignore_mask] = 0
        trues = trues.clone()
        trues.data[ignore_mask] = 0

    if weight is not None:
        w = cuda_variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w

    intersection = (preds * trues).sum(1)
    scores = 2. * (intersection + 1) / (preds.sum(1) + trues.sum(1) + 1)

    if is_average:
        score = scores.sum() / num
        return torch.clamp(score, 0., 1.)
    else:
        return scores


class DiceLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=None):
        super().__init__()
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, outputs, target, weight=None):
        return 1 - dice_loss(outputs, target, weight=weight, is_average=self.size_average,
                             ignore_index=self.ignore_index)


# for binary segmentation
class BCEDiceLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=None, w_bce=1, w_dice=1):
        super().__init__()
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.dice = DiceLoss(size_average=size_average, ignore_index=ignore_index)
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.full_divis = float(w_bce + w_dice)

    def forward(self, outputs, targets, weight=None):
        outputs = functional.softmax(outputs, dim=1)
        outputs = outputs[:, 1, :, :]
        outputs = outputs.unsqueeze(1)
        outputs = outputs.contiguous()  # binary segm after softmax: get scores for class #1

        dice_comp = self.dice(outputs, targets.float(), weight=weight)
        if self.ignore_index is None:
            bce_comp = nn.BCELoss(size_average=self.size_average, weight=weight)(outputs, targets.float())
            res = self.w_bce * bce_comp + self.w_dice * dice_comp
        else:
            # when neutral class exists
            mask = targets != self.ignore_index
            bce_comp = nn.BCELoss(size_average=self.size_average, weight=weight)(outputs[mask], targets[mask].float())
            res = self.w_bce * bce_comp + self.w_dice * dice_comp

        res /= self.full_divis
        return res


class NLLLoss:
    def __init__(self, ignore_index):
        self.nll_loss = nn.NLLLoss2d(ignore_index=ignore_index)

    def __call__(self, outputs, targets):
        outputs = functional.log_softmax(outputs, dim=1)
        targets = targets.squeeze(1)
        return self.nll_loss(outputs, targets)


class Accuracy:  # multiclass
    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index

    def __call__(self, outputs, targets):
        # outputs = F.softmax(outputs, dim=1)
        outputs = outputs.data.cpu().numpy()
        outputs = np.argmax(outputs, axis=1)

        targets = targets.data.cpu().numpy()
        targets = np.squeeze(targets, 1)  # 4d to 3d

        total_pixels = np.prod(outputs.shape)
        if self.ignore_index is not None:
            total_pixels -= np.sum((targets == self.ignore_index).astype(int))
        correct_pixels = np.sum((outputs == targets).astype(int))
        res = correct_pixels / total_pixels
        return res


class Dice:  # binary only
    def __init__(self, threshold=0.5, ignore_index=None):
        self.threshold = threshold
        self.ignore_index = ignore_index

    def __call__(self, outputs, targets):
        eps = 1e-5
        num_patches = outputs.size(0)

        # outputs = F.softmax(outputs, dim=1)
        outputs = outputs.data.cpu().numpy()
        outputs = np.argmax(outputs, axis=1)

        targets = targets.data.cpu().numpy()
        targets = np.squeeze(targets, 1)  # 4d to 3d

        outputs = outputs.reshape(num_patches, -1)
        targets = targets.reshape(num_patches, -1)

        if self.ignore_index is not None:
            ignore_mask = targets == self.ignore_index
            outputs[ignore_mask] = 0
            targets[ignore_mask] = 0
        nominator = np.sum(outputs*targets, axis=1)
        res = np.sum((2.*nominator/(np.sum(outputs, axis=1) + np.sum(targets, axis=1) + eps))) / outputs.shape[0]
        return res
