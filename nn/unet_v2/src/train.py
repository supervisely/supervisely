# coding: utf-8

import os
import os.path as osp
import math
from collections import defaultdict
from copy import deepcopy

import cv2
import torch
import torch.nn.functional as functional
from torch.optim import Adam
from torch.utils.data import DataLoader
import supervisely_lib as sly
from supervisely_lib import logger
from supervisely_lib.utils.pytorch import cuda_variable

from common import create_model, SettingsValidator, TrainConfigRW, WeightsRW
from dataset import PytorchSlyDataset
from metrics import Accuracy, Dice, NLLLoss, BCEDiceLoss
from debug_saver import DebugSaver


# float progress of training, since zero
def epoch_float(epoch, train_it, train_its):
    res = epoch + train_it / float(train_its)
    return res


# to decide if we need validation or not (with val_every parameter)
class EvalPlanner:
    def __init__(self, epochs, val_every):
        self.epochs = epochs
        self.val_every = val_every
        self.total_val_cnt = self.validations_cnt(epochs, val_every)
        self._val_cnt = 0

    @property
    def performed_val_cnt(self):
        return self._val_cnt

    @staticmethod
    def validations_cnt(ep_float, val_every):
        res = math.floor(ep_float / val_every + 1e-9)
        return res

    def validation_performed(self):
        self._val_cnt += 1

    def need_validation(self, epoch_flt):
        req_val_cnt = self.validations_cnt(epoch_flt, self.val_every)
        need_val = req_val_cnt > self._val_cnt
        return need_val


# decrease lr after 'patience' calls w/out loss improvement
class LRPolicyWithPatience:
    def __init__(self, optim_cls, init_lr, patience, lr_divisor, model):
        self.optimizer = None
        self.optim_cls = optim_cls
        self.lr = init_lr
        self.patience = patience
        self.lr_divisor = lr_divisor
        self.losses = []
        self.last_reset_idx = 0

        logger.info('Selected optimizer.', extra={'optim_class': self.optim_cls.__name__})
        self._reset(model)

    def _reset(self, model):
        self.optimizer = self.optim_cls(model.parameters(), lr=self.lr)
        logger.info('Learning Rate has been updated.', extra={'lr': self.lr})

    def reset_if_needed(self, new_loss, model):
        self.losses.append(new_loss)
        no_recent_update = (len(self.losses) - self.last_reset_idx) > self.patience
        no_loss_improvement = min(self.losses[-self.patience:]) > min(self.losses)
        if no_recent_update and no_loss_improvement:
            self.lr /= float(self.lr_divisor)
            self._reset(model)
            self.last_reset_idx = len(self.losses)


class UnetV2Trainer:
    default_settings = {
        'dataset_tags': {
            'train': 'train',
            'val': 'val',
        },
        'batch_size': {
            'train': 1,
            'val': 1,
        },
        'data_workers': {
            'train': 0,
            'val': 0,
        },
        'allow_corrupted_samples': {
            'train': 0,
            'val': 0,
        },
        'special_classes': {
            'background': 'bg',
            'neutral': 'neutral',
        },
        'input_size': {
            'width': 128,
            'height': 128,
        },
        'epochs': 2,
        'val_every': 0.5,
        'lr': 0.1,
        'momentum': 0.9,
        'lr_decreasing': {
            'patience': 1000,
            'lr_divisor': 5,
        },
        'loss_weights': {
            'bce': 1.0,
            'dice': 1.0,
        },
        'weights_init_type': 'transfer_learning',  # 'continue_training',
        'validate_with_model_eval': True,
        'gpu_devices': [0],
    }

    bkg_input_idx = 0
    neutral_input_idx = 255

    def _determine_settings(self):
        input_config = self.helper.task_settings
        logger.info('Input config', extra={'config': input_config})
        config = deepcopy(self.default_settings)
        sly.update_recursively(config, input_config)
        logger.info('Full config', extra={'config': config})
        SettingsValidator.validate_train_cfg(config)
        self.config = config

    def _determine_model_classes(self):
        spec_cls = self.config['special_classes']
        self.class_title_to_idx, self.out_classes = sly.create_segmentation_classes(
            in_project_classes=self.helper.in_project_meta.classes,
            bkg_title=spec_cls['background'],
            neutral_title=spec_cls['neutral'],
            bkg_color=self.bkg_input_idx,
            neutral_color=self.neutral_input_idx,
        )
        logger.info('Determined model internal class mapping', extra={'class_mapping': self.class_title_to_idx})
        logger.info('Determined model out classes', extra={'classes': self.out_classes.py_container})

    def _determine_out_config(self):
        self.out_config = {
            'settings': self.config,
            'out_classes': self.out_classes.py_container,
            'class_title_to_idx': self.class_title_to_idx,
        }

    # for 'continue_training', requires exact match
    def _check_prev_model_config(self):
        prev_model_dir = self.helper.paths.model_dir
        prev_config_rw = TrainConfigRW(prev_model_dir)
        if not prev_config_rw.train_config_exists:
            raise RuntimeError('Unable to continue_training, config for previous training wasn\'t found.')
        prev_config = prev_config_rw.load()

        old_class_mapping = prev_config.get('class_title_to_idx', {})
        if self.class_title_to_idx != old_class_mapping:
            raise RuntimeError('Unable to continue training, class mapping is inconsistent with previous model.')

    def _construct_and_fill_model(self):
        self.device_ids = sly.remap_gpu_devices(self.config['gpu_devices'])
        self.model = create_model(n_cls=len(self.out_classes), device_ids=self.device_ids)

        if self.helper.model_dir_is_empty():
            logger.info('Weights will not be inited.')
            # @TODO: add random init (m.weight.data.normal_(0, math.sqrt(2. / n))
        else:
            wi_type = self.config['weights_init_type']
            ewit = {'weights_init_type': wi_type}
            logger.info('Weights will be inited from given model.', extra=ewit)

            weights_rw = WeightsRW(self.helper.paths.model_dir)
            if wi_type == 'transfer_learning':
                self.model = weights_rw.load_for_transfer_learning(self.model)
            elif wi_type == 'continue_training':
                self._check_prev_model_config()
                self.model = weights_rw.load_strictly(self.model)

            logger.info('Weights are loaded.', extra=ewit)

    def _construct_criterion_and_metrics(self):
        neutral = self.neutral_input_idx
        self.metrics = {
            'accuracy': Accuracy(ignore_index=neutral)
        }

        if len(self.out_classes) == 2:
            logger.info('Binary segmentation, will use both BCE & Dice loss components.')
            self.metrics.update({
                'dice': Dice(ignore_index=neutral)
            })
            l_weights = self.config['loss_weights']
            self.criterion = BCEDiceLoss(ignore_index=neutral, w_bce=l_weights['bce'], w_dice=l_weights['dice'])
        else:
            logger.info('Multiclass segmentation, will use NLLLoss only.')
            self.criterion = NLLLoss(ignore_index=neutral)

        self.val_metrics = {
            'loss': self.criterion,
            **self.metrics
        }
        logger.info('Selected metrics.', extra={'metrics': list(self.metrics.keys())})

    def _construct_data_loaders(self):
        logger.info('Will collect samples (img/ann pairs).')

        name_to_tag = self.config['dataset_tags']
        project_fs = sly.ProjectFS.from_disk_dir_project(self.helper.paths.project_dir)
        logger.info('Project structure has been read. Samples: {}.'.format(project_fs.pr_structure.image_cnt))

        samples_dct = sly.samples_by_tags(
            tags=list(name_to_tag.values()), project_fs=project_fs, project_meta=self.helper.in_project_meta
        )

        src_size = self.config['input_size']
        input_size_wh = (src_size['width'], src_size['height'])

        self.pytorch_datasets = {}
        for the_name, the_tag in name_to_tag.items():
            samples_lst = samples_dct[the_tag]
            the_ds = PytorchSlyDataset(
                project_meta=self.helper.in_project_meta,
                samples=samples_lst,
                out_size_wh=input_size_wh,
                class_mapping=self.class_title_to_idx,
                bkg_color=self.bkg_input_idx,
                allow_corrupted_cnt=self.config['allow_corrupted_samples'][the_name]
            )
            self.pytorch_datasets[the_name] = the_ds
            logger.info('Prepared dataset.', extra={
                'dataset_purpose': the_name, 'dataset_tag': the_tag, 'sample_cnt': len(samples_lst)
            })

        self.data_loaders = {}
        for name, need_shuffle in [
            ('train', True),
            ('val', False),
        ]:
            # note that now batch_size from config determines batch for single device
            batch_sz = self.config['batch_size'][name]
            batch_sz_full = batch_sz * len(self.device_ids)
            n_workers = self.config['data_workers'][name]
            self.data_loaders[name] = DataLoader(
                dataset=self.pytorch_datasets[name],
                batch_size=batch_sz_full,  # it looks like multi-gpu validation works
                num_workers=n_workers,
                shuffle=need_shuffle,
            )
        logger.info('DataLoaders are constructed.')

        self.train_iters = len(self.data_loaders['train'])
        self.val_iters = len(self.data_loaders['val'])
        self.epochs = self.config['epochs']
        self.eval_planner = EvalPlanner(epochs=self.epochs, val_every=self.config['val_every'])

    def __init__(self):
        logger.info('Will init all required to train.')
        self.helper = sly.TaskHelperTrain()

        self._determine_settings()
        self._determine_model_classes()
        self._determine_out_config()
        self._construct_and_fill_model()
        self._construct_criterion_and_metrics()

        self._construct_data_loaders()
        self.epoch_flt = 0  # real progress

        logger.info('Model is ready to train.')

    def _dump_model(self, is_best, opt_data):
        out_dir = self.helper.checkpoints_saver.get_dir_to_write()
        TrainConfigRW(out_dir).save(self.out_config)
        WeightsRW(out_dir).save(self.model)
        self.helper.checkpoints_saver.saved(is_best, opt_data)

    def _validation(self):
        logger.info("Before validation", extra={'epoch': self.epoch_flt})
        if self.config['validate_with_model_eval']:
            self.model.eval()

        metrics_values = defaultdict(int)
        samples_cnt = 0

        for val_it, (inputs, targets) in enumerate(self.data_loaders['val']):
            inputs, targets = cuda_variable(inputs, volatile=True), cuda_variable(targets)
            outputs = self.model(inputs)
            full_batch_size = inputs.size(0)
            for name, metric in self.val_metrics.items():
                metric_value = metric(outputs, targets)
                if isinstance(metric_value, torch.autograd.Variable):  # for val loss
                    metric_value = metric_value.data[0]
                metrics_values[name] += metric_value * full_batch_size
            samples_cnt += full_batch_size

            logger.info("Validation in progress", extra={'epoch': self.epoch_flt,
                                                         'val_iter': val_it, 'val_iters': self.val_iters})

        for name in metrics_values:
            metrics_values[name] /= float(samples_cnt)

        sly.report_metrics_validation(self.epoch_flt, metrics_values)

        self.model.train()
        logger.info("Validation has been finished", extra={'epoch': self.epoch_flt})
        return metrics_values

    def train(self):
        progress = sly.progress_counter_train(self.epochs, self.train_iters)
        self.model.train()

        lr_decr = self.config['lr_decreasing']
        policy = LRPolicyWithPatience(
            optim_cls=Adam,
            init_lr=self.config['lr'],
            patience=lr_decr['patience'],
            lr_divisor=lr_decr['lr_divisor'],
            model=self.model
        )
        best_val_loss = float('inf')

        debug_saver = None
        debug_save_prob = float(os.getenv('DEBUG_PATCHES_PROB', 0.0))
        if debug_save_prob > 0:
            target_multi = int(255.0 / len(self.out_classes))
            debug_saver = DebugSaver(odir=osp.join(self.helper.paths.debug_dir, 'debug_patches'),
                                     prob=debug_save_prob,
                                     target_multi=target_multi)

        for epoch in range(self.epochs):
            logger.info("Before new epoch", extra={'epoch': self.epoch_flt})

            for train_it, (inputs_cpu, targets_cpu) in enumerate(self.data_loaders['train']):
                inputs, targets = cuda_variable(inputs_cpu), cuda_variable(targets_cpu)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if debug_saver is not None:
                    out_cls = functional.softmax(outputs, dim=1)
                    debug_saver.process(inputs_cpu, targets_cpu, out_cls.data.cpu())

                policy.optimizer.zero_grad()
                loss.backward()
                policy.optimizer.step()

                metric_values_train = {'loss': loss.data[0]}
                for name, metric in self.metrics.items():
                    metric_values_train[name] = metric(outputs, targets)

                progress.iter_done_report()

                self.epoch_flt = epoch_float(epoch, train_it + 1, self.train_iters)
                sly.report_metrics_training(self.epoch_flt, metric_values_train)

                if self.eval_planner.need_validation(self.epoch_flt):
                    metrics_values_val = self._validation()
                    self.eval_planner.validation_performed()

                    val_loss = metrics_values_val['loss']
                    model_is_best = val_loss < best_val_loss
                    if model_is_best:
                        best_val_loss = val_loss
                        logger.info('It\'s been determined that current model is the best one for a while.')

                    self._dump_model(model_is_best, opt_data={
                        'epoch': self.epoch_flt,
                        'val_metrics': metrics_values_val,
                    })

                    policy.reset_if_needed(val_loss, self.model)

            logger.info("Epoch was finished", extra={'epoch': self.epoch_flt})


def main():
    cv2.setNumThreads(0)  # important for pytorch dataloaders
    x = UnetV2Trainer()  # load model & prepare all
    x.train()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(logger, sly.TaskPaths().debug_dir)
    sly.main_wrapper('UNET_V2_TRAIN', main)
