# coding: utf-8

import os
import os.path as osp
import math
from copy import deepcopy

import cv2
import numpy as np
import tensorflow as tf
import supervisely_lib as sly
from supervisely_lib import logger

from model import ICNet_BN
from tools import prepare_label, TrainConfigRW
from image_reader import ImageReader


IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
RANDOM_SEED = 1234

# Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
# @TODO: to config
LAMBDA1 = 0.16
LAMBDA2 = 0.4
LAMBDA3 = 1.0


# float progress of training, since zero
def epoch_float(epoch, train_it, train_its):
    res = epoch + train_it / float(train_its)
    return res


def get_model(images, num_classes, filter_scale, for_eval=False):
    net = ICNet_BN(
        {'data': images},
        is_training=not for_eval, evaluation=for_eval,
        num_classes=num_classes, filter_scale=filter_scale
    )
    return net


def get_mask(gt, num_classes, ignore_label):
    less_equal_class = tf.less_equal(gt, num_classes-1)
    not_equal_ignore = tf.not_equal(gt, ignore_label)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    indices = tf.squeeze(tf.where(mask), 1)
    return indices


def create_loss(output, label, num_classes, ignore_label):
    raw_pred = tf.reshape(output, [-1, num_classes])
    label_proc = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    raw_gt = tf.reshape(label_proc, [-1, ])

    indices = get_mask(raw_gt, num_classes, ignore_label)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
    reduced_loss = tf.reduce_mean(loss)

    raw_output_up = tf.argmax(output, dimension=3)

    return reduced_loss, pred, gt, label_proc, raw_output_up


def forward_and_all_losses(net, label_batch, num_classes, ignore_label, weight_decay):
    sub4_out = net.layers['sub4_out']
    sub24_out = net.layers['sub24_out']
    sub124_out = net.layers['conv6_cls']

    loss_sub4, _, _, _, _ = create_loss(sub4_out, label_batch, num_classes, ignore_label)
    loss_sub24, _, _, _, _ = create_loss(sub24_out, label_batch, num_classes, ignore_label)
    loss_sub124, pred, gt, label_proc, raw_output_up = \
        create_loss(sub124_out, label_batch, num_classes, ignore_label)
    l2_losses = [weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]

    reduced_loss = LAMBDA1 * loss_sub4 + LAMBDA2 * loss_sub24 + LAMBDA3 * loss_sub124 + tf.add_n(l2_losses)

    return reduced_loss, (loss_sub4, loss_sub24, loss_sub124), pred, gt, label_proc, raw_output_up


def get_trainable_vars(train_beta_gamma):
    all_trainable = [v for v in tf.trainable_variables()
                     if ('beta' not in v.name and 'gamma' not in v.name) or train_beta_gamma]
    return all_trainable


def get_grads(loss, train_beta_gamma, update_mean_var):
    all_trainable = get_trainable_vars(train_beta_gamma)

    # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
    if not update_mean_var:
        update_ops = None
    else:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        grads = tf.gradients(loss, all_trainable)

    return grads


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        average_grads.append(grad)
    return average_grads


def get_val_loss(split_images, split_labels, num_classes, weight_decay, device_ids, filter_scale, ignore_label_idx):
    losses_val = []
    with tf.name_scope("Valid"):
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for curr_dev_id in device_ids:
                with tf.device('/gpu:{}'.format(curr_dev_id)):
                    with tf.name_scope('clone_{}'.format(curr_dev_id)) as scope:
                        net_val = get_model(split_images[curr_dev_id], num_classes, filter_scale, for_eval=True)

                        reduced_loss_val, internal_losses, prediction, gt_val, val_label, val_output = \
                            forward_and_all_losses(net_val, split_labels[curr_dev_id],
                                                   num_classes, ignore_label_idx, weight_decay)
                        losses_val.append(reduced_loss_val)

    total_val_loss = tf.stack(values=losses_val)
    total_val_loss = tf.reduce_mean(total_val_loss)

    return total_val_loss, val_label, val_output  # @TODO: val_output from last device only!! average it


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


class ICNetTrainer:
    default_settings = {
        'dataset_tags': {
            'train': 'train',
            'val': 'val',
        },
        'batch_size': {
            'train': 1,
            'val': 1,
        },
        'special_classes': {
            'background': 'bg',
            'neutral': 'neutral',
        },
        'input_size': {
            'width': 256,
            'height': 256,
        },
        'epochs': 2,
        'val_every': 0.5,
        'lr': 0.1,
        'weight_decay': 0.0001,
        'momentum': 0.9,
        'filter_scale': 2,  # @TODO: add validation, train with =2 only (now)
        'train_beta_gamma': True,
        'update_mean_var': True,
        'lr_decreasing': {
            'power': 0.1,
        },
        'weights_init_type': 'transfer_learning',  # 'continue_training',
        'gpu_devices': [0],
    }

    neutral_input_idx = 255

    def _determine_settings(self):
        input_config = self.helper.task_settings
        logger.info('Input config', extra={'config': input_config})
        config = deepcopy(self.default_settings)
        sly.update_recursively(config, input_config)
        logger.info('Full config', extra={'config': config})
        # SettingsValidator.validate_train_cfg(config)  # @TODO: add later?
        self.config = config

    def _determine_model_classes(self):
        spec_cls = self.config['special_classes']
        self.class_title_to_idx, self.out_classes = sly.create_segmentation_classes(
            in_project_classes=self.helper.in_project_meta.classes,
            bkg_title=spec_cls['background'],
            neutral_title=spec_cls['neutral'],
            bkg_color=0,
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

    def _explore_input_project(self):
        logger.info('Will collect samples (img/ann pairs).')

        name_to_tag = self.config['dataset_tags']
        project_fs = sly.ProjectFS.from_disk_dir_project(self.helper.paths.project_dir)
        logger.info('Project structure has been read. Samples: {}.'.format(project_fs.pr_structure.image_cnt))

        self.samples_dct = sly.samples_by_tags(
            tags=list(name_to_tag.values()), project_fs=project_fs, project_meta=self.helper.in_project_meta
        )
        logger.info('Annotations are splitted by tags.')

        self.samples_cnt = {k: len(self.samples_dct[v]) for k, v in name_to_tag.items()}
        gpu_count = len(self.device_ids)
        self.iters_cnt = {k: int(np.ceil(float(self.samples_cnt[k]) / (self.config['batch_size'][k] * gpu_count)))
                          for k in name_to_tag.keys()}  # internal cnt, per epoch or per validation

        self.epochs = self.config['epochs']
        self.total_train_iters = self.iters_cnt['train'] * self.epochs
        self.eval_planner = EvalPlanner(epochs=self.epochs, val_every=self.config['val_every'])

    def _tf_common_init(self):
        gpu_count = len(self.device_ids)
        src_size = self.config['input_size']
        input_size_wh = (src_size['width'], src_size['height'])
        init_lr = self.config['lr']
        power = self.config['lr_decreasing']['power']
        momentum = self.config['momentum']
        filter_scale = self.config['filter_scale']
        weight_decay = self.config['weight_decay']
        num_classes = len(self.out_classes)
        train_beta_gamma = self.config['train_beta_gamma']
        update_mean_var = self.config['update_mean_var']

        with tf.device('/cpu:0'):
            self.coord = tf.train.Coordinator()
            splitted_images = {}
            splitted_labels = {}

            with tf.name_scope("create_inputs"):
                for name, need_shuffle in [
                    ('train', True),
                    ('val', False),
                ]:
                    reader = ImageReader(
                        ia_descrs=self.samples_dct[name],
                        input_size_wh=input_size_wh,
                        random_scale=False,
                        random_mirror=False,
                        img_mean=IMG_MEAN,
                        coord=self.coord,
                        in_pr_meta=self.helper.in_project_meta,
                        class_to_idx=self.class_title_to_idx,
                        shuffle=need_shuffle
                    )
                    batch_sz = self.config['batch_size'][name]
                    img_batch, lbl_batch = reader.dequeue(batch_sz * gpu_count)
                    split_images = tf.split(img_batch, gpu_count, 0)
                    split_labels = tf.split(lbl_batch, gpu_count, 0)
                    splitted_images[name] = split_images
                    splitted_labels[name] = split_labels

            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)

            self.tf_label = tf.placeholder(dtype=tf.int32)  # , shape=[None])
            self.tf_prediction = tf.placeholder(dtype=tf.int32)  # , shape=[None])
            self.tf_metric, self.tf_metric_update = tf.metrics.accuracy(
                self.tf_label, self.tf_prediction, name="use_metric_acc"
            )
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="use_metric_acc")
            self.running_vars_initializer = tf.variables_initializer(var_list=running_vars)

            base_lr = tf.constant(init_lr)
            self.step_ph = tf.placeholder(dtype=tf.float32, shape=())
            learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - self.step_ph / self.total_train_iters), power))
            opt = tf.train.MomentumOptimizer(learning_rate, momentum)

            all_grads = []
            losses = []
            with tf.variable_scope(tf.get_variable_scope()):
                for curr_dev_id in self.device_ids:
                    with tf.device('/gpu:{}'.format(curr_dev_id)):
                        with tf.name_scope('clone_{}'.format(curr_dev_id)) as scope:
                            spl_img = splitted_images['train'][curr_dev_id]
                            spl_lbl = splitted_labels['train'][curr_dev_id]

                            net = get_model(spl_img, num_classes, filter_scale)

                            reduced_loss, internal_losses, prediction, gt, self.v1, self.v2 = \
                                forward_and_all_losses(
                                    net, spl_lbl, num_classes, self.neutral_input_idx, weight_decay
                                )
                            # loss_sub4, loss_sub24, loss_sub124 = internal_losses
                            losses.append(reduced_loss)
                            tf.get_variable_scope().reuse_variables()

                            grads = get_grads(reduced_loss, train_beta_gamma, update_mean_var)
                            all_grads.append(grads)

            self.total_loss = tf.stack(values=losses)
            self.total_loss = tf.reduce_mean(self.total_loss)

            mean_grads = average_gradients(all_grads)

            all_trainable = get_trainable_vars(train_beta_gamma)

            # Apply the gradients to adjust the shared variables.
            self.train_op = opt.apply_gradients(zip(mean_grads, all_trainable), global_step=global_step)

            # # Group all updates to into a single train op.
            # train_op = tf.group(apply_gradient_conv_op, apply_gradient_fc_w_op,
            #                     apply_gradient_fc_b_op)

            self.total_val_loss, self.v1_val, self.v2_val = get_val_loss(
                splitted_images['val'], splitted_labels['val'],
                num_classes, weight_decay, self.device_ids, filter_scale, self.neutral_input_idx
            )

            # Set up tf session and initialize variables.
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            self.sess = tf.Session(config=config)
            init = tf.global_variables_initializer()

            self.sess.run(init)

            # Saver for storing checkpoints of the model.
            self.saver = tf.train.Saver(var_list=tf.global_variables(), save_relative_paths=True)

    def _load_weights(self):
        if self.helper.model_dir_is_empty():
            logger.info('Weights will not be inited.')
            # @TODO: add random init? is it possible to train model in the case?

        else:
            wi_type = self.config['weights_init_type']
            ewit = {'weights_init_type': wi_type}
            logger.info('Weights will be inited from given model.', extra=ewit)

            if wi_type == 'transfer_learning':
                logger.info('Last layer will be reinitialized')
                restore_var = [v for v in tf.global_variables() if
                               ('conv6_cls' not in v.name) and ('sub4_out' not in v.name) and
                               ('sub24_out' not in v.name) and ('sub124_out' not in v.name) and (
                                           'Momentum' not in v.name)]
            elif wi_type == 'continue_training':
                self._check_prev_model_config()
                restore_var = [v for v in tf.global_variables() if 'Momentum' not in v.name]
            else:
                raise NotImplemented()

            restore_var = [x for x in restore_var if 'global_step' not in x.name]
            loader = tf.train.Saver(var_list=restore_var)
            ckpt_path = osp.join(self.helper.paths.model_dir, 'model.ckpt')
            loader.restore(self.sess, ckpt_path)
            logger.info('Weights are loaded.', extra=ewit)

    def _dump_model(self, is_best, opt_data):
        out_dir = self.helper.checkpoints_saver.get_dir_to_write()
        TrainConfigRW(out_dir).save(self.out_config)

        model_fpath = os.path.join(out_dir, 'model.ckpt')
        self.saver.save(self.sess, model_fpath)

        self.helper.checkpoints_saver.saved(is_best, opt_data)

    def __init__(self):
        logger.info('Will init all required to train.')
        self.helper = sly.TaskHelperTrain()

        self._determine_settings()
        self.device_ids = sly.remap_gpu_devices(self.config['gpu_devices'])
        tf.set_random_seed(RANDOM_SEED)  # @TODO: from config

        self._determine_model_classes()
        self._determine_out_config()
        self._explore_input_project()
        self._tf_common_init()
        self._load_weights()
        self.epoch_flt = 0  # real progress

        logger.info('Model is ready to train.')

    def train(self):
        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

        progress = sly.progress_counter_train(self.epochs, self.iters_cnt['train'])
        best_val_loss = float('inf')
        internal_step = 0

        for epoch in range(self.epochs):
            logger.info("Before new epoch", extra={'epoch': self.epoch_flt})

            for train_it in range(self.iters_cnt['train']):
                feed_dict = {self.step_ph: internal_step}
                internal_step += 1
                loss_value, _ = self.sess.run([self.total_loss, self.train_op], feed_dict=feed_dict)

                self.sess.run(self.running_vars_initializer)
                # Update the running variables on new batch of samples
                plh_label = self.sess.run(self.v1)
                plh_prediction = self.sess.run(self.v2)
                feed_dict = {self.tf_label: plh_label, self.tf_prediction: plh_prediction}
                train_accuracy = self.sess.run(self.tf_metric_update, feed_dict=feed_dict)  # from last GPU

                metrics_values_train = {
                    'loss': loss_value,
                    'accuracy': train_accuracy
                }

                progress.iter_done_report()
                self.epoch_flt = epoch_float(epoch, train_it + 1, self.iters_cnt['train'])
                sly.report_metrics_training(self.epoch_flt, metrics_values_train)

                if self.eval_planner.need_validation(self.epoch_flt):
                    logger.info("Before validation", extra={'epoch': self.epoch_flt})

                    overall_val_loss = 0
                    overall_val_accuracy = 0
                    for val_it in range(self.iters_cnt['val']):
                        overall_val_loss += self.sess.run(self.total_val_loss)
                        self.sess.run(self.running_vars_initializer)
                        # Update the running variables on new batch of samples
                        plh_label = self.sess.run(self.v1_val)
                        plh_prediction = self.sess.run(self.v2_val)
                        feed_dict = {self.tf_label: plh_label, self.tf_prediction: plh_prediction}
                        overall_val_accuracy += self.sess.run(self.tf_metric_update, feed_dict=feed_dict)

                        logger.info("Validation in progress", extra={'epoch': self.epoch_flt,
                                                                     'val_iter': val_it,
                                                                     'val_iters': self.iters_cnt['val']})

                    metrics_values_val = {
                        'loss': overall_val_loss / self.iters_cnt['val'],
                        'accuracy': overall_val_accuracy / self.iters_cnt['val']
                    }
                    sly.report_metrics_validation(self.epoch_flt, metrics_values_val)
                    logger.info("Validation has been finished", extra={'epoch': self.epoch_flt})

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

            logger.info("Epoch was finished", extra={'epoch': self.epoch_flt})

        self.coord.request_stop()
        self.coord.join(threads)


def main():
    cv2.setNumThreads(0)
    x = ICNetTrainer()  # load model & prepare all
    x.train()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(logger, sly.TaskPaths().debug_dir)
    sly.main_wrapper('ICNET_TRAIN', main)
