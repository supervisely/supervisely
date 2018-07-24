import collections
import numpy as np
import tensorflow as tf
import supervisely_lib as sly
import deployment.model_deploy as model_deploy
import deeplab.input_preprocess as input_preprocess
import deeplab.model as model

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from deeplab.utils import train_utils
from deeplab.model_utils import ModelOptions
from common import EvalPlanner
from supervisely_lib import logger

slim = tf.contrib.slim
prefetch_queue = slim.prefetch_queue


model_variant = "xception_65"
clone_on_cpu = False
num_replicas = 1
num_ps_tasks = 0
task = 0
weight_decay = 0.00004
last_layer_gradient_multiplier = 1.0
upsample_logits = True
image_pyramid = None


def epoch_float(epoch, train_it, train_its):
    res = epoch + train_it / float(train_its)
    return res


def _build_deeplab(inputs_queue,
                   outputs_to_num_classes,
                   input_size,
                   atrous_rates,
                   output_stride,
                   fine_tune_batch_norm):
    samples = inputs_queue.dequeue()

    model_options = ModelOptions(
        outputs_to_num_classes=outputs_to_num_classes,
        crop_size=input_size,
        atrous_rates=atrous_rates,
        output_stride=output_stride)
    outputs_to_scales_to_logits = model.multi_scale_logits(
        samples['image'],
        model_options=model_options,
        image_pyramid=image_pyramid,
        weight_decay=weight_decay,
        is_training=True,
        fine_tune_batch_norm=True)

    for output, num_classes in outputs_to_num_classes.items():
        train_utils.add_softmax_cross_entropy_loss_for_each_scale(
            outputs_to_scales_to_logits[output],
            samples['label'],
            num_classes,
            255,
            loss_weight=1.0,
            upsample_logits=upsample_logits,
            scope=output)

    return outputs_to_scales_to_logits


def _build_deeplab_val(scope,
                       inputs_queue,
                       outputs_to_num_classes,
                       input_size,
                       atrous_rates,
                       output_stride):
    samples = inputs_queue.dequeue()

    model_options = ModelOptions(
        outputs_to_num_classes=outputs_to_num_classes,
        crop_size=input_size,
        atrous_rates=atrous_rates,
        output_stride=output_stride)
    outputs_to_scales_to_logits = model.multi_scale_logits_val(
        samples['image'],
        model_options=model_options,
        image_pyramid=image_pyramid,
        weight_decay=weight_decay,
        is_training=True,
        fine_tune_batch_norm=True)
    for output, num_classes in outputs_to_num_classes.items():
        loss = train_utils.get_softmax_cross_entropy_val_loss(
                                        outputs_to_scales_to_logits[output],
                                        samples['label'],
                                        num_classes,
                                        255,
                                        loss_weight=1.0,
                                        upsample_logits=upsample_logits,
                                        scope=scope)

    return outputs_to_scales_to_logits, loss


def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


def ann2label(ann_fpath, class_mapping, project_meta, out_size_wh):
    ann_packed = sly.json_load(ann_fpath)
    ann = sly.Annotation.from_packed(ann_packed, project_meta)
    (w, h) = ann.image_size_wh
    # ann.normalize_figures()  # @TODO: enaaaable!
    # will not resize figures: resize gt instead

    gt = np.zeros((h, w), dtype=np.uint8)  # default bkg
    for fig in ann['objects']:
        gt_color = class_mapping.get(fig.class_title, None)
        if gt_color is None:
            raise RuntimeError('Missing class mapping (title to index). Class {}.'.format(fig.class_title))
        fig.draw(gt, gt_color)

    gt = sly.resize_inter_nearest(gt, out_size_wh).astype(np.float32)
    return gt


def read_image(image_path):
    # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
    path_length = string_length_tf(image_path)[0]
    file_extension = tf.substr(image_path, path_length - 3, 3)
    file_cond = tf.equal(file_extension, 'jpg')

    image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                    lambda: tf.image.decode_png(tf.read_file(image_path)))

    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def get(data_dict,
        crop_size,
        num_threads=2,
        is_training=True,
        model_variant=None):

    batch_size = data_dict['batch_size']
    # img_names, img_paths, ann_paths, labels_mapping, project_meta = structure
    img_names = [sample.img_path for sample in data_dict['samples']]
    img_paths = [sample.img_path for sample in data_dict['samples']]
    ann_paths = [sample.ann_path for sample in data_dict['samples']]

    def load_ann(ann_path, labels_mapping=data_dict['classes_mapping'], project_meta=data_dict['project_meta'], out_size=crop_size):
        label = ann2label(ann_path, class_mapping=labels_mapping, project_meta=project_meta, out_size_wh=out_size)

        return label

    def read_images_and_labels(input_queue, out_size=crop_size):
        value = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(value)
        image = tf.image.resize_images(image, out_size)
        label_part = tf.identity(input_queue[1])
        label = tf.py_func(load_ann, [label_part], tf.float32, stateful=False)
        label.set_shape(out_size)
        label = tf.cast(label, tf.float32)

        return image, label

    image_name = ops.convert_to_tensor(img_names, dtype=dtypes.string)

    images = ops.convert_to_tensor(img_paths)
    anns = ops.convert_to_tensor(ann_paths)
    input_images_and_labels_queue = tf.train.slice_input_producer([images, anns],
                                                                  shuffle=True)

    image, label = read_images_and_labels(input_images_and_labels_queue)

    if label is not None:
        if label.shape.ndims == 2:
            label = tf.expand_dims(label, 2)
        elif label.shape.ndims == 3 and label.shape.dims[2] == 1:
            pass
        else:
            raise ValueError('Input label shape must be [height, width], or '
                             '[height, width, 1].')

        label.set_shape([None, None, 1])
    original_image, image, label = input_preprocess.preprocess_image_and_label(
                                        image,
                                        label,
                                        crop_height=crop_size[0],
                                        crop_width=crop_size[1],
                                        ignore_label=255,
                                        is_training=is_training,
                                        model_variant=model_variant)
    # image.set_shape([crop_size[1], crop_size[0], 3])
    # label.set_shape([crop_size[1], crop_size[0], 1])
    # label = tf.cast(label, tf.int32)

    sample = {'image': image, 'image_name': image_name, 'label': label}

    return tf.train.batch(
        sample,
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=32 * batch_size,
        allow_smaller_final_batch=is_training,
        dynamic_pad=True)


def get_clones_val_losses(clones, regularization_losses, val_losses):
    clones_losses = []
    num_clones = len(clones)
    if regularization_losses is None:
        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
    for i, clone in enumerate(clones):
        with tf.name_scope(clone.scope):
            clone_loss = train_utils._gather_clone_loss(clone, num_clones, regularization_losses, val_losses[i])
            if clone_loss is not None:
                clones_losses.append(clone_loss)
            # Only use regularization_losses for the first clone
            regularization_losses = None
    # Compute the total_loss summing all the clones_losses.
    total_val_loss = tf.add_n(clones_losses, name='total_val_loss')
    return total_val_loss


Clone = collections.namedtuple('Clone',
                               ['outputs',  # Whatever model_fn() returned.
                                'scope',  # The scope used to create it.
                                'device',  # The device used to create.
                                ])


def create_val_clones(num_clones, config, model_fn, args=None, kwargs=None):
    clones = []
    losses = []
    args = args or []
    kwargs = kwargs or {}
    with slim.arg_scope([slim.model_variable, slim.variable],
                        device=config.variables_device()):
        for i in range(num_clones):
            with tf.name_scope(config.clone_scope(i)) as clone_scope:
                clone_device = config.clone_device(i)
                with tf.device(clone_device):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        outputs, loss = model_fn(clone_scope, *args, **kwargs)
                        losses.append(loss)
                        clones.append(Clone(outputs, clone_scope, clone_device))
    return clones, losses


def train(data_dicts,
          class_num,
          input_size,
          lr,
          n_epochs,
          num_clones,
          iters_cnt,
          val_every,
          model_init_fn,
          save_cback,
          atrous_rates=[6, 12, 18],
          fine_tune_batch_norm=True,
          output_stride=16
          ):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
    config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=num_replicas,
        num_ps_tasks=num_ps_tasks)

    with tf.Graph().as_default():
        with tf.device(config.inputs_device()):
            samples = get(
                data_dicts['train'],
                input_size,
                is_training=True,
                model_variant=model_variant)
            samples_val = get(
                data_dicts['val'],
                input_size,
                is_training=True,
                model_variant=model_variant)

        inputs_queue = prefetch_queue.prefetch_queue(
            samples, capacity=128 * config.num_clones, dynamic_pad=True)
        inputs_queue_val = prefetch_queue.prefetch_queue(
            samples_val, capacity=128 * config.num_clones, dynamic_pad=True)
        coord = tf.train.Coordinator()

        # Create the global step on the device storing the variables.
        with tf.device(config.variables_device()):
            global_step = tf.train.create_global_step()

            # Define the model and create clones.
            model_fn = _build_deeplab
            model_args = (inputs_queue, {
                'semantic': class_num
            }, input_size, atrous_rates, output_stride, fine_tune_batch_norm)
            clones = model_deploy.create_clones(config, model_fn, args=model_args)

            # Gather update_ops from the first clone. These contain, for example,
            # the updates for the batch_norm variables created by model_fn.
            first_clone_scope = config.clone_scope(0)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Build the optimizer based on the device specification.
        with tf.device(config.optimizer_device()):
            learning_rate = lr
            optimizer = tf.train.AdamOptimizer(learning_rate)

        with tf.device(config.variables_device()):
            total_loss, grads_and_vars = model_deploy.optimize_clones(
                clones, optimizer)
            total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')

            model_fn_val = _build_deeplab_val
            model_args_val = (inputs_queue_val, {
                'semantic': class_num
            }, input_size, atrous_rates, output_stride)
            val_clones, val_losses = create_val_clones(num_clones, config, model_fn_val, args=model_args_val)
            val_total_loss = get_clones_val_losses(val_clones, None, val_losses)
            # Modify the gradients for biases and last layer variables.
            last_layers = model.get_extra_layer_scopes()
            grad_mult = train_utils.get_model_gradient_multipliers(
                last_layers, last_layer_gradient_multiplier)
            if grad_mult:
                grads_and_vars = slim.learning.multiply_gradients(
                    grads_and_vars, grad_mult)

            # Create gradient update op.
            grad_updates = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)
            with tf.control_dependencies([update_op]):
                train_tensor = tf.identity(total_loss, name='train_op')

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        coord.clear_stop()
        sess = tf.Session(config=config)

        graph = ops.get_default_graph()
        with graph.as_default():
            with ops.name_scope('init_ops'):
                init_op = variables.global_variables_initializer()
                ready_op = variables.report_uninitialized_variables()
                local_init_op = control_flow_ops.group(
                    variables.local_variables_initializer(),
                    lookup_ops.tables_initializer())

        # graph.finalize()
        sess.run([init_op, ready_op, local_init_op])
        queue_runners = graph.get_collection(ops.GraphKeys.QUEUE_RUNNERS)
        threads = []
        for qr in queue_runners:
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

        # # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        # #     print(i)
        # vary_23 = [v for v in tf.global_variables() if v.name == 'xception_65/middle_flow/block1/unit_8/xception_module/separable_conv3_depthwise/BatchNorm/moving_mean:0'][0]
        #
        # beta_23 = [v for v in tf.global_variables() if v.name == 'xception_65/middle_flow/block1/unit_8/xception_module/separable_conv3_depthwise/BatchNorm/gamma:0'][0]
        # for i in range(1000):
        #     train_loss = sess.run(train_tensor)
        #     print(train_loss)
        #     vary, beta = sess.run([vary_23, beta_23])
        #     print('mean', vary[0:3])
        #     print('beta', beta[0:3])
        #     if (i + 1) % 10 == 0:
        #         for i in range(10):
        #             val_loss = sess.run(val_total_loss)
        #             vary, beta = sess.run([vary_23, beta_23])
        #             print('mean val', vary[0:3])
        #             print('beta', beta[0:3])
        #             print('VAl_loss', val_loss)

        model_init_fn(sess)
        saver = tf.train.Saver()
        eval_planner = EvalPlanner(n_epochs, val_every)
        progress = sly.progress_counter_train(n_epochs, iters_cnt['train'])
        best_val_loss = float('inf')
        epoch_flt = 0

        for epoch in range(n_epochs):
            logger.info("Before new epoch", extra={'epoch': epoch_flt})
            for train_it in range(iters_cnt['train']):
                total_loss = sess.run(train_tensor)

                metrics_values_train = {
                    'loss': total_loss,
                }

                progress.iter_done_report()
                epoch_flt = epoch_float(epoch, train_it + 1, iters_cnt['train'])
                sly.report_metrics_training(epoch_flt, metrics_values_train)

                if eval_planner.need_validation(epoch_flt):
                    logger.info("Before validation", extra={'epoch': epoch_flt})

                    overall_val_loss = 0
                    for val_it in range(iters_cnt['val']):
                        overall_val_loss += sess.run(val_total_loss)

                        logger.info("Validation in progress", extra={'epoch': epoch_flt,
                                                                     'val_iter': val_it,
                                                                     'val_iters': iters_cnt['val']})

                    metrics_values_val = {
                        'loss': overall_val_loss / iters_cnt['val'],
                    }
                    sly.report_metrics_validation(epoch_flt, metrics_values_val)
                    logger.info("Validation has been finished", extra={'epoch': epoch_flt})

                    eval_planner.validation_performed()

                    val_loss = metrics_values_val['loss']
                    model_is_best = val_loss < best_val_loss
                    if model_is_best:
                        best_val_loss = val_loss
                        logger.info('It\'s been determined that current model is the best one for a while.')

                    save_cback(saver,
                               sess,
                               model_is_best,
                               opt_data={
                                   'epoch': epoch_flt,
                                   'val_metrics': metrics_values_val,
                               })

            logger.info("Epoch was finished", extra={'epoch': epoch_flt})
