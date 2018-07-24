from object_detection.builders import model_builder
from object_detection.utils import dataset_util
from dataset_tools import build_dataset

import collections
import functools

import tensorflow as tf

from object_detection.builders import optimizer_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import batcher
from object_detection.core import preprocessor
from object_detection.utils import ops as util_ops
from object_detection.utils import variables_helper
from deployment import model_deploy

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.training import coordinator
from object_detection.core import standard_fields as fields

import supervisely_lib as sly
from supervisely_lib import logger
from common import EvalPlanner

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.ERROR)


def create_input_queue(batch_size_per_clone, create_tensor_dict_fn,
                       batch_queue_capacity, num_batch_queue_threads,
                       prefetch_queue_capacity, data_augmentation_options):
    """Sets up reader, prefetcher and returns input queue.

    Args:
      batch_size_per_clone: batch size to use per clone.
      create_tensor_dict_fn: function to create tensor dictionary.
      batch_queue_capacity: maximum number of elements to store within a queue.
      num_batch_queue_threads: number of threads to use for batching.
      prefetch_queue_capacity: maximum capacity of the queue used to prefetch
                               assembled batches.
      data_augmentation_options: a list of tuples, where each tuple contains a
        data augmentation function and a dictionary containing arguments and their
        values (see preprocessor.py).

    Returns:
      input queue: a batcher.BatchQueue object holding enqueued tensor_dicts
        (which hold images, boxes and targets).  To get a batch of tensor_dicts,
        call input_queue.Dequeue().
    """
    tensor_dict = create_tensor_dict_fn()
    tensor_dict[fields.InputDataFields.image] = tf.expand_dims(
        tensor_dict[fields.InputDataFields.image], 0)
    tensor_dict[fields.InputDataFields.groundtruth_boxes].set_shape([None, 4])
    tensor_dict[fields.InputDataFields.groundtruth_classes].set_shape([None])
    tensor_dict[fields.InputDataFields.num_groundtruth_boxes].set_shape(())
    images = tensor_dict[fields.InputDataFields.image]
    float_images = tf.to_float(images)
    tensor_dict[fields.InputDataFields.image] = float_images

    include_instance_masks = (fields.InputDataFields.groundtruth_instance_masks
                              in tensor_dict)
    include_keypoints = (fields.InputDataFields.groundtruth_keypoints
                         in tensor_dict)
    if data_augmentation_options:
        tensor_dict = preprocessor.preprocess(
            tensor_dict, data_augmentation_options,
            func_arg_map=preprocessor.get_default_func_arg_map(
                include_instance_masks=include_instance_masks,
                include_keypoints=include_keypoints))

    input_queue = batcher.BatchQueue(
        tensor_dict,
        batch_size=batch_size_per_clone,
        batch_queue_capacity=batch_queue_capacity,
        num_batch_queue_threads=num_batch_queue_threads,
        prefetch_queue_capacity=prefetch_queue_capacity)
    return input_queue


def get_inputs(input_queue, num_classes, merge_multiple_label_boxes=False):
  """Dequeues batch and constructs inputs to object detection model.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    num_classes: Number of classes.
    merge_multiple_label_boxes: Whether to merge boxes with multiple labels
      or not. Defaults to false. Merged boxes are represented with a single
      box and a k-hot encoding of the multiple labels associated with the
      boxes.

  Returns:
    images: a list of 3-D float tensor of images.
    image_keys: a list of string keys for the images.
    locations_list: a list of tensors of shape [num_boxes, 4]
      containing the corners of the groundtruth boxes.
    classes_list: a list of padded one-hot tensors containing target classes.
    masks_list: a list of 3-D float tensors of shape [num_boxes, image_height,
      image_width] containing instance masks for objects if present in the
      input_queue. Else returns None.
    keypoints_list: a list of 3-D float tensors of shape [num_boxes,
      num_keypoints, 2] containing keypoints for objects if present in the
      input queue. Else returns None.
    weights_lists: a list of 1-D float32 tensors of shape [num_boxes]
      containing groundtruth weight for each box.
  """
  read_data_list = input_queue.dequeue()
  label_id_offset = 1
  def extract_images_and_targets(read_data):
    """Extract images and targets from the input dict."""
    image = read_data[fields.InputDataFields.image]
    key = ''
    if fields.InputDataFields.source_id in read_data:
      key = read_data[fields.InputDataFields.source_id]
    location_gt = read_data[fields.InputDataFields.groundtruth_boxes]
    classes_gt = tf.cast(read_data[fields.InputDataFields.groundtruth_classes],
                         tf.int32)
    classes_gt -= label_id_offset
    if merge_multiple_label_boxes:
      location_gt, classes_gt, _ = util_ops.merge_boxes_with_multiple_labels(
          location_gt, classes_gt, num_classes)
    else:
      classes_gt = util_ops.padded_one_hot_encoding(
          indices=classes_gt, depth=num_classes, left_pad=0)
    masks_gt = read_data.get(fields.InputDataFields.groundtruth_instance_masks)
    keypoints_gt = read_data.get(fields.InputDataFields.groundtruth_keypoints)
    if (merge_multiple_label_boxes and (
        masks_gt is not None or keypoints_gt is not None)):
      raise NotImplementedError('Multi-label support is only for boxes.')
    weights_gt = read_data.get(
        fields.InputDataFields.groundtruth_weights)
    return (image, key, location_gt, classes_gt, masks_gt, keypoints_gt,
            weights_gt)

  return zip(*map(extract_images_and_targets, read_data_list))


def _create_losses(input_queue, create_model_fn, train_config):
    """Creates loss function for a DetectionModel.

    Args:
      input_queue: BatchQueue object holding enqueued tensor_dicts.
      create_model_fn: A function to create the DetectionModel.
      train_config: a train_pb2.TrainConfig protobuf.
    """
    detection_model = create_model_fn()
    (images, _, groundtruth_boxes_list, groundtruth_classes_list,
     groundtruth_masks_list, groundtruth_keypoints_list, _) = get_inputs(
        input_queue,
        detection_model.num_classes,
        train_config.merge_multiple_label_boxes)

    preprocessed_images = []
    true_image_shapes = []
    for image in images:
        resized_image, true_image_shape = detection_model.preprocess(image)
        preprocessed_images.append(resized_image)
        true_image_shapes.append(true_image_shape)

    images = tf.concat(preprocessed_images, 0)
    true_image_shapes = tf.concat(true_image_shapes, 0)

    if any(mask is None for mask in groundtruth_masks_list):
        groundtruth_masks_list = None
    if any(keypoints is None for keypoints in groundtruth_keypoints_list):
        groundtruth_keypoints_list = None

    detection_model.provide_groundtruth(groundtruth_boxes_list,
                                        groundtruth_classes_list,
                                        groundtruth_masks_list,
                                        groundtruth_keypoints_list)
    prediction_dict = detection_model.predict(images, true_image_shapes)

    losses_dict = detection_model.loss(prediction_dict, true_image_shapes)
    for loss_tensor in losses_dict.values():
        tf.losses.add_loss(loss_tensor)


def _create_losses_val(input_queue, create_model_fn, train_config):
    """Creates loss function for a DetectionModel.

    Args:
      input_queue: BatchQueue object holding enqueued tensor_dicts.
      create_model_fn: A function to create the DetectionModel.
      train_config: a train_pb2.TrainConfig protobuf.
    """
    detection_model = create_model_fn()
    (images, _, groundtruth_boxes_list, groundtruth_classes_list,
     groundtruth_masks_list, groundtruth_keypoints_list, _) = get_inputs(
        input_queue,
        detection_model.num_classes,
        train_config.merge_multiple_label_boxes)

    preprocessed_images = []
    true_image_shapes = []
    for image in images:
        resized_image, true_image_shape = detection_model.preprocess(image)
        preprocessed_images.append(resized_image)
        true_image_shapes.append(true_image_shape)

    images = tf.concat(preprocessed_images, 0)
    true_image_shapes = tf.concat(true_image_shapes, 0)

    if any(mask is None for mask in groundtruth_masks_list):
        groundtruth_masks_list = None
    if any(keypoints is None for keypoints in groundtruth_keypoints_list):
        groundtruth_keypoints_list = None

    detection_model.provide_groundtruth(groundtruth_boxes_list,
                                        groundtruth_classes_list,
                                        groundtruth_masks_list,
                                        groundtruth_keypoints_list)
    prediction_dict = detection_model.predict(images, true_image_shapes)

    losses_dict = detection_model.loss(prediction_dict, true_image_shapes)
    losses = []
    for loss_tensor in losses_dict.values():
        losses.append(loss_tensor)
    return losses


# float progress of training, since zero
def epoch_float(epoch, train_it, train_its):
    res = epoch + train_it / float(train_its)
    return res


def configs_from_pipeline(pipeline_config):
    configs = {}
    configs["model"] = pipeline_config.model
    configs["train_config"] = pipeline_config.train_config
    configs["train_input_config"] = pipeline_config.train_input_reader
    configs["eval_config"] = pipeline_config.eval_config
    configs["eval_input_config"] = pipeline_config.eval_input_reader

    return configs


def get_val_loss(num_clones, input_queue, create_model_fn, train_config):
    losses_val = []
    # with tf.name_scope("Valid"):
    with slim.arg_scope([slim.model_variable, slim.variable], device='/device:CPU:0'):
        for curr_dev_id in range(num_clones):
            with tf.device('/gpu:{}'.format(curr_dev_id)):
                with tf.name_scope('clone_{}'.format(curr_dev_id)) as scope:
                    with tf.variable_scope(tf.get_variable_scope(),
                                           reuse=True):
                        losses = _create_losses_val(input_queue, create_model_fn, train_config)
                        clones_loss = tf.add_n(losses)
                        clones_loss = tf.divide(clones_loss, 1.0 * num_clones)
                        losses_val.append(clones_loss)
    total_val_loss = tf.add_n(losses_val, name='val_loss')
    return total_val_loss


def train(datasets_dicts,
          epochs,
          val_every,
          iters_cnt,
          validate_with_eval_model,
          pipeline_config,
          num_clones=1,
          save_cback=None):
    logger.info('Start train')
    configs = configs_from_pipeline(pipeline_config)

    model_config = configs['model']
    train_config = configs['train_config']

    create_model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)
    detection_model = create_model_fn()

    def get_next(dataset):
        return dataset_util.make_initializable_iterator(
            build_dataset(dataset)).get_next()

    create_tensor_dict_fn = functools.partial(get_next, datasets_dicts['train'])
    create_tensor_dict_fn_val = functools.partial(get_next, datasets_dicts['val'])

    data_augmentation_options = [
        preprocessor_builder.build(step)
        for step in train_config.data_augmentation_options]

    with tf.Graph().as_default():
        # Build a configuration specifying multi-GPU and multi-replicas.
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=4,
            clone_on_cpu=False,
            replica_id=0,
            num_replicas=1,
            num_ps_tasks=0,
            worker_job_name='lonely_worker')

        # Place the global step on the device storing the variables.
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        with tf.device(deploy_config.inputs_device()):
            coord = coordinator.Coordinator()
            input_queue = create_input_queue(
                train_config.batch_size, create_tensor_dict_fn,
                train_config.batch_queue_capacity,
                train_config.num_batch_queue_threads,
                train_config.prefetch_queue_capacity, data_augmentation_options)

            input_queue_val = create_input_queue(
                train_config.batch_size, create_tensor_dict_fn_val,
                train_config.batch_queue_capacity,
                train_config.num_batch_queue_threads,
                train_config.prefetch_queue_capacity, data_augmentation_options)

        # create validation graph
        create_model_fn_val = functools.partial(
            model_builder.build,
            model_config=model_config,
            is_training=not validate_with_eval_model)

        with tf.device(deploy_config.optimizer_device()):
            training_optimizer, optimizer_summary_vars = optimizer_builder.build(
                train_config.optimizer)
            for var in optimizer_summary_vars:
                tf.summary.scalar(var.op.name, var, family='LearningRate')

        train_losses = []
        grads_and_vars = []
        with slim.arg_scope([slim.model_variable, slim.variable], device='/device:CPU:0'):
            for curr_dev_id in range(num_clones):
                with tf.device('/gpu:{}'.format(curr_dev_id)):
                    with tf.name_scope('clone_{}'.format(curr_dev_id)) as scope:
                        with tf.variable_scope(tf.get_variable_scope(),
                                               reuse=True if curr_dev_id > 0 else None):
                            losses = _create_losses_val(input_queue, create_model_fn, train_config)
                            clones_loss = tf.add_n(losses)
                            clones_loss = tf.divide(clones_loss, 1.0 * num_clones)
                            grads = training_optimizer.compute_gradients(clones_loss)
                            train_losses.append(clones_loss)
                            grads_and_vars.append(grads)
                            if curr_dev_id == 0:
                                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        val_total_loss = get_val_loss(num_clones, input_queue_val, create_model_fn_val, train_config)

        with tf.device(deploy_config.optimizer_device()):
            total_loss = tf.add_n(train_losses)
            grads_and_vars = model_deploy._sum_clones_gradients(grads_and_vars)
            total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

            # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
            if train_config.bias_grad_multiplier:
                biases_regex_list = ['.*/biases']
                grads_and_vars = variables_helper.multiply_gradients_matching_regex(
                    grads_and_vars,
                    biases_regex_list,
                    multiplier=train_config.bias_grad_multiplier)

            # Optionally freeze some layers by setting their gradients to be zero.
            if train_config.freeze_variables:
                grads_and_vars = variables_helper.freeze_gradients_matching_regex(
                    grads_and_vars, train_config.freeze_variables)

            # Optionally clip gradients
            if train_config.gradient_clipping_by_norm > 0:
                with tf.name_scope('clip_grads'):
                    grads_and_vars = slim.learning.clip_gradient_norms(
                        grads_and_vars, train_config.gradient_clipping_by_norm)

            # Create gradient updates.
            grad_updates = training_optimizer.apply_gradients(grads_and_vars,
                                                              global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops, name='update_barrier')
            with tf.control_dependencies([update_op]):
                train_tensor = tf.identity(total_loss, name='train_op')

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        coord.clear_stop()
        sess = tf.Session(config=config)
        saver = tf.train.Saver()

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

        logger.info('Start restore')
        if train_config.fine_tune_checkpoint:
            var_map = detection_model.restore_map(
                            fine_tune_checkpoint_type=train_config.fine_tune_checkpoint_type,
                            load_all_detection_checkpoint_vars=(
                                train_config.load_all_detection_checkpoint_vars))
            available_var_map = (variables_helper.
                                    get_variables_available_in_checkpoint(
                                    var_map, train_config.fine_tune_checkpoint))
            if 'global_step' in available_var_map:
                del available_var_map['global_step']
            init_saver = tf.train.Saver(available_var_map)
            logger.info('Restoring model weights from previous checkpoint.')
            init_saver.restore(sess, train_config.fine_tune_checkpoint)
            logger.info('Model restored.')

        eval_planner = EvalPlanner(epochs, val_every)
        progress = sly.progress_counter_train(epochs, iters_cnt['train'])
        best_val_loss = float('inf')
        epoch_flt = 0

        for epoch in range(epochs):
            logger.info("Before new epoch", extra={'epoch': epoch_flt})
            for train_it in range(iters_cnt['train']):
                total_loss, np_global_step = sess.run([train_tensor, global_step])

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
        coord.request_stop()
        coord.join(threads)
