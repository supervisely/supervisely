import tensorflow as tf
from google.protobuf import text_format
import sys
sys.path.append('./models/research/object_detection/')
from protos import pipeline_pb2 as pb


def load_sample_config(BASE_CONFIG_FILEPATH):
    config = pb.TrainEvalPipelineConfig()
    with tf.gfile.GFile(BASE_CONFIG_FILEPATH, 'r') as f:
        config = text_format.Merge(f.read(), config)
    return config


def default(d, key, value):
    return value if not key in d else d[key]


def remake_mask_rcnn_config(config, train_input, train_steps, n_classes, size, batch_size, lr, checkpoint=None):
    config.model.faster_rcnn.num_classes = n_classes

    config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = size[0]
    config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = size[1]

    config.train_config.batch_size = batch_size
    config.train_config.optimizer.adam_optimizer.learning_rate.constant_learning_rate.learning_rate = lr
    config.train_config.num_steps = train_steps
    if checkpoint:
        config.train_config.fine_tune_checkpoint = checkpoint
        config.train_config.from_detection_checkpoint = True
    else:
        config.train_config.fine_tune_checkpoint = ""
        config.train_config.from_detection_checkpoint = False

    config.train_input_reader.tf_record_input_reader.input_path[0] = train_input
    return config


def save_config(filepath, config):
    with open(filepath, 'w') as f:
        f.write(text_format.MessageToString(config))

