"""Set up the preprocessing pipeline."""
import collections
import logging

import tensorflow as tf
from gp_tools.tf import one_hot

LOGGER = logging.getLogger(__name__)


#: Define what an example looks like for this model family.
Examples = collections.namedtuple(
    "Examples",
    "paths, inputs, targets, conditioning")


def transform(image, mode, config, nearest=False):
    """Apply all relevant image transformations."""
    r = image
    if nearest:
        rm = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    else:
        rm = tf.image.ResizeMethod.BILINEAR
    if config["flip"] and mode in ['train', 'trainval']:
        r = tf.image.random_flip_left_right(r, seed=config['seed'])
    if (config["scale_size"] != config["crop_size"] and
            mode in ['train', 'trainval']):
        r = tf.image.resize_images(r,
                                   [config["scale_size"],
                                    config["scale_size"]],
                                   method=rm)
        if config["scale_size"] > config["crop_size"]:
            offset = tf.cast(tf.floor(
                tf.random_uniform([2],
                                  0,
                                  config["scale_size"] -
                                  config["crop_size"] + 1,
                                  seed=config["seed"])),
                             dtype=tf.int32)
            r = r[offset[0]:offset[0] + config["crop_size"],
                  offset[1]:offset[1] + config["crop_size"],
                  :]
        elif config["scale_size"] < config["crop_size"]:
            raise Exception("scale size cannot be less than crop size")
    else:
        if (image.get_shape().as_list()[0] != config["crop_size"] or
                image.get_shape().as_list()[1] != config["crop_size"]):
            r = tf.image.resize_images(r,
                                       [config["crop_size"],
                                        config["crop_size"]],
                                       method=rm)
    return r


def prepare(load_dict, mode, config):
    conditioning = None
    paths = load_dict['original_filename']
    if config["model_version"] in ['vae', 'cvae']:
        # Ignore the image. Only care for the labels.
        if config["input_as_class"]:
            labels = load_dict['labels']
            labels.set_shape((config["scale_size"], config["scale_size"], 3))
            labels = labels[:, :, 0]
            labels = tf.cast(one_hot(labels, 22), tf.float32) - 0.5
            labels = transform(labels, mode, config, True)
            labels.set_shape((config["crop_size"], config["crop_size"], 22))
        else:
            labels = load_dict['label_vis']
            labels.set_shape((config["scale_size"], config["scale_size"], 3))
            labels = tf.cast(labels, tf.float32)
            labels = labels * 2. / 255. - 1.
            labels = transform(labels, mode, config)
            labels.set_shape((config["crop_size"], config["crop_size"], 3))
        inputs = labels
        # Conditioning?
        if config["model_version"] == 'cvae':
            if config["conditioning_as_class"]:
                conditioning = load_dict["bodysegments"]
                conditioning.set_shape((config["scale_size"],
                                        config["scale_size"], 3))
                conditioning = conditioning[:, :, 0]
                conditioning = tf.cast(one_hot(conditioning, 7),
                                       tf.float32) - 0.5
                conditioning = transform(conditioning, mode, config, True)
                conditioning.set_shape((config["crop_size"],
                                        config["crop_size"], 7))
            else:
                conditioning = load_dict["bodysegments_vis"]
                conditioning.set_shape((config["scale_size"],
                                        config["scale_size"], 3))
                conditioning = (tf.cast(conditioning, tf.float32) *
                                2. / 255. - 1.)
                conditioning = transform(conditioning, mode, config)
                conditioning.set_shape((config["crop_size"],
                                        config["crop_size"], 3))
    else:
        if config["input_as_class"]:
            inputs = load_dict['labels']
            inputs.set_shape((config["scale_size"], config["scale_size"], 3))
            inputs = inputs[:, :, 0]
            inputs = tf.cast(one_hot(inputs, 22), tf.float32) - 0.5
            inputs = transform(inputs, mode, config, True)
            inputs.set_shape((config["crop_size"], config["crop_size"], 22))
        else:
            inputs = load_dict['label_vis']
            inputs.set_shape((config["scale_size"], config["scale_size"], 3))
            inputs = tf.cast(inputs, tf.float32) * 2. / 255. - 1.
            inputs = transform(inputs, mode, config)
            inputs.set_shape((config["crop_size"], config["crop_size"], 3))
        labels = load_dict['image']
        labels.set_shape((config["scale_size"], config["scale_size"], 3))
        labels = tf.cast(labels, tf.float32) * 2. / 255. - 1.
        labels = transform(labels, mode, config)
        labels.set_shape((config["crop_size"], config["crop_size"], 3))
        if config.get("portray_additional_conditioning", None):
            conditioning = load_dict[config['portray_additional_conditioning']]
            conditioning.set_shape((config["scale_size"],
                                    config["scale_size"], 3))
            conditioning = (tf.cast(conditioning, tf.float32) *
                            2. / 255. - 1.)
            conditioning = transform(conditioning, mode, config)
            conditioning.set_shape((config["crop_size"],
                                    config["crop_size"], 3))
    inputs = tf.cast(inputs, tf.float32)
    labels = tf.cast(labels, tf.float32)
    if conditioning is not None:
        conditioning = tf.cast(conditioning, tf.float32)
        return paths, inputs, labels, conditioning
    else:
        return paths, inputs, labels


def preprocess(nsamples, loader_list, mode, config):
    if mode == "sample" and config["model_version"] != 'cvae':
        return Examples(
            paths=None,
            inputs=None,
            targets=None,
            conditioning=None
        )
    """
    if mode == 'sample' and config["model_version"] == 'cvae':
        LOGGER.info("Producing %d samples per image.",
                    config["cvae_nsamples_per_image"])
        mult_input_paths = []
        for fp in input_paths:
            mult_input_paths.extend([fp] * config["cvae_nsamples_per_image"])
        input_paths = mult_input_paths
    """
    min_after_dequeue = 200
    capacity = min_after_dequeue + 4 * config["batch_size"]
    with tf.name_scope("preprocess"):
        example_list = [prepare(load_dict, mode, config)
                        for load_dict in loader_list]
    if mode in ['train', 'trainval']:
        prep_tuple = tf.train.shuffle_batch_join(
            example_list,
            batch_size=config["batch_size"],
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    else:
        real_bs = config["batch_size"]
        if mode == 'sample' and config["model_version"] == 'cvae':
            assert config["batch_size"] % config["cvae_nsamples_per_image"] == 0, (
                "cvae_nsamples_per_image must be a divisor of batch_size!")
            real_bs = config["batch_size"] // config["cvae_nsamples_per_image"]
        prep_tuple = tf.train.batch(
            example_list[0],
            batch_size=real_bs,
            capacity=capacity,
            num_threads=1,  # For determinism.
            allow_smaller_final_batch=False)
    paths = prep_tuple[0]
    inputs = prep_tuple[1]
    targets = prep_tuple[2]
    if len(prep_tuple) > 3:
        conditioning = prep_tuple[3]
    else:
        conditioning = None
    if mode == 'sample':
        assert config["model_version"] in ['cvae', 'vae']
        targets = None
        inputs = None
        if config["model_version"] == 'vae':
            paths = None
        else:
            path_batches = [tf.string_join([tf.identity(paths), tf.constant("_s" + str(batch_idx))])
                            for batch_idx in range(config["cvae_nsamples_per_image"])]
            paths = tf.concat(path_batches, axis=0)
            conditioning = tf.concat([tf.identity(conditioning)
                                      for _ in range(config["cvae_nsamples_per_image"])],
                                     axis=0)
    return Examples(
        paths=paths,
        inputs=inputs,
        targets=targets,
        conditioning=conditioning,
    )
