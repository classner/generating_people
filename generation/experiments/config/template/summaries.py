"""Summarize the networks actions."""
import os.path as path
import logging
import clustertools.visualization as vs
import tensorflow as tf
import numpy as np
import cv2
import sys
sys.path.insert(0, path.join(path.dirname(__file__), '..', '..', '..', '..'))
from config import CMAP  # noqa: E402


LOGGER = logging.getLogger(__name__)


def postprocess_colormap(cls, postprocess=True):
    """Create a colormap out of the classes and postprocess the face."""
    batch = vs.apply_colormap(cls, vmin=0, vmax=21, cmap=CMAP)
    cmap = vs.apply_colormap(np.array(range(22), dtype='uint8'),
                             vmin=0, vmax=21, cmap=CMAP)
    COLSET = cmap[18:22]
    FCOL = cmap[11]
    if postprocess:
        kernel = np.ones((2, 2), dtype=np.uint8)
        for im in batch:
            for col in COLSET:
                # Extract the map of the matching color.
                colmap = np.all(im == col, axis=2).astype(np.uint8)
                # Erode.
                while np.sum(colmap) > 10:
                    colmap = cv2.erode(colmap, kernel)
                # Prepare the original map for remapping.
                im[np.all(im == col, axis=2)] = FCOL
                # Backproject.
                im[colmap == 1] = col
    return batch[:, :, :, :3]


def deprocess(config, image, argmax=False, postprocess=True):
    if argmax:
        def cfunc(x): return postprocess_colormap(x, postprocess)
        return tf.py_func(cfunc, [tf.argmax(image, 3)], tf.uint8)
    else:
        return tf.image.convert_image_dtype((image + 1) / 2, dtype=tf.uint8,
                                            saturate=True)


def create_summaries(mode, examples, model, config):
    LOGGER.info("Setting up summaries and fetches...")
    with tf.variable_scope("deprocessing"):
        # Deprocess images. #######################################################
        if mode != 'sample':
            # Inputs.
            with tf.name_scope("deprocess_inputs"):
                deprocessed_inputs = deprocess(config,
                                               examples.inputs,
                                               config["input_as_class"])
            # Targets.
            with tf.name_scope("deprocess_targets"):
                deprocessed_targets = deprocess(
                    config,
                    examples.targets,
                    (config["model_version"] in ['vae', 'cvae'] and
                     config["input_as_class"]))
        else:
            deprocessed_inputs = None
            deprocessed_targets = None
        if mode != 'sample' or config["model_version"] == 'cvae':
            paths = examples.paths
        else:
            paths = None
        if config["model_version"] == 'cvae':
            with tf.name_scope("deprocessed_conditioning"):
                deprocessed_conditioning = deprocess(
                    config,
                    examples.conditioning,
                    config["conditioning_as_class"])
        elif (config["model_version"] == 'portray' and
              examples.conditioning is not None):
            deprocessed_conditioning = deprocess(config,
                                                 examples.conditioning)
        else:
            deprocessed_conditioning = None
        with tf.name_scope("deprocess_outputs"):
            deprocessed_outputs = deprocess(
                config,
                model.outputs,
                (config["input_as_class"] and
                 config["model_version"] in ['vae', 'cvae']))
        if (config["input_as_class"] and
                config["model_version"] in ['vae', 'cvae']):
            with tf.name_scope("deprocess_unpostprocessed_outputs"):
                deprocessed_unpostprocessed_outputs = deprocess(
                    config, model.outputs, True, False)
        else:
            deprocessed_unpostprocessed_outputs = None
        # Encode the images. ######################################################
        display_fetches = dict()
        with tf.name_scope("encode_images"):
            for name, res in [
                    ('inputs', deprocessed_inputs),
                    ('conditioning', deprocessed_conditioning),
                    ('outputs', deprocessed_outputs),
                    ('unpostprocessed_outputs',
                     deprocessed_unpostprocessed_outputs),
                    ('targets', deprocessed_targets),
            ]:
                if res is not None:
                    display_fetches[name] = tf.map_fn(tf.image.encode_png,
                                                      res,
                                                      dtype=tf.string,
                                                      name=name+'_pngs')
    if mode != 'sample':
        display_fetches['y'] = model.y
        display_fetches['z'] = model.z

    if mode != 'sample' or config["model_version"] == 'cvae':
        display_fetches['paths'] = paths

    # Create the summaries. ###################################################
    if deprocessed_inputs is not None:
        with tf.name_scope("inputs_summary"):
            tf.summary.image("inputs", deprocessed_inputs)
    if deprocessed_targets is not None:
        with tf.name_scope("targets_summary"):
            tf.summary.image("targets", deprocessed_targets)
    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", deprocessed_outputs)
    if deprocessed_conditioning is not None:
        with tf.name_scope("conditioning_summary"):
            tf.summary.image("conditioning", deprocessed_conditioning)
    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real",
                         tf.image.convert_image_dtype(model.predict_real,
                                                      dtype=tf.uint8))
    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake",
                         tf.image.convert_image_dtype(model.predict_fake,
                                                      dtype=tf.uint8))
    tf.summary.histogram("z_mean", model.z_mean)
    tf.summary.histogram("z_log_sigma_sq", model.z_log_sigma_sq)
    tf.summary.histogram("z", model.z)

    if mode in ['train', 'trainval']:
        tf.summary.scalar("loss/discriminator", model.discrim_loss)
        tf.summary.scalar("loss/generator_GAN", model.gen_loss_GAN)
        tf.summary.scalar("loss/generator_recon", model.gen_loss_recon)
        tf.summary.scalar("loss/generator_latent", model.gen_loss_latent)
        tf.summary.scalar("loss/generator_accuracy", model.gen_accuracy)
        test_fetches = {}
    else:
        # These fetches will be evaluated and averaged at test time.
        test_fetches = {}
        test_fetches["loss/discriminator"] = model.discrim_loss
        test_fetches["loss/generator_GAN"] = model.gen_loss_GAN
        test_fetches["loss/generator_recon"] = model.gen_loss_recon
        test_fetches["loss/generator_latent"] = model.gen_loss_latent
        test_fetches["loss/generator_accuracy"] = model.gen_accuracy

    LOGGER.info("Summaries and fetches complete.")
    return display_fetches, test_fetches
