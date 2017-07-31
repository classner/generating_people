#!/usr/bin/env python2
"""Main control for the experiments."""
import os
import os.path as path
import ast
import sys
from glob import glob
import signal
import imp
import logging
import time
import numpy as np
import socket

import tqdm
import click
import tensorflow as tf
from tensorflow.python.client import timeline

from clustertools.log import LOGFORMAT
import data
sys.path.insert(0, path.join('..'))
from config import EXP_DATA_FP  # noqa: E402


LOGGER = logging.getLogger(__name__)
EXP_DIR = path.join(path.dirname(__file__), 'experiments')


@click.command()
@click.argument("mode",
                type=click.Choice([
                    "train", "val", "trainval", "test", "sample", "transform"]
                ))
@click.argument("exp_name",
                type=click.Path(exists=True, writable=True, file_okay=False))
@click.option("--num_threads", type=click.INT, default=8,
              help="Number of data preprocessing threads.")
@click.option("--no_checkpoint", type=click.BOOL, is_flag=True,
              help="Ignore checkpoints.")
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Checkpoint to use for restoring (+.meta).")
@click.option("--n_samples", type=click.INT, default=100,
              help="The number of samples to sample.")
@click.option("--out_fp", type=click.Path(writable=True), default=None,
              help="If specified, write test or sample results there.")
@click.option("--override_dset_suffix", type=click.STRING, default=None,
              help="If specified, override the configure dset_suffix.")
@click.option("--custom_options", type=click.STRING, default="",
              help="Provide model specific custom options.")
@click.option("--no_output", type=click.BOOL, default=False, is_flag=True,
              help="Don't store results in test modes.")
def cli(**args):
    """Main control for the experiments."""
    mode = args['mode']
    exp_name = args['exp_name'].strip("/")
    assert exp_name.startswith(path.join("experiments", "config"))
    exp_purename = path.basename(exp_name)
    exp_feat_fp = path.join("experiments", "features", exp_purename)
    exp_log_fp = path.join("experiments", "states", exp_purename)
    if not path.exists(exp_feat_fp):
        os.makedirs(exp_feat_fp)
    if not path.exists(exp_log_fp):
        os.makedirs(exp_log_fp)
    # Set up file logging.
    fh = logging.FileHandler(path.join(exp_log_fp, 'run.py.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(LOGFORMAT)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)
    LOGGER.info("Running on host: %s", socket.getfqdn())
    if "JOB_ID" in os.environ.keys():
        LOGGER.info("Condor job id: %s", os.environ["JOB_ID"])
    LOGGER.info("Running mode `%s` for experiment `%s`.", mode, exp_name)
    # Configuration.
    exp_config_mod = imp.load_source('_exp_config',
                                     path.join(exp_name, 'config.py'))
    exp_config = exp_config_mod.adjust_config(
        exp_config_mod.get_config(), mode)
    assert mode in exp_config["supp_modes"], (
        "Unsupported mode by this model: %s, available: %s." % (
            mode, str(exp_config["supp_modes"])))
    if args["override_dset_suffix"] is not None:
        LOGGER.warn("Overriding dset suffix to `%s`!",
                    args["override_dset_suffix"])
        exp_config["dset_suffix"] = args["override_dset_suffix"]
    if args['custom_options'] != '':
        custom_options = ast.literal_eval(args["custom_options"])
        exp_config.update(custom_options)
    exp_config['num_threads'] = args["num_threads"]
    exp_config['n_samples'] = args["n_samples"]
    LOGGER.info("Configuration:")
    for key, val in exp_config.items():
        LOGGER.info("%s = %s", key, val)
    # Data setup.
    with tf.device('/cpu:0'):
        nsamples, steps_per_epoch, loader_list = \
            data.get_dataset(EXP_DATA_FP, mode, exp_config)
    LOGGER.info("%d examples prepared, %d steps per epoch.",
                nsamples, steps_per_epoch)
    LOGGER.info("Setting up preprocessing...")
    exp_prep_mod = imp.load_source('_exp_preprocessing',
                                   path.join(exp_name, 'preprocessing.py'))
    with tf.device('/cpu:0'):
        examples = exp_prep_mod.preprocess(nsamples, loader_list, mode,
                                           exp_config)
    # Checkpointing.
    if args['no_checkpoint']:
        assert args['checkpoint'] is None
    if not args["no_checkpoint"]:
        LOGGER.info("Looking for checkpoints...")
        if args['checkpoint'] is not None:
            checkpoint = args['checkpoint'][:-5]
        else:
            checkpoint = tf.train.latest_checkpoint(exp_log_fp)
        if checkpoint is None:
            LOGGER.info("No checkpoint found. Continuing without.")
            checkpoint, load_session, load_graph = None, None, None
        else:
            load_graph = tf.Graph()
            with load_graph.as_default():
                meta_file = checkpoint + '.meta'
                LOGGER.info("Restoring graph from `%s`...", meta_file)
                rest_saver = tf.train.import_meta_graph(meta_file,
                                                        clear_devices=True)
                LOGGER.info("Graph restored. Loading checkpoint `%s`...",
                            checkpoint)
                load_session = tf.Session()
                rest_saver.restore(load_session, checkpoint)
    else:
        checkpoint, load_session, load_graph = None, None, None
    if mode not in ['train', 'trainval'] and load_session is None:
        raise Exception("The mode %s requires a checkpoint!" % (mode))
    # Build model.
    model_mod = imp.load_source('_model',
                                path.join(exp_name, 'model.py'))
    model = model_mod.create_model(
            mode, examples, exp_config, (load_session, load_graph))
    del load_graph  # Free graph.
    if load_session is not None:
        load_session.close()
    # Setup summaries.
    summary_mod = imp.load_source('_summaries',
                                  path.join(exp_name, 'summaries.py'))
    display_fetches, test_fetches = summary_mod.create_summaries(
        mode, examples, model, exp_config)
    # Stats.
    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v))
                                         for v in tf.trainable_variables()])
    # Prepare output.
    out_mod = imp.load_source("_write_output",
                              path.join(exp_name, 'write_output.py'))
    # Preparing session.
    if mode in ['train', 'trainval']:
        saver = tf.train.Saver(max_to_keep=exp_config["kept_saves"])
    else:
        saver = None
    sess_config = tf.ConfigProto(log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sw = tf.summary.FileWriter(path.join(exp_log_fp, mode))
    summary_op = tf.summary.merge_all()
    prepared_session = tf.Session(config=sess_config)
    initializer = tf.global_variables_initializer()
    epoch = 0
    with prepared_session as sess:
        # from tensorflow.python import debug as tf_debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        LOGGER.info("Parameter count: %d.", sess.run(parameter_count))
        LOGGER.info("Starting queue runners...")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        LOGGER.info("Initializing variables...")
        sess.run(initializer)
        fetches = {}
        fetches["global_step"] = model.global_step
        global_step = sess.run(fetches)["global_step"][0]
        LOGGER.info("On global step: %d.", global_step)
        if len(glob(path.join(exp_log_fp, mode, 'events.*'))) == 0:
            LOGGER.info("Summarizing graph...")
            sw.add_graph(sess.graph, global_step=global_step)
        if mode in ['val', 'test']:
            image_dir = path.join(exp_feat_fp, 'step_' + str(global_step))
        elif mode in ['sample']:
            image_dir = path.join(exp_feat_fp,
                                  time.strftime("%Y-%m-%d_%H-%M-%S",
                                                time.gmtime()))
        else:
            image_dir = exp_log_fp
        if args["out_fp"] is not None:
            image_dir = args["out_fp"]
        if not args["no_output"]:
            LOGGER.info("Writing image status to `%s`.", image_dir)
        else:
            image_dir = None
        if mode in ['val', 'test', 'sample', 'transform']:
            shutdown_requested = [False]
            def SIGINT_handler(signal, frame):  # noqa: E306
                LOGGER.warn("Received SIGINT.")
                shutdown_requested[0] = True
            signal.signal(signal.SIGINT, SIGINT_handler)
            # run a single epoch over all input data
            if mode in ['val', 'test', 'transform']:
                num_ex = steps_per_epoch
            else:
                if exp_config['model_version'] == 'cvae':
                    num_ex = steps_per_epoch * exp_config["cvae_nsamples_per_image"]
                else:
                    num_ex = int(np.ceil(float(args["n_samples"]) /
                                         exp_config["batch_size"]))
            av_results = dict((name, []) for name in test_fetches.keys())
            av_placeholders = dict((name, tf.placeholder(tf.float32))
                                   for name in test_fetches.keys())
            for name in test_fetches.keys():
                tf.summary.scalar(name, av_placeholders[name],
                                  collections=['evaluation'])
            test_summary = tf.summary.merge_all('evaluation')
            display_fetches.update(test_fetches)
            for b_id in tqdm.tqdm(range(num_ex)):
                results = sess.run(display_fetches)
                if not args['no_output']:
                    index_fp = out_mod.save_images(results, image_dir, mode,
                                                   exp_config, batch=b_id)
                # Check for problems with this result.
                results_valid = True
                for key in test_fetches.keys():
                    if not np.isfinite(results[key]):
                        if 'paths' in results.keys():
                            LOGGER.warn("There's a problem with results for "
                                        "%s! Skipping.", results['paths'][0])
                        else:
                            LOGGER.warn("Erroneous result for batch %d!",
                                        b_id)
                        results_valid = False
                        break
                if results_valid:
                    for key in test_fetches.keys():
                        av_results[key].append(results[key])
                if shutdown_requested[0]:
                    break
            LOGGER.info("Results:")
            feed_results = dict()
            for key in test_fetches.keys():
                av_results[key] = np.mean(av_results[key])
                feed_results[av_placeholders[key]] = av_results[key]
                LOGGER.info("  %s: %s", key, av_results[key])
            if not shutdown_requested[0]:
                sw.add_summary(sess.run(test_summary, feed_dict=feed_results),
                               global_step)
            else:
                LOGGER.warn("Not writing results to tf summary due to "
                            "incomplete evaluation.")
            if not args['no_output']:
                LOGGER.info("Wrote index at `%s`.", index_fp)
        elif mode in ["train", "trainval"]:
            # Training.
            max_steps = 2**32
            last_summary_written = time.time()
            if exp_config["max_epochs"] is not None:
                max_steps = steps_per_epoch * exp_config["max_epochs"]
            if exp_config["max_steps"] is not None:
                max_steps = exp_config["max_steps"]
            shutdown_requested = [False]  # Needs to be mutable to access.
            # Register signal handler to save on Ctrl-C.
            def SIGINT_handler(signal, frame):  # noqa: E306
                LOGGER.warn("Received SIGINT. Saving model...")
                saver.save(sess,
                           path.join(exp_log_fp, "model"),
                           global_step=model.global_step)
                shutdown_requested[0] = True
            signal.signal(signal.SIGINT, SIGINT_handler)
            pbar = tqdm.tqdm(total=(max_steps - global_step) *
                             exp_config["batch_size"])
            for step in range(global_step, max_steps):
                def should(freq, epochs=False):
                    if epochs:
                        return freq > 0 and ((epoch + 1) % freq == 0 and
                                             (step + 1) % steps_per_epoch == 0
                                             or step == max_steps - 1)
                    else:
                        return freq > 0 and ((step + 1) % freq == 0 or
                                             step == max_steps - 1)
                options = None
                run_metadata = None
                if should(exp_config["trace_freq"]):
                    options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                # Setup fetches.
                fetches = {
                    "train": model.train,
                    "global_step": model.global_step,
                }
                if ((time.time() - last_summary_written) >
                        exp_config["summary_freq"]):
                    fetches["summary"] = summary_op
                if (should(exp_config["display_freq"], epochs=True) or
                    should(exp_config["save_freq"], epochs=True) or
                        step == max_steps - 1):
                    fetches["display"] = display_fetches
                # Run!
                results = sess.run(fetches, options=options,
                                   run_metadata=run_metadata)
                # Write.
                if (should(exp_config["save_freq"], epochs=True) or
                    results["global_step"] == 1 or
                        step == max_steps - 1):
                    # Save directly at first iteration to make sure this is
                    # working.
                    LOGGER.info("Saving model...")
                    gs = model.global_step
                    saver.save(sess,
                               path.join(exp_log_fp, "model"),
                               global_step=gs)
                if "summary" in results.keys():
                    sw.add_summary(results["summary"],
                                   results["global_step"])
                    last_summary_written = time.time()
                if "display" in results.keys():
                    LOGGER.info("saving display images")
                    out_mod.save_images(results["display"],
                                        image_dir,
                                        mode,
                                        exp_config,
                                        step=results["global_step"][0])
                if should(exp_config["trace_freq"]):
                    LOGGER.info("recording trace")
                    sw.add_run_metadata(
                        run_metadata, "step_%d" % results["global_step"])
                    trace = timeline.Timeline(
                        step_stats=run_metadata.step_stats)
                    with open(path.join(
                            exp_log_fp,
                            "timeline.json"), "w") as trace_file:
                        trace_file.write(trace.generate_chrome_trace_format())
                    # Enter 'chrome://tracing' in chrome to open the file.
                epoch = results["global_step"] // steps_per_epoch
                pbar.update(exp_config["batch_size"])
                if shutdown_requested[0]:
                    break
            pbar.close()
        LOGGER.info("Shutting down...")
        coord.request_stop()
        coord.join(threads)
    data.cleanup()
    LOGGER.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    logging.getLogger("clustertools.db.tools").setLevel(logging.WARN)
    logging.getLogger("PIL.Image").setLevel(logging.WARN)
    cli()
