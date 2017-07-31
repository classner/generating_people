"""Common loading infrastructure."""
import tempfile
import subprocess
import shutil
import math
import os.path as path
from glob import glob
import logging
import clustertools.db.tools as cdbt
import tensorflow as tf


LOGGER = logging.getLogger(__name__)
TMP_DIRS = []


def prepare(path_queue, data_def, config):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(path_queue)
    return cdbt.decode_tf_tensors(
        serialized_example, data_def[0], data_def[1], as_list=False)


def get_dataset(EXP_DATA_FP, mode, config):
    global TMP_DIRS
    suffix = config["dset_suffix"]
    dataset_dir = path.join(EXP_DATA_FP, config['dset_type'], suffix)
    LOGGER.info("Preparing dataset from `%s`.", dataset_dir)
    if not path.exists(dataset_dir):
        raise Exception("input_dir does not exist: %s." % (dataset_dir))
    input_paths = []
    if mode in ['train', 'trainval']:
        input_paths.extend(glob(path.join(dataset_dir, "train_p*.tfrecords")))
    if mode in ['trainval', 'val', 'sample']:
        input_paths.extend(glob(path.join(dataset_dir, 'val_p*.tfrecords')))
    if mode in ['test']:
        input_paths.extend(glob(path.join(dataset_dir, 'test_p*.tfrecords')))
    if len(input_paths) == 0:
        # Assume that it's a directory instead of tfrecord structures.
        # Create a temporary directory with tfrecord files.
        tmp_dir = tempfile.mkdtemp(dir=dataset_dir, prefix='tmp_')
        TMP_DIRS.append(tmp_dir)
        input_paths = []
        if mode in ['train', 'trainval']:
            input_paths.extend(glob(path.join(dataset_dir, "train")))
        if mode in ['trainval', 'val', 'sample']:
            input_paths.extend(glob(path.join(dataset_dir, 'val')))
        if mode in ['test']:
            input_paths.extend(glob(path.join(dataset_dir, 'test')))
        for fp in input_paths:
            subprocess.check_call([
                "tfrpack",
                fp,
                "--out_fp",
                path.join(tmp_dir, path.basename(fp)),
            ])
        input_paths = glob(path.join(tmp_dir, "*.tfrecords"))
    if len(input_paths) == 0:
        raise Exception("`%s` contains no files for this mode" % (dataset_dir))
    nsamples, colnames, coltypes = cdbt.scan_tfdb(input_paths)
    data_def = (colnames, coltypes)
    LOGGER.info("Found %d samples with columns: %s.", nsamples, data_def[0])
    with tf.name_scope("load_data"):
        path_queue = tf.train.string_input_producer(
            input_paths,
            capacity=200 + 8 * config["batch_size"],
            shuffle=(mode == "train"))
        if mode in ['train', 'trainval']:
            # Prepare for shuffling.
            loader_list = [prepare(path_queue, data_def, config)
                           for _ in range(config["num_threads"])]
        else:
            loader_list = [prepare(path_queue, data_def, config)]
    steps_per_epoch = int(math.ceil(float(nsamples) / config["batch_size"]))
    return nsamples, steps_per_epoch, loader_list


def cleanup():
    """Deletes temprorary directories if necessary."""
    for tmp_fp in TMP_DIRS:
        shutil.rmtree(tmp_fp)
