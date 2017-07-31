#!/usr/bin/env
"""Create the dataset."""
# pylint: disable=wrong-import-position, invalid-name
import os
import os.path as path
import sys
import logging

import numpy as np
import scipy
import scipy.misc
import click

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))
from config import CMAP  # pylint: disable=no-name-in-module
from clustertools.log import LOGFORMAT
from clustertools.visualization import apply_colormap
import clustertools.db.tools as cdb


LOGGER = logging.getLogger(__name__)


# Build lr-swapper.
chick10k_mapping = {
    0: 0,     # Background.
    1: 1,     # Hat.
    2: 2,     # Hair.
    3: 3,     # Sunglasses.
    4: 4,     # Top.
    5: 5,     # Skirt.
    6: 6,     # ??
    7: 7,     # Dress.
    8: 8,     # Belt.
    9: 10,    # Left shoe.
    10: 9,    # Right shoe.
    11: 11,   # Skin, face.
    12: 13,   # Skin, left leg.
    13: 12,   # Skin, right leg.
    14: 15,   # Skin, left arm.
    15: 14,   # Skin, right arm,
    16: 16,   # Bag.
    17: 17,   # ??.
    18: 18,   # lips.
    19: 19,   # nose.
    20: 21,   # leye.
    21: 20,   # reye.
}

def lrswap_regions(annotations):
    """Swap left and right annotations."""
    assert annotations.ndim == 2
    swapspec = chick10k_mapping
    swapped = np.empty_like(annotations)
    for py in range(annotations.shape[0]):
        for px in range(annotations.shape[1]):
            swapped[py, px] = swapspec[annotations[py, px]]
    return swapped

def pad_height(image, crop):
    """Pad the height to given crop size."""
    if image.ndim == 2:
        image = image[:, :, None]
    image = np.vstack((np.zeros((int(np.floor(max([0, crop - image.shape[0]]) / 2.)),
                                 image.shape[1],
                                 image.shape[2]), dtype=image.dtype),
                       image,
                       np.zeros((int(np.ceil(max([0, crop - image.shape[0]]) / 2.)),
                                 image.shape[1],
                                 image.shape[2]), dtype=image.dtype)))
    if image.shape[2] == 1:  # pylint: disable=no-else-return
        return image[:, :, 0]
    else:
        return image

def pad_width(image, crop):
    """Pad the width to given crop size."""
    if image.ndim == 2:
        image = image[:, :, None]
    image = np.hstack((np.zeros((image.shape[0],
                                 int(np.floor(max([0, crop - image.shape[1]]) / 2.)),
                                 image.shape[2]), dtype=image.dtype),
                       image,
                       np.zeros((image.shape[0],
                                 int(np.ceil(max([0, crop - image.shape[1]]) / 2.)),
                                 image.shape[2]), dtype=image.dtype)))
    if image.shape[2] == 1:
        return image[:, :, 0]
    else:
        return image

crop = 0
def convert(inputs):
    imname = inputs['original_filename']
    image = inputs['image']
    labels = inputs['labels']
    label_vis = inputs['label_vis']
    results = []
    segmentation = labels[:, :, 0]
    norm_factor = float(crop) / max(image.shape[:2])
    image = scipy.misc.imresize(image, norm_factor, interp='bilinear')
    segmentation = scipy.misc.imresize(segmentation, norm_factor, interp='nearest')
    if image.shape[0] < crop:
        # Pad height.
        image = pad_height(image, crop)
        segmentation = pad_height(segmentation, crop)
    if image.shape[1] < crop:
        image = pad_width(image, crop)
        segmentation = pad_width(segmentation, crop)
    labels = np.dstack([segmentation] * 3)
    label_vis = apply_colormap(segmentation, vmax=21, vmin=0, cmap=CMAP)[:, :, :3]
    results.append([imname, image * (labels != 0), labels, label_vis])
    # Swapped version.
    imname = path.splitext(imname)[0] + '_swapped' + path.splitext(imname)[1]
    image = image[:, ::-1]
    segmentation = segmentation[:, ::-1]
    segmentation = lrswap_regions(segmentation)
    labels = np.dstack([segmentation] * 3)
    label_vis = apply_colormap(segmentation, vmax=21, vmin=0, cmap=CMAP)[:, :, :3]
    results.append([imname, image * (labels != 0), labels, label_vis])
    return results


@click.command()
@click.argument("suffix", type=click.STRING)
@click.option("--crop_size", type=click.INT, default=286,
              help="Crop size for the images.")
def cli(suffix, crop_size):  # pylint: disable=too-many-locals, too-many-arguments
    """Create clothing segmentation to fashion image dataset."""
    global crop
    np.random.seed(1)
    crop = crop_size
    LOGGER.info("Creating generation dataset with target "
                "image size %f and suffix `%s`.",
                crop, suffix)
    assert ' ' not in suffix
    dset_fp = path.join(path.dirname(__file__), '..', 'data', 'people', suffix)
    if path.exists(dset_fp):
        if not click.confirm("Dataset folder exists: `%s`! Continue?" % (
                dset_fp)):
            return
    else:
        os.makedirs(dset_fp)
    converter = cdb.TFConverter([
        ('original_filename', cdb.SPECTYPES.text),
        ('image', cdb.SPECTYPES.imlossless),
        ('labels', cdb.SPECTYPES.imlossless),
        ('label_vis', cdb.SPECTYPES.imlossless),
    ])
    LOGGER.info("Processing...")
    for pname in ['train', 'val', 'test']:
        converter.open(
            path.join(path.dirname('__file__'), '..', '..', 'data', pname),
            path.join(dset_fp, pname))
        converter.convert_dset(convert,
                               num_threads=16,
                               progress=True)
        converter.close()
    LOGGER.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    cli()  # pylint: disable=no-value-for-parameter
