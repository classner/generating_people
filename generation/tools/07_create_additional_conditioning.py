#!/usr/bin/env python2
import imp
import os.path as path
from glob import glob
import logging
import click
import tqdm
import scipy.misc as sm
import numpy as np

from clustertools.log import LOGFORMAT
from geometric_median import geometric_median
import pymp
config = imp.load_source(
    'config',
    path.abspath(path.join(path.dirname(__file__),
                           "..", "..", "config.py")))


LOGGER = logging.getLogger(__name__)


def process_image(im_fp, dset_part):
    bn = path.basename(im_fp)
    dn = path.dirname(im_fp)
    img_idx = int(bn[:bn.find("_")])
    im = sm.imread(im_fp)
    segmentation = sm.imread(path.join(path.dirname(im_fp),
                                       "{0:0{width}d}_labels:png.png".format(
                                           img_idx, width=bn.find("_"))))
    if segmentation.ndim == 3:
        segmentation = segmentation[:, :, 0]
    region_ids = sorted(np.unique(segmentation))
    img_colors = np.zeros_like(im)
    for region_id in region_ids:
        if region_id == 0:
            # Background is already 0-labeled.
            continue
        region_id_set = [region_id]
        if region_id == 9:  # left shoe.
            region_id_set.append(10)
        elif region_id == 10:
            region_id_set.append(9)
        elif region_id in [11, 14, 15, 19]:  # Skin.
            region_id_set = [11, 14, 15, 19]
        elif region_id in [12, 13]:  # legs.
            region_id_set = [12, 13]
        elif region_id in [20, 21]:
            region_id_set = [20, 21]  # eyes.
        region_colors = np.vstack(
            [im[segmentation == idx] for idx in region_id_set])
        med_color = geometric_median(region_colors)
        img_colors[segmentation == region_id] = med_color
    sm.imsave(path.join(dn, "{0:0{width}d}_segcolors.png".format(
        img_idx, width=bn.find("_"))), img_colors)


@click.command()
def cli():
    LOGGER.info("Processing...")
    for part in ['train', 'val', 'test']:
        im_fps = sorted(glob(path.join('..', 'data', 'pose', 'extracted', part,
                                       '*_image*.png')))
        # Filter.
        im_fps = [im_fp for im_fp in im_fps
                  if ('pose' not in path.basename(im_fp) and
                      'body' not in path.basename(im_fp))]
        with pymp.Parallel(12, if_=True) as p:
            for im_fp in p.iterate(tqdm.tqdm(im_fps)):
                process_image(im_fp, part)
    LOGGER.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    cli()
