#!/usr/bin/env python2
import imp
import os.path as path
from glob import glob
import logging
import click
import tqdm
import numpy as np
import scipy.misc as sm

from clustertools import visualization as vs
from clustertools.log import LOGFORMAT
import up_tools.model as upm
import up_tools.render_segmented_views as upr
import pymp
config = imp.load_source(
    'config',
    path.abspath(path.join(path.dirname(__file__),
                           "..", "..", "config.py")))


LOGGER = logging.getLogger(__name__)


def process_image(im_fp, dset_part, out_folder):
    bn = path.basename(im_fp)
    dn = path.dirname(im_fp)
    img_idx = int(bn[:bn.find("_")])
    body_fp = path.join(dn, bn + '_body.pkl')
    im = sm.imread(im_fp)
    if not path.exists(body_fp):
        raise Exception("Body fit not found for `%s` (`%s`)!" % (im_fp, body_fp))
    rendering = upr.render_body_impl(body_fp,
                                     resolution=(im.shape[0], im.shape[1]),
                                     quiet=True,
                                     use_light=False)[0]
    annotation = upm.regions_to_classes(rendering, upm.six_region_groups,
                                        warn_id=str(img_idx))
    out_fp = path.join(out_folder, dset_part,
                       "{:0{width}d}_bodysegments.png".format(
        img_idx, width=bn.find("_")))
    sm.imsave(out_fp, annotation)
    out_fp = path.join(out_folder, dset_part,
                       "{:0{width}d}_bodysegments_vis.png".format(
        img_idx, width=bn.find("_")))
    sm.imsave(out_fp, vs.apply_colormap(annotation, vmin=0, vmax=6,
                                        cmap=config.CMAP)[:, :, 0:3])


@click.command()
@click.option("--dset_folder",
              type=click.Path(exists=True, readable=True, file_okay=False),
              default=path.join('..', 'data', 'pose', 'input'),
              help="The dataset folder to process.")
@click.option("--out_folder",
              type=click.Path(exists=True, writable=True, file_okay=False),
              default=path.join('..', 'data', 'pose', 'extracted'),
              help="The output folder.")
def cli(dset_folder=path.join('..', 'data', 'pose', 'input'),
        out_folder=path.join('..', 'data', 'pose', 'extracted')):
    LOGGER.info("Processing...")
    for part in ['train', 'val', 'test']:
        im_fps = sorted(glob(path.join(dset_folder, part, '*_image*.png')))
        # Filter.
        im_fps = [im_fp for im_fp in im_fps
                  if ('pose' not in path.basename(im_fp) and
                      'body' not in path.basename(im_fp))]
        with pymp.Parallel(12, if_=False) as p:
            for im_fp in p.iterate(tqdm.tqdm(im_fps)):
                process_image(im_fp, part, out_folder)
    LOGGER.info("Done.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    cli()
