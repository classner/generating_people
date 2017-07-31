#!/usr/bin/env python2
"""Assemble the fashion dataset."""
import os
import os.path as path
import sys
import scipy.misc as sm
from glob import glob
import numpy as np
import click
import logging
import cv2
import dlib
from clustertools.log import LOGFORMAT
from clustertools.visualization import apply_colormap
import clustertools.db.tools as cdbt

sys.path.insert(0, path.join('..', '..'))
from config import CHICTOPIA_DATA_FP, CMAP  # noqa: E402


LOGGER = logging.getLogger(__name__)


def getface(sketch, BORDER=0):
    # Face has value 11.
    minind_x = 255
    maxind_x = 0
    for row_idx in range(sketch.shape[0]):
        if 11 in sketch[row_idx, :]:
            minind_x = min([minind_x, np.argmax(sketch[row_idx, :] == 11)])
            maxind_x = max(maxind_x, sketch.shape[1] - 1 - np.argmax(sketch[row_idx, ::-1] == 11))
    minind_y = 255
    maxind_y = 0
    for col_idx in range(sketch.shape[1]):
        if 11 in sketch[:, col_idx]:
            minind_y = min([minind_y, np.argmax(sketch[:, col_idx] == 11)])
            maxind_y = max([maxind_y, sketch.shape[0] - 1 - np.argmax(sketch[::-1, col_idx] == 11)])
    LOGGER.debug("Without border: min_y=%d, max_y=%d, min_x=%d, max_x=%d",
                minind_y, maxind_y, minind_x, maxind_x)
    minind_x = max([0, minind_x - BORDER])
    maxind_x = min([sketch.shape[1] - 1, maxind_x + BORDER])
    minind_y = max([0, minind_y - BORDER])
    maxind_y = min([sketch.shape[0] - 1, maxind_y + BORDER])
    LOGGER.debug("With border: min_y=%d, max_y=%d, min_x=%d, max_x=%d",
                minind_y, maxind_y, minind_x, maxind_x)
    # Make the area rectangular.
    if maxind_y - minind_y != maxind_x - minind_x:
        if maxind_y - minind_y > maxind_x - minind_x:
            # Height is longer, enlarge width.
            diff = maxind_y - minind_y - maxind_x + minind_x
            if minind_x < int(np.floor(diff / 2.)):
                maxind_x = maxind_x + (diff - minind_x)
                minind_x = 0
            elif sketch.shape[1] - maxind_x - int(np.ceil(diff / 2.)) < 0:
                minind_x = minind_x - (diff - sketch.shape[1] + maxind_x)
                maxind_x = sketch.shape[1] - 1
            else:
                minind_x = minind_x - int(np.floor(diff / 2.))
                maxind_x = maxind_x + int(np.ceil(diff / 2.))
        else:
            # Width is longer, enlarge height.
            diff = - (maxind_y - minind_y - maxind_x + minind_x)
            if minind_y < int(np.floor(diff / 2.)):
                maxind_y = maxind_y + (diff - minind_y)
                minind_y = 0
            elif sketch.shape[0] - maxind_y - int(np.ceil(diff / 2.)) < 0:
                minind_y = minind_y - (diff - sketch.shape[0] + maxind_y)
                maxind_y = sketch.shape[0] - 1
            else:
                minind_y = minind_y - int(np.floor(diff / 2.))
                maxind_y = maxind_y + int(np.ceil(diff / 2.))
    if maxind_y - minind_y <= 0 or maxind_x - minind_x <= 0:
        LOGGER.warn("No face detected in image!")
        return None
    return minind_y, maxind_y, minind_x, maxind_x


fdetector = dlib.get_frontal_face_detector()
spredictor = dlib.shape_predictor(path.join(path.dirname(__file__),
                                            'shape_predictor_68_face_landmarks.dat'))
if not path.exists(spredictor):
    LOGGER.critical("Please download and unpack the face shape model from "
                    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
                    "to `%s`.", spredictor)
    sys.exit(1)

def prepare(im_idx):
    # Load.
    image = sm.imread(path.join(CHICTOPIA_DATA_FP, 'JPEGImages',
                                '%s.jpg' % (im_idx)))
    if image.ndim != 3:
        return []
    resize_factor = 513. / max(image.shape[:2])
    im_resized = sm.imresize(image, resize_factor)
    annotation = sm.imread(path.join(CHICTOPIA_DATA_FP, 'SegmentationClassAug',
                                     '%s.png' % (im_idx)))
    resannot = sm.imresize(annotation, resize_factor, interp='nearest')
    # Holes.
    kernel = np.ones((7, 7), np.uint8)
    closed_annot = cv2.morphologyEx(resannot, cv2.MORPH_CLOSE, kernel)
    grad = cv2.morphologyEx(resannot, cv2.MORPH_BLACKHAT, kernel)
    to_fill = np.logical_and(resannot == 0, grad > 0)
    resannot[to_fill] = closed_annot[to_fill]
    # Face detection.
    FDEBUG = False
    # For debugging.
    if FDEBUG:
        win = dlib.image_window()
        win.clear_overlay()
        win.set_image(im_resized)
    face_box = getface(resannot)
    max_IOU = 0.
    if face_box is not None:
        most_likely_det = None
        dets, _, _ = fdetector.run(im_resized, 1, -1)
        for k, d in enumerate(dets):
            # Calculate IOU with ~ground truth.
            ar_pred = (d.right() - d.left()) * (d.bottom() - d.top())
            face_points = (resannot == 11)
            face_pos = np.where(face_points)
            inters_x = np.logical_and(face_pos[1] >= d.left(),
                                      face_pos[1] < d.right())
            inters_y = np.logical_and(face_pos[0] >= d.top(),
                                      face_pos[0] < d.bottom())
            inters_p = np.sum(np.logical_and(inters_x, inters_y))
            outs_p = np.sum(face_points) - inters_p
            IOU = float(inters_p) / (outs_p + ar_pred)
            if IOU > 1.:
                import ipdb; ipdb.set_trace()
            if IOU > 0.3 and IOU > max_IOU:
                most_likely_det = d
                max_IOU = IOU
        if most_likely_det is not None:
            shape = spredictor(im_resized, most_likely_det)
            # Save hat, hair and sunglasses (likely to cover eyes or nose).
            hat = (resannot == 1)
            hair = (resannot == 2)
            sungl = (resannot == 3)
            # Add annotations:
            an_lm = {
                (48, 67): 18,  # lips
                (27, 35): 19,  # nose
                (36, 41): 20,  # leye
                (42, 47): 21,  # reye
            }
            for rng, ann_id in an_lm.items():
                poly = np.empty((2, rng[1] - rng[0]),
                                dtype=np.int64)
                for point_idx, point_id in enumerate(range(*rng)):
                    poly[0, point_idx] = shape.part(point_id).x
                    poly[1, point_idx] = shape.part(point_id).y
                # Draw additional annotations.
                poly = poly.T.copy()
                cv2.fillPoly(
                    resannot,
                    [poly],
                    (ann_id,))
            # Write back hat, hair and sungl.
            resannot[hat] = 1
            resannot[hair] = 2
            resannot[sungl] = 3
            if FDEBUG:
                win.add_overlay(shape)
                win.add_overlay(most_likely_det)
                dlib.hit_enter_to_continue()
        else:
            # No reliable face found.
            return []
    return [(
        '%s.jpg' % (im_idx),
        im_resized,
        np.dstack([resannot] * 3),
        apply_colormap(resannot, vmin=0, vmax=21, cmap=CMAP)[:, :, :3]
        )]


@click.command()
def cli():
    """Assemble a unified fashion dataset."""
    np.random.seed(1)
    out_fp = path.join(path.dirname(__file__), '..', '..', 'data')
    LOGGER.info("Using output directory `%s`.", out_fp)
    if not path.exists(out_fp):
        os.mkdir(out_fp)
    db_fns = glob(path.join(out_fp, '*.tfrecords'))
    for db_fn in db_fns:
        os.unlink(db_fn)
    chic10k_root = CHICTOPIA_DATA_FP
    chic10k_im_fps = sorted(glob(path.join(chic10k_root, 'JPEGImages', '*.jpg')))
    chic10k_ids = [path.basename(im_fp)[:path.basename(im_fp).index('.')]
                   for im_fp in chic10k_im_fps]
    perm = np.random.permutation(chic10k_ids)
    train_ids = perm[:int(len(perm) * 0.8)]
    val_ids = perm[int(len(perm) * 0.8):int(len(perm) * 0.9)]
    test_ids = perm[int(len(perm) * 0.9):]
    creator = cdbt.TFRecordCreator([
        ('original_filename', cdbt.SPECTYPES.text),
        # It is critical here to use lossless compression - the adversary will
        # otherwise pick up JPEG compression cues.
        ('image', cdbt.SPECTYPES.imlossless),
        ('labels', cdbt.SPECTYPES.imlossless),
        ('label_vis', cdbt.SPECTYPES.imlossless),
    ],
                                examples_per_file=300)
    for pname, pids in zip(['train', 'val', 'test'],
                           [train_ids, val_ids, test_ids]):
        creator.open(path.join(out_fp, pname))
        creator.add_to_dset(prepare,
                            pids,
                            num_threads=16,
                            progress=True)
        creator.close()
    LOGGER.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    cli()
