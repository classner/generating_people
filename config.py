#!/usr/bin/env python2
"""Configuration values for the project."""
import os.path as path
import click
if __name__ != '__main__':
    import matplotlib
    import scipy.misc as sm
    import numpy as np


############################ EDIT HERE ########################################
SMPL_FP = path.expanduser("~/smpl")
UP_FP = path.expanduser("~/git/up")
CHICTOPIA_DATA_FP = path.expanduser("~/datasets/chictopia10k")
GEN_DATA_FP = path.abspath(path.join(path.dirname(__file__),
                                     'data', 'intermediate'))
EXP_DATA_FP = path.abspath(path.join(path.dirname(__file__),
                                     'generation', 'data'))

if __name__ != '__main__':
    CMAP = matplotlib.colors.ListedColormap(sm.imread(path.join(
        path.dirname(__file__), "cm_22.png"))[0]
                                            .astype(np.float32) / 255.)


###############################################################################
# Infrastructure. Don't edit.                                                 #
###############################################################################

@click.command()
@click.argument('key', type=click.STRING)
def cli(key):
    """Print a config value to STDOUT."""
    if key in globals().keys():
        print globals()[key]
    else:
        raise Exception("Requested configuration value not available! "
                        "Available keys: " +
                        str([kval for kval in globals().keys() if kval.isupper()]) +
                        ".")


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter

