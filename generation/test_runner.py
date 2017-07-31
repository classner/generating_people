#!/usr/bin/env python2
import os
import os.path as path
import subprocess
import time
from glob import glob
import logging
import click
from tensorflow.tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator)
from clustertools.log import LOGFORMAT


LOGGER = logging.getLogger(__name__)


def get_unevaluated_checkpoint(result_fp, events_acc):
    LOGGER.info("Scanning for unevaluated checkpoints...")
    # Get all checkpoints.
    checkpoints = sorted(
        [int(path.basename(val)[6:-5])
         for val in glob(path.join(result_fp, 'model-*.meta'))])
    LOGGER.debug("Available checkpoints: %s.", checkpoints)
    events_acc.Reload()
    available_tags = events_acc.Tags()['scalars']
    if available_tags:
        test_tag = events_acc.Tags()['scalars'][0]
        LOGGER.debug("Using tag `%s` for check.", test_tag)
        recorded_steps = sorted([ev.step
                                 for ev in events_acc.Scalars(test_tag)])
        LOGGER.debug("Recorded steps: %s.", recorded_steps)
    else:
        LOGGER.debug("No recorded steps found.")
        recorded_steps = []
    # Merge.
    for cp_idx in checkpoints:
        if cp_idx not in recorded_steps:
            LOGGER.info("Detected unevaluated checkpoint: %d.", cp_idx)
            return cp_idx
    LOGGER.info("Scan complete. No new checkpoints found.")
    return None


@click.command()
@click.argument("state_fp", type=click.Path(exists=True, readable=True))
@click.argument("monitor_set", type=click.Choice(["val", "test"]))
@click.option("--check_interval", type=click.INT, default=60,
              help="Interval in seconds between checks for new checkpoints.")
def cli(state_fp, monitor_set, check_interval=60):
    """Start a process providing validation/test results for a training."""
    LOGGER.info("Starting monitoring for result path `%s` and set `%s`.",
                state_fp, monitor_set)
    if not path.exists(path.join(state_fp, monitor_set)):
        os.makedirs(path.join(state_fp, monitor_set))
    events_acc = EventAccumulator(path.join(state_fp, monitor_set))
    while True:
        cp_idx = get_unevaluated_checkpoint(state_fp, events_acc)
        if cp_idx is not None:
            LOGGER.info("Running evaluation for checkpoint %d...", cp_idx)
            subprocess.check_call(["./run.py",
                                   monitor_set,
                                   path.join("experiments", "config",
                                             path.basename(state_fp)),
                                   "--checkpoint",
                                   path.join(state_fp, 'model-%d.meta' % (
                                       cp_idx)),
                                   "--no_output"])
        else:
            time.sleep(check_interval)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    cli()  # pylint: disable=no-value-for-parameter
