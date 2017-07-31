import imp
import logging
import os.path as path

LOGGER = logging.getLogger(__name__)


def get_config():
    CONF_FP = path.join(path.dirname(__file__), "options.py")
    LOGGER.info("Loading experiment configuration from `%s`...", CONF_FP)
    options = imp.load_source('_options',
                              path.abspath(path.join(path.dirname(__file__),
                                                     'options.py')))
    # with open(CONF_FP, 'r') as inf:
    #     config = json.loads(inf.read())
    LOGGER.info("Done.")
    return options.config


def adjust_config(config, mode):
    # Don't misuse this!
    # Results are better if batchnorm is always used in 'training' mode and
    # normalizes the observed distribution. That's why it's important to leave
    # the batchsize unchanged.
    #if mode not in ['train', 'trainval']:
    #    config['batch_size'] = 1
    return config
