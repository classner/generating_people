"""Tensorflow tools."""
import logging
import numpy as np
import tensorflow as tf


LOGGER = logging.getLogger(__name__)


def get_val_or_initializer(load_tpl, initializer, varname,
                           sloppy=False, dtype=tf.float32):
    """Get the variable value from an existing graph or the initializer."""
    orig_sess, orig_graph = load_tpl
    full_name = tf.get_variable_scope().name + '/' + varname + ':0'
    if orig_graph is not None:
        LOGGER.debug("Restoring `%s`." % (full_name))
        with orig_graph.as_default():
            orig_var = [var for var in tf.global_variables()
                        if var.name == full_name]
            if len(orig_var) != 1:
                if not sloppy:
                    LOGGER.critical("Missing value for variable `%s`!" % (
                        full_name))
                    import ipdb; ipdb.set_trace()  # noqa: E702
                else:
                    orig_var = None
                    init_value = None
            else:
                orig_var = orig_var[0]
                init_value = orig_var.eval(session=orig_sess)
        if orig_var is not None:
            init_value = tf.convert_to_tensor(init_value, dtype=dtype)
    else:
        init_value = None
    if init_value is not None:
        return lambda *args, **kwargs: tf.cast(
            init_value, kwargs.get('dtype', tf.float32))
    else:
        return initializer


def get_or_load_variable(load_tpl, *args, **kwargs):
    """Get a variable value from an existing graph or create one."""
    orig_sess, orig_graph = load_tpl
    full_name = tf.get_variable_scope().name + '/' + args[0] + ':0'
    if orig_graph is not None:
        LOGGER.debug("Restoring `%s`." % (full_name))
        with orig_graph.as_default():
            orig_var = [var for var in tf.global_variables()
                        if var.name == full_name]
            if len(orig_var) != 1:
                if 'sloppy' not in kwargs.keys() or not kwargs['sloppy']:
                    LOGGER.critical("Missing value for variable `%s`!" % (
                        full_name))
                    import ipdb; ipdb.set_trace()  # noqa: E702
                else:
                    orig_var = None
                    init_value = None
            else:
                orig_var = orig_var[0]
                init_value = orig_var.eval(session=orig_sess)
        if orig_var is not None:
            new_dt = tf.float32
            if 'dtype' in kwargs.keys():
                new_dt = kwargs['dtype']
            init_value = tf.convert_to_tensor(init_value, dtype=new_dt)
    else:
        init_value = None
    if init_value is not None:
        trainable = True
        if "trainable" in kwargs.keys():
            trainable = kwargs["trainable"]
        return tf.get_variable(args[0],
                               initializer=init_value,
                               trainable=trainable)
    else:
        if 'sloppy' in kwargs.keys():
            del kwargs['sloppy']
        return tf.get_variable(*args, **kwargs)


def one_hot(inputs, num_classes):
    """
    One hot encoding with fixed number of classes.

    # noqa: E501
    See also: http://stackoverflow.com/questions/35226198/is-this-one-hot-encoding-in-tensorflow-fast-or-flawed-for-any-reason
    """
    inshape = inputs.get_shape().as_list()
    assert len(inshape) <= 2
    for shcomp in inshape:
        assert shcomp is not None
    input_vec = tf.reshape(inputs, (-1, 1))
    table = tf.constant(np.identity(num_classes, dtype=np.float32))
    embeddings = tf.nn.embedding_lookup(table, tf.cast(input_vec, tf.int32))
    outshape = inshape + [num_classes, ]
    output = tf.reshape(embeddings, outshape)
    return output
