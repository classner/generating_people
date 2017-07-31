"""Defining the model or model family."""
import collections
import logging

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tfl

from gp_tools.tf import get_or_load_variable, get_val_or_initializer

# flake8: noqa: E501
LOGGER = logging.getLogger(__name__)
EPS = 1e-12
#: Defines the field of a model. Required:
#: gobal_step, train
Model = collections.namedtuple(
    "Model",
    "global_step, train, "  # Required
    "z_mean, z_log_sigma_sq, z, y, outputs, predict_real, predict_fake, "
    "discrim_loss, gen_loss_GAN, gen_loss_recon, gen_loss_latent, gen_accuracy")


# Graph components. ###########################################################
def conv(batch_input, out_channels, stride, orig_graph):
    """Convolution without bias as in Isola's paper."""
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = get_or_load_variable(
            orig_graph,
            "filter",
            [4, 4, in_channels, out_channels],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels],
        # [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input,
                              [[0, 0], [1, 1], [1, 1], [0, 0]],
                              mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1],
                            padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input, orig_graph, is_training):
    return tfl.batch_norm(
        input,
        decay=0.9,
        scale=True,
        epsilon=1E-5,
        activation_fn=None,
        param_initializers={
            'beta': get_val_or_initializer(orig_graph,
                                           tf.constant_initializer(0.),
                                           'BatchNorm/beta'),
            'gamma': get_val_or_initializer(orig_graph,
                                            tf.random_normal_initializer(1.0,
                                                                         0.02),
                                            'BatchNorm/gamma'),
            'moving_mean': get_val_or_initializer(orig_graph,
                                                  tf.constant_initializer(0.),
                                                  'BatchNorm/moving_mean'),
            'moving_variance': get_val_or_initializer(orig_graph,
                                                      tf.ones_initializer(),
                                                      'BatchNorm/moving_variance')
        },
        is_training=is_training,
        fused=True,  # new implementation with a fused kernel => speedup.
    )


def deconv(batch_input, out_channels, orig_graph):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = \
            batch_input.get_shape().as_list()
        filter = get_or_load_variable(
            orig_graph,
            "filter", [4, 4, out_channels, in_channels], dtype=tf.float32,
            initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels],
        # [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(
            batch_input, filter,
            [batch, in_height * 2, in_width * 2, out_channels],
            [1, 2, 2, 1],
            padding="SAME")
        return conv


# Processing part. ############################################################
def create_generator(generator_inputs,
                     generator_outputs_channels,
                     mode,
                     config,
                     full_graph,
                     conditioning,
                     rwgrid,
                     is_training):
    if generator_inputs is None:
        assert config["model_version"] in ['vae', 'cvae']
        LOGGER.info("Omitting encoder network - sampling...")
    if (config.get("pix2pix_zbatchnorm", False) and
            config["model_version"] == 'portray'):
        assert config["batch_size"] > 1
    layers = []
    if config["model_version"] == 'cvae':
        # Build y encoder.
        y_layers = []
        # y_encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("y_encoder_1"):
            if config["cvae_downscale_y"]:
                output = tf.image.resize_images(
                    conditioning,
                    [conditioning.get_shape().as_list()[1] // 2,
                     conditioning.get_shape().as_list()[2] // 2],
                    method=tf.image.ResizeMethod.BILINEAR)
            else:
                output = conv(conditioning, config["ngf"], 2, full_graph)
            y_layers.append(output)
        layer_specs = [
            config["ngf"] * 2,  # y_encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            config["ngf"] * 4,  # y_encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            config["ngf"] * 8,  # y_encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            config["ngf"] * 8,  # y_encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            config["ngf"] * 8,  # y_encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            config["ngf"] * 8,  # y_encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            config["ngf"] * 8,  # y_encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]
        for ch_idx, out_channels in enumerate(layer_specs):
            with tf.variable_scope("y_encoder_%d" % (len(y_layers) + 1)):
                if config["cvae_downscale_y"]:
                    output = tf.image.resize_images(
                        y_layers[-1],
                        [y_layers[-1].get_shape().as_list()[1] // 2,
                         y_layers[-1].get_shape().as_list()[2] // 2],
                        method=tf.image.ResizeMethod.BILINEAR)
                else:
                    rectified = lrelu(y_layers[-1], 0.2)
                    # [batch, in_height, in_width, in_channels] =>
                    # [batch, in_height/2, in_width/2, out_channels]
                    convolved = conv(rectified, out_channels, 2, full_graph)
                    if ch_idx != len(layer_specs) - 1:
                        output = batchnorm(convolved, full_graph, is_training)
                    else:
                        output = convolved
                y_layers.append(output)
        y = y_layers[-1]
    else:
        y = tf.zeros((1,), dtype=tf.float32)

    if generator_inputs is not None:
        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = conv(generator_inputs, config["ngf"], 2, full_graph)
            layers.append(output)

        layer_specs = [
            config["ngf"] * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            config["ngf"] * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            config["ngf"] * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            config["ngf"] * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            config["ngf"] * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            config["ngf"] * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ]
        if (config['model_version'] not in ['vae', 'cvae'] or
                config['cvae_nosampling']):
            layer_specs.append(config["ngf"] * 8)
            # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]

        for ch_idx, out_channels in enumerate(layer_specs):
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                if config['model_version'] == 'cvae':
                    if not config['cvae_noconn'] or ch_idx == 0:
                        input = tf.concat([layers[-1],
                                           y_layers[len(layers) - 1]],
                                          axis=3)
                    else:
                        input = layers[-1]
                else:
                    input = layers[-1]
                rectified = lrelu(input, 0.2)
                # [batch, in_height, in_width, in_channels] =>
                # [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(rectified, out_channels, 2, full_graph)
                if (config['model_version'] in ['vae', 'cvae'] or
                    ch_idx != len(layer_specs) - 1 or
                    (ch_idx == len(layer_specs) - 1 and
                     config.get("pix2pix_zbatchnorm", False) and
                     config["model_version"] == 'portray') or
                        config['cvae_noconn']):
                    output = batchnorm(convolved, full_graph, is_training)
                else:
                    output = convolved
                layers.append(output)

        if (config['model_version'] in ["vae", 'cvae']
                and not config['cvae_nosampling']):
            # VAE infrastructure.
            with tf.variable_scope("vae_encoder"):
                with tf.variable_scope("z_mean"):
                    if config['model_version'] == 'cvae':
                        input = tf.concat([layers[-1],
                                           y_layers[len(layers) - 1]],
                                          axis=3)
                    else:
                        input = layers[-1]
                    weights = get_or_load_variable(
                        full_graph,
                        "weights", [np.prod(input.get_shape().as_list()[1:]),
                                    config["nz"]],
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(0, 0.02))
                    biases = get_or_load_variable(
                        full_graph, "biases", [config["nz"], ],
                        dtype=tf.float32)
                    z_mean = tf.add(
                        tf.matmul(tf.reshape(input,
                                             (config["batch_size"], -1)),
                                  weights),
                        biases)
                with tf.variable_scope("z_log_sigma_sq"):
                    weights = get_or_load_variable(
                        full_graph, "weights", [
                            np.prod(input.get_shape().as_list()[1:]),
                            config["nz"]],
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(0, 0.02))
                    biases = get_or_load_variable(full_graph,
                                                  "biases", [config["nz"], ],
                                                  dtype=tf.float32)
                    if config['model_version'] == 'cvae':
                        input = tf.concat([layers[-1],
                                           y_layers[len(layers) - 1]],
                                          axis=3)
                    else:
                        input = layers[-1]
                    # Save the 0.5, should be learned.
                    z_log_sigma_sq = tf.add(
                        tf.matmul(
                            tf.reshape(input, (config["batch_size"], -1)),
                            weights),
                        biases)

    if (config['model_version'] in ['vae', 'cvae']
            and not config["cvae_nosampling"]):
        with tf.variable_scope("vae_latent_representation"):
            if rwgrid is None:
                eps = tf.random_normal((config["batch_size"],
                                        config["nz"]), 0, 1,
                                       dtype=tf.float32)
                if generator_inputs is not None:
                    # z = mu + sigma*epsilon
                    z = tf.add(z_mean,
                               tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
                else:
                    z = eps
                    z_mean = tf.constant(np.zeros((config["nz"],),
                                                  dtype=np.float32),
                                         dtype=tf.float32)
                    z_log_sigma_sq = tf.constant(np.ones((config["nz"],),
                                                         dtype=np.float32),
                                                 dtype=tf.float32)
                z = z[:, None, None, :]
            else:
                z = tf.py_func(rwgrid.sample, [], tf.float32)
                z_mean = tf.constant(np.zeros((config["nz"],),
                                              dtype=np.float32),
                                     dtype=tf.float32)
                z_log_sigma_sq = tf.constant(np.ones((config["nz"],),
                                                     dtype=np.float32),
                                             dtype=tf.float32)
            z.set_shape([config["batch_size"], 1, 1, config["nz"]])
            layers.append(z)
    else:
        if generator_inputs is None:
            raise Exception(
                "Sampling required for this model configuration!"
                "Model must be VAE or CVAE and cvae_nosampling may not be "
                "set!")
        z = layers[-1]
        z_mean = tf.constant(np.zeros((z.get_shape().as_list()[3],),
                                      dtype=np.float32),
                             dtype=tf.float32)
        z_log_sigma_sq = tf.constant(np.ones((z.get_shape().as_list()[3],),
                                             dtype=np.float32),
                                     dtype=tf.float32)

    layer_specs = [
        (config["ngf"] * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (config["ngf"] * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (config["ngf"] * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (config["ngf"] * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (config["ngf"] * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (config["ngf"] * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (config["ngf"], 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]
    if config["model_version"] in ['vae', 'cvae']:
        for spec_idx in [0, 1, 2]:
            layer_specs[spec_idx] = (config["ngf"] * 8, 0.)

    if generator_inputs is None:
        assert config["model_version"] in ['vae', 'cvae']
        num_encoder_layers = 8
    else:
        num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if config["model_version"] == 'cvae' and decoder_layer == 0:
                layers.append(tf.concat([y, z], axis=3))
            if config["model_version"] == 'portray' and decoder_layer != 0:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
            else:
                input = layers[-1]
            if (decoder_layer == 0 and
                config["model_version"] in ['vae', 'cvae']):
                # Don't use a ReLU on the latent encoding.
                rectified = input
            else:
                rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] =>
            # [batch, in_height*2, in_width*2, out_channels]
            rs = rectified.get_shape().as_list()
            if rs[0] is None:
                rectified.set_shape([config["batch_size"],
                                     rs[1], rs[2], rs[3]])
            output = deconv(rectified, out_channels, full_graph)
            output = batchnorm(output, full_graph, is_training)
            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)
            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] =>
    # [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = layers[-1]
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels, full_graph)
        unnormalized_output = output
        if (config["input_as_class"] and
                config["model_version"] in ['vae', 'cvae']):
            output = tf.sigmoid(output) - 0.5
        else:
            output = tf.tanh(output)
        layers.append(output)

    return layers[-1], z_mean, z_log_sigma_sq, z, unnormalized_output, y


def create_discriminator(discrim_inputs, discrim_targets, config, full_graph,
                         is_training):
    n_layers = 3
    if discrim_inputs is None or discrim_targets is None:
        LOGGER.info("Omitting discriminator, no inputs or targets provided.")
        return tf.constant(np.zeros((config["batch_size"], 30, 30, 1),
                                    dtype=np.float32),
                           dtype=tf.float32)
    layers = []

    # 2x [batch, height, width, in_channels] =>
    # [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(input, config["ndf"], 2, full_graph)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = config["ndf"] * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer has stride 1
            convolved = conv(layers[-1], out_channels, stride, full_graph)
            normalized = batchnorm(convolved, full_graph, is_training)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, 1, 1, full_graph)
        unnormalized_output = convolved
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1], unnormalized_output


# Interface function. #########################################################
def create_model(mode, examples, config, load_info):
    LOGGER.info("Building model...")
    if (config["model_version"] == 'portray' and
            config.get("portray_additional_conditioning", False)):
        inputs = tf.concat([examples.inputs, examples.conditioning], axis=3)
    else:
        inputs = examples.inputs
    conditioning = examples.conditioning
    targets = examples.targets
    if config["model_version"] == 'cvae':
        discrim_ref = conditioning
    else:
        discrim_ref = inputs

    with tf.variable_scope("generator"):
        if mode == 'sample' and config["model_version"] in ['cvae', 'vae']:
            if config["input_as_class"]:
                out_channels = 22
            else:
                out_channels = 3
        else:
            out_channels = int(targets.get_shape()[-1])
        # We leave batchnorm always in training mode, because it gives slightly
        # better performance to normalize the observed distribution.
        outputs, z_mean, z_log_sigma_sq, z, unnormalized_outputs, y =\
            create_generator(inputs, out_channels, mode,
                             config, load_info, conditioning, None,
                             True)
                             #mode in ['train', 'trainval'])

    # Create two copies of discriminator, one for real pairs and one for fake
    # pairs. Both share the same underlying variables.
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            if (config["gan_weight"] != 0. and
                    mode not in ['sample', 'transform']):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real, predict_real_unnorm = create_discriminator(
                    discrim_ref, targets, config, load_info, True)
                    #mode in ['train', 'trainval'])
            else:
                predict_real = tf.constant(np.zeros((config["batch_size"],
                                                     30, 30, 1),
                                                    dtype=np.float32))
    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            if (config["gan_weight"] != 0. and
                    mode not in ['sample', 'transform']):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake, predict_fake_unnorm = create_discriminator(
                    discrim_ref, outputs, config, load_info, True)
                    #mode in ['train', 'trainval'])
            else:
                predict_fake = tf.constant(np.ones((config["batch_size"],
                                                    30, 30, 1),
                                                   dtype=np.float32))
    # Loss. ###################################
    if config["model_version"] in ['vae', 'cvae']:
        reduction_op = tf.reduce_sum
    else:
        reduction_op = tf.reduce_mean
    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        if config["gan_weight"] != 0. and mode not in ['sample', 'transform']:
            discrim_loss = tf.reduce_mean(
                reduction_op(-(tf.log(predict_real + EPS) +
                               tf.log(1 - predict_fake + EPS)),
                             axis=[1, 2, 3]))
        else:
            discrim_loss = tf.constant(0, tf.float32)

    with tf.name_scope("generator_loss"):
        if targets is not None:
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(
                reduction_op(-tf.log(predict_fake + EPS), axis=[1, 2, 3]))
            if (config["input_as_class"] and
                    config["model_version"] in ['vae', 'cvae']):
                labels = targets + .5
                if config["class_weights"] is not None:
                    LOGGER.info("Using class weights (unmentioned classes "
                                "have weight one): %s.",
                                config["class_weights"])
                    # Determine loss weight matrix.
                    ones = tf.constant(
                        np.ones((labels.get_shape().as_list()[:3]),
                                dtype=np.float32))
                    cwm = tf.identity(ones)
                    for cw_tuple in config["class_weights"].items():
                        cwm = tf.where(
                            tf.equal(labels[:, :, :, cw_tuple[0]], 1.),
                            ones * cw_tuple[1],  # if condition is True
                            cwm  # if condition is False
                        )
                        if not config.get("class_weights_only_recall", True):
                            # Assuming the scaling is equal for all classes
                            # (except the factor 1 ones) this works.
                            cwm = tf.where(
                                tf.equal(tf.argmax(unnormalized_outputs,
                                                   axis=3),
                                         cw_tuple[0]),
                                ones * cw_tuple[1],
                                cwm
                            )
                if config["iac_softmax"]:
                    gen_loss_recon = tf.nn.softmax_cross_entropy_with_logits(
                        logits=unnormalized_outputs, labels=labels)
                else:
                    gen_loss_recon = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=unnormalized_outputs, labels=labels)
                    if config["class_weights"] is not None:
                        cwm = tf.stack([cwm] * 22, axis=3)
                if config["class_weights"] is not None:
                    # Apply.
                    gen_loss_recon *= cwm
                gen_loss_recon = tf.reduce_mean(
                    reduction_op(gen_loss_recon, axis=[1, 2]))
            else:
                # L1.
                gen_loss_recon = tf.reduce_mean(
                    reduction_op(tf.abs(targets - outputs),
                                 axis=[1, 2, 3]))
            if (config["model_version"] in ['vae', 'cvae'] and
                config["latent_weight"] != 0. and
                    not config["cvae_nosampling"]):
                gen_loss_latent = tf.reduce_mean(
                    -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
                                         - tf.square(z_mean)
                                         - tf.exp(z_log_sigma_sq), [1, ]))
            else:
                gen_loss_latent = tf.constant(0, tf.float32)
            gen_loss = (gen_loss_GAN * config["gan_weight"] +
                        gen_loss_recon * config["recon_weight"] +
                        gen_loss_latent * config["latent_weight"])
        else:
            gen_loss_GAN = tf.constant(0, tf.float32)
            gen_loss_recon = tf.constant(0, tf.float32)
            gen_loss_latent = tf.constant(0, tf.float32)
            gen_loss = tf.constant(0, tf.float32)

    with tf.variable_scope("global_step"):
        global_step = get_or_load_variable(load_info,
                                           "global_step",
                                           (1,),
                                           dtype=tf.int64,
                                           initializer=tf.constant_initializer(
                                               value=0,
                                               dtype=tf.int64),
                                           trainable=False,
                                           sloppy=True)
        incr_global_step = tf.assign(global_step, global_step+1)

    if targets is not None:
        # For batchnorm running statistics updates.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.name_scope("discriminator_train"):
            with tf.control_dependencies(update_ops):
                if config["gan_weight"] != 0.:
                    discrim_tvars = [var for var in tf.trainable_variables()
                                     if var.name.startswith("discriminator")]
                    discrim_optim = tf.train.AdamOptimizer(config["lr"],
                                                           config["beta1"])
                    discrim_train = discrim_optim.minimize(
                        discrim_loss,
                        var_list=discrim_tvars)
                else:
                    dummyvar = tf.Variable(0)
                    discrim_train = tf.assign(dummyvar, 0)
        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables()
                             if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(config["lr"],
                                                   config["beta1"])
                gen_train = gen_optim.minimize(gen_loss, var_list=gen_tvars)
        train = tf.group(incr_global_step, gen_train)
        display_discrim_loss = discrim_loss
        display_gen_loss_GAN = (gen_loss_GAN *
                                config["gan_weight"])
        display_gen_loss_recon = (gen_loss_recon *
                                  config["recon_weight"])
        display_gen_loss_latent = (gen_loss_latent *
                                   config["latent_weight"])
        if (config["model_version"] in ['vae', 'cvae'] and
            config["input_as_class"]):
            gen_accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(unnormalized_outputs, axis=3),
                             tf.argmax(targets, axis=3)), tf.float32))
        else:
            gen_accuracy = tf.constant(0., dtype=tf.float32)
    else:
        train = incr_global_step
        display_discrim_loss = discrim_loss
        display_gen_loss_GAN = gen_loss_GAN
        display_gen_loss_recon = gen_loss_recon
        display_gen_loss_latent = gen_loss_latent
        gen_accuracy = tf.constant(0., dtype=tf.float32)
    LOGGER.info("Model complete.")
    return Model(
        z_mean=z_mean,
        z_log_sigma_sq=z_log_sigma_sq,
        z=z,
        y=y,
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=display_discrim_loss,
        gen_loss_GAN=display_gen_loss_GAN,
        gen_loss_recon=display_gen_loss_recon,
        gen_loss_latent=display_gen_loss_latent,
        gen_accuracy=gen_accuracy,
        outputs=outputs,
        train=train,
        global_step=global_step,
    )
