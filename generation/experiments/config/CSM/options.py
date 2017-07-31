config = {
    # Model. #######################
    # Supported are: 'pix2pix', 'vae', 'cvae'.
    "model_version": 'cvae',
    # What modes are available for this model.
    "supp_modes": ['train', 'val', 'trainval', 'test', 'sample'],
    # Number of Generator Filters in the first layer.
    # Increases gradually to max. 8 * ngf.
    "ngf": 64,  # 64
    # Number of Discriminator Filers in the first layer.
    # Increases gradually to max. 8 * ndf.
    "ndf": 64,  # 64
    # Number of latent space dimensions (z) for the vae.
    "nz": 32,
    # If "input_as_class", use softmax xentropy. Otherwise sigmoid xentropy.
    "iac_softmax": False,
    # No interconnections inside of the model.
    "cvae_noconn": True,
    # Omit variational sampling, making the CVAE a CAE.
    "cvae_nosampling": False,
    # How many samples to draw per image for CVAE.
    "cvae_nsamples_per_image": 5,
    # Instead of building a y encoder, just downscale y.
    "cvae_downscale_y": False,
    # Use batchnorm for the vector encoding in the pix2pix model.
    "pix2pix_zbatchnorm": True,

    # Data and preprocessing. ########################
    # Whether to use on-line data augmentation by flipping.
    "flip": False,
    # Whether to treat input data as classification or image.
    "input_as_class": True,
    # Whether to treat conditioning data as classification or image.
    "conditioning_as_class": True,
    # If input_as_class True, it is possible to give class weights here. (If
    # undesired, set to `None`.) This is a dictionary from class=>weight.
    "class_weights": {
        18: 10.,
        19: 10.,
        20: 10.,
        21: 10.,
    },
    # If True, weights are applied only for recall computation. Otherwise,
    # an approximation is used for Precision as well assuming that all weights
    # apart from the 1.0 weights are equal.
    "class_weights_only_recall": True,
    # Scale the images up to this size before cropping
    # them to `crop_size`.
    "scale_size": 286,
    "crop_size": 256,
    # Type of image content.
    "dset_type": "people",
    "dset_suffix": "full",

    # Optimizer ####################
    "max_epochs": 70,
    "max_steps": None,
    "batch_size": 40,
    "lr": 0.0002,  # Adam lr.
    "beta1": 0.5,  # Adam beta1 param.
    "gan_weight": 0.0,
    "recon_weight": 1.0,
    "latent_weight": 7.0,

    # Infrastructure. ##############
    # Save summaries after every x seconds.
    "summary_freq": 30,
    # Create traces after every x batches.
    "trace_freq": 0,
    # After every x epochs, save model.
    "save_freq": 10,
    # Keep x saves.
    "kept_saves": 10,
    # After every x epochs, render images.
    "display_freq": 10,
    # Random seed to use.
    "seed": 538728914,
}
