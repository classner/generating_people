config = {
    # Model. #######################
    # Supported are: 'portray', 'vae', 'cvae'.
    "model_version": 'portray',
    # What modes are available for this model.
    "supp_modes": ['train', 'val', 'trainval', 'test'],
    # Number of Generator Filters in the first layer.
    # Increases gradually to max. 8 * ngf.
    "ngf": 64,
    # Number of Discriminator Filers in the first layer.
    # Increases gradually to max. 8 * ndf.
    "ndf": 64,
    # Number of latent space dimensions (z) for the vae.
    "nz": 512,
    # If "input_as_class", use softmax xentropy. Otherwise sigmoid xentropy.
    "iac_softmax": True,
    # No interconnections inside of the model.
    "cvae_noconn": False,
    # Omit variational sampling, making the CVAE a CAE.
    "cvae_nosampling": True,
    # How many samples to draw per image for CVAE.
    "cvae_nsamples_per_image": 5,
    # Instead of building a y encoder, just downscale y.
    "cvae_downscale_y": False,

    # Data. ########################
    # Whether to use on-line data augmentation by flipping.
    "flip": False,
    # Whether to treat input data as classification or image.
    "input_as_class": True,
    # Whether to treat conditioning data as classification or image (CVAE only).
    "conditioning_as_class": False,
    # If input_as_class True, it is possible to give class weights here. (If
    # undesired, set to `None`.) This is a dictionary from class=>weight.
    "class_weights": None,
    # If True, weights are applied only for recall computation. Otherwise,
    # an approximation is used for Precision as well assuming that all weights
    # apart from the 1.0 weights are equal.
    "class_weights_only_recall": True,
    # Scale the images up to this size before cropping
    # them to `crop_size`.
    "scale_size": 286,
    "crop_size": 256,
    # Type of image content. Currently only supports "people".
    "dset_type": "people",
    "dset_suffix": "full",

    # Optimizer ####################
    "max_epochs": 150,
    "max_steps": None,
    "batch_size": 1,
    "lr": 0.0002,  # Adam lr.
    "beta1": 0.5,  # Adam beta1 param.
    "gan_weight": 1.0,
    "recon_weight": 100.0,
    "latent_weight": 0.,

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
