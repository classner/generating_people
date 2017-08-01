OA# Generating People code repository

Requirements:

* OpenCV (on Ubuntu, e.g., install libopencv-dev and python-opencv).
* SMPL (download at http://smpl.is.tue.mpg.de/downloads) and unzip to a
  place of your choice.
* Edit the file `config.py` to set up the paths.
* `tensorflow` or `tensorflow-gpu` in a version >=v1.1.0 (I did not want to add
  it to the requirements to force installation of the GPU or non-GPU version).

The rest of the requirements is then automatically installed when running:

```
python setup.py develop
```

## Setting up the data

The scripts in `generation/tools/` transform the Chictopia data to construct to
the final database. Iteratively go through the scripts to create it. Otherwise,
download the pre-processed data from our website
(http://files.is.tuebingen.mpg.de/classner/gp/), unzip it to the folder
`generation/data/pose/extracted` and only run the last script

```
./09_pack_db.sh full
```

## Training / running models

Model configuration and training artifacts are in the `experiments` folder. The
`config` subfolder contains model configurations (LSM=latent sketch module,
CSM=conditional sketch module, PM=portray module, PSM_class=portray module with
class input). You can track the contents of this folder with git since it's
lightweight and no artifacts are stored there. To create a new model, just copy
`template` (or link to the files in it) and change `options.py` in the new
folder.

To run training/validation/testing use 

```
./run.py [train,val,trainval,test,{sample}] experiments/config/modelname
```

where `trainval` runs a training on training+validation. Artifacts during
training are written to `experiments/states/modelname` (you can run a
tensorboard there for monitoring). The generated results from testing are stored
in `experiments/features/modelname/runstate`, where runstate is either a
training stage or point in time (if sampling). You can use the `test_runner.py`
script to automatically scan for newly created training checkpoints and
validating/testing them with the command 

```
./test_runner.py experiments/states/modelname [val, test]
```

Pre-trained models can be downloaded from
http://files.tuebingen.mpg.de/classner/gp .

## Generating people

If you have trained or downloaded the LSM and PM models, you can use a
convenience script to sample people. For this, navigate to the `generation`
folder and run

```
./generate.sh n_people [out_folder]
```

to generate `n_people` to the optionally specified `out_folder`. If unspecified,
the output folder is set to `generated`.

## Citing

If you use this code for your research, please consider citing us:

```
@INPROCEEDINGS{Lassner:GeneratingPeople:2017,
  author    = {Christoph Lassner and Gerard Pons-Moll and Peter V. Gehler},
  title     = {A Generative Model for People in Clothing},
  year      = {2017},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision}
}
```

## Acknowledgements

Our models are strongly inspired by the pix2pix line of work by Isola et al.
(https://phillipi.github.io/pix2pix/). Parts of the code are inspired by the
implementation by Christopher Hesse (https://affinelayer.com/pix2pix/). Overall,
this repository is set up similar to the Deeplab project structure, enabling
efficient model specification, tracking and training
(http://liangchiehchen.com/projects/DeepLabv2_resnet.html) and combining it with
the advantages of Tensorboard.
