#!/bin/bash
set -e  # Exit on error.
if [ -z ${1+x} ]; then
   echo Please specify the number of people! >&2; exit 1
fi
npeople=$1
re='^[0-9]+$'
if ! [[ $1 =~ $re ]] ; then
    echo "Error: specify a number" >&2; exit 1
fi

if [ -z ${2+x} ]; then
    out_fp=generated
else
    out_fp=$2
fi
if [ -e ${out_fp} ]; then
    echo "Output folder exists: ${out_fp}. Please pick a non-existing folder." >&2
    exit 1
fi

# Check environment.
if [ -e tmp ]; then
    echo "The directory 'tmp' exists, maybe from an incomplete previous run?" >&2
    echo "If so, please delete it and rerun so that it can be used cleanly." >&2
    exit 1
fi
if [ -e data/people/tmp ]; then
    echo "The directory 'data/people/tmp' exists, maybe from an incomplete previous run?" >&2
    echo "If so, please delete it and rerun so that it can be used cleanly." >&2
    exit 1
fi
if [ ! -d experiments/states/LSM ]; then
    echo "State folder for the latent sketch module not found at " >&2
    echo "'experiments/states/LSM'. Either run the training (./run.py trainval experiments/config/LSM) " >&2
    echo "or download a pretrained model from http://gp.is.tuebingen.mpg.de." >&2
    exit 1
fi
if [ ! -d experiments/states/PM ]; then
    echo "State folder for the portray module not found at " >&2
    echo "'experiments/states/PM'. Either run the training (./run.py trainval experiments/config/PM) " >&2
    echo "or download a pretrained model from http://gp.is.tuebingen.mpg.de." >&2
    exit 1
fi

echo Generating $1 people...
echo Sampling sketches...
./run.py sample experiments/config/LSM --out_fp tmp --n_samples ${npeople}
echo Done.
echo Preparing for portray module...
mkdir tmp/portray_dset
for sample_idx in $(seq 0 $((${npeople}-1))); do
    fullid=$(printf "%04d" ${sample_idx})
    # Simulate full dataset.
    cp tmp/images/${sample_idx}_outputs.png tmp/portray_dset/${fullid}_bodysegments:png.png
    cp tmp/images/${sample_idx}_outputs.png tmp/portray_dset/${fullid}_bodysegments_vis:png.png
    cp tmp/images/${sample_idx}_outputs.png tmp/portray_dset/${fullid}_image:png.png
    cp tmp/images/${sample_idx}_outputs.png tmp/portray_dset/${fullid}_labels:png.png
    cp tmp/images/${sample_idx}_outputs.png tmp/portray_dset/${fullid}_label_vis:png.png
    echo ${fullid}_sample.png > tmp/portray_dset/${fullid}_original_filename.txt
done
echo Creating archive...
mkdir -p data/people/tmp
tfrpack tmp/portray_dset --out_fp data/people/tmp/test
echo Done.
echo Creating images...
./run.py test experiments/config/PM --override_dset_suffix tmp --out_fp ${out_fp}
echo Done.
echo Cleaning up...
rm -rf tmp
rm -rf data/people/tmp
echo Done.

