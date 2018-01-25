#!/bin/bash
set -e  # Exit on error.
if false; then
if [ -z ${1+x} ]; then
   echo Please specify the number of people! >&2; exit 1
fi
npeople=$1
re='^[0-9]+$'
if ! [[ $1 =~ $re ]] ; then
    echo "Error: specify a number" >&2; exit 1
fi
if [ -z ${2+x} ] || [ ! ${2: -4} == ".png" ] || [ ! -f $2 ]; then
    echo "Error: please provide a path to the image and .pkl file with the "\
         "conditioning (provide the image filename in the form "\
         "'00001_image.png'; '00001_body.pkl' must exist as well." >&2; exit 1
else
    image_fn=$(basename $2)
    body_fn="${image_fn%_*}_body.pkl"
    image_fp=$2
    body_fp="$(dirname ${image_fp})/${body_fn}"
    if [ ! -f ${body_fp} ]; then
        echo "Error: please provide a path to the image and .pkl file with the "\
             "conditioning (provide the image filename in the form "\
             "'00001_image.png'; '00001_body.pkl' must exist as well. "\
             "Did't find '${body_fp}'.">&2; exit 1
    fi
fi
if [ -z ${3+x} ]; then
    out_fp=generated
else
    out_fp=$3
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
if [ ! -d experiments/states/CSM ]; then
    echo "State folder for the conditional sketch module not found at " >&2
    echo "'experiments/states/CSM'. Either run the training (./run.py trainval experiments/config/CSM) " >&2
    echo "or download a pretrained model from http://gp.is.tuebingen.mpg.de." >&2
    exit 1
fi
if [ ! -d experiments/states/PM ]; then
    echo "State folder for the portray module not found at " >&2
    echo "'experiments/states/PM'. Either run the training (./run.py trainval experiments/config/PM) " >&2
    echo "or download a pretrained model from http://gp.is.tuebingen.mpg.de." >&2
    exit 1
fi

echo Generating ${npeople} people from conditioning provided in ${body_fp}...
echo Creating 2D conditioning segments...
mkdir -p tmp/dset/test
cp ${image_fp} tmp/dset/test/0_image.png
cp ${body_fp} tmp/dset/test/0_image.png_body.pkl
mkdir -p tmp/prepared_input/test
./tools/06_render_bodies.py --dset_folder=tmp/dset --out_folder=tmp/prepared_input
cp ${image_fp} tmp/prepared_input/test/0_image.png
cp tmp/prepared_input/test/0_bodysegments.png tmp/prepared_input/test/0_labels.png
cp tmp/prepared_input/test/0_bodysegments_vis.png tmp/prepared_input/test/0_labels_vis.png
echo "0_conditioning.png" > tmp/prepared_input/test/0_original_filename.txt
cp tmp/prepared_input/test/0_bodysegments_vis.png tmp/prepared_input/test/0_segcolors.png
echo Creating archive...
mkdir -p data/people/tmp
tfrpack tmp/prepared_input/test --out_fp data/people/tmp/val
echo Sampling with conditioning...
mkdir -p tmp/csm_out
./run.py sample experiments/config/CSM --override_dset_suffix tmp --out_fp tmp/csm_out --n_samples ${npeople}
echo Done.
echo Preparing for portray module...
mkdir tmp/portray_dset
for sample_idx in $(seq 0 $((${npeople}-1))); do
    fullid=$(printf "%04d" ${sample_idx})
    # Simulate full dataset.
    cp tmp/csm_out/images/${sample_idx}-0_conditioning_outputs.png tmp/portray_dset/${fullid}_bodysegments:png.png
    cp tmp/csm_out/images/${sample_idx}-0_conditioning_outputs.png tmp/portray_dset/${fullid}_bodysegments_vis:png.png
    cp tmp/csm_out/images/${sample_idx}-0_conditioning_outputs.png tmp/portray_dset/${fullid}_image:png.png
    cp tmp/csm_out/images/${sample_idx}-0_conditioning_outputs.png tmp/portray_dset/${fullid}_labels:png.png
    cp tmp/csm_out/images/${sample_idx}-0_conditioning_outputs.png tmp/portray_dset/${fullid}_label_vis:png.png
    echo ${fullid}_sample.png > tmp/portray_dset/${fullid}_original_filename.txt
done
echo Creating archive...
mkdir -p data/people/tmp2
tfrpack tmp/portray_dset --out_fp data/people/tmp2/test
echo Done.
echo Creating images...
./run.py test experiments/config/PM --override_dset_suffix tmp2 --out_fp ${out_fp}
echo Done.
echo Cleaning up...
rm -rf tmp
rm -rf data/people/tmp
rm -rf data/people/tmp2
echo Done.
