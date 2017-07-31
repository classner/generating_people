#!/bin/sh

trap "exit" INT

for dset_part in train val test; do
    $(../../config.py UP_FP)/3dfit/bodyfit.py \
         ../data/pose/input/$dset_part/ --use_inner_penetration --only_missing \
          --allow_subsampling
done
