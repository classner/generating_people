#!/bin/sh

trap "exit" INT

out_suff=$1
if [ ! -d ../data/people ]; then
    mkdir ../data/people
fi

if [ -d ../data/people/${out_suff} ]; then
    echo A dataset with this suffix exists!
    exit 1
fi
mkdir ../data/people/${out_suff}

for dset_part in train val test; do
    ~/git/clustertools/clustertools/scripts/tfrpack.py \
        ../data/pose/extracted/${dset_part} \
        --out_fp ../data/people/${out_suff}/${dset_part}
done
