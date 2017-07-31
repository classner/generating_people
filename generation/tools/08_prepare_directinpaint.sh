#!/usr/bin/env zsh

if [ "${#}" -ne 2 ]; then
    echo Provide in and output!
    exit 1
fi

indset=$1
outdset=$2

mkdir ../data/people/${outdset}
mkdir ../data/people/${outdset}/test
for im in ../data/people/${indset}/*/*_pix2pix.png; do
    echo $(basename ${im})
    cp ${im} ../data/people/${outdset}/test/
done
