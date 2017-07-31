#!/bin/bash

if [ ! -d ../data/pose ]; then
    mkdir ../data/pose
fi

if [ ! -d ../data/pose/extracted ]; then
    mkdir ../data/pose/extracted
    mkdir ../data/pose/extracted/{train,val,test}
    mkdir ../data/pose/input
    mkdir ../data/pose/input/{train,val,test}
fi

for part in train val test; do
    tfrcat ../data/people/$1/${part} --out_fp ../data/pose/extracted/${part}
    cp ../data/pose/extracted/${part}/*_image*.png ../data/pose/input/${part}/
done

