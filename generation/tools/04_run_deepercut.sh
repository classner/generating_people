#!/bin/sh

up_fp=$(../../config.py UP_FP)

for part in train val test; do
    ${up_fp}/pose/pose_deepercut.py ../data/pose/input/${part}
done
