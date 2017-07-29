#!/usr/bin/env bash

rm -rf build
mkdir build
cd build
cmake ../
make

nthreads=4
l=0.001
lambda=0.1
dim=50
path="../../mf_data/ml10m"

./nomad_double --nthreads $nthreads --lrate $l --reg $lambda --dim $dim --path $path