#!/usr/bin/env bash

rm -rf build
mkdir build
cd build
cmake ../
make

nthreads=4
l="0.002"
lambda="0.05"
dim=50
path="../../mf_data/netflix"

./nomad_double --nthreads $nthreads --lrate $l --reg $lambda --dim $dim --path $path