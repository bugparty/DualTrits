#!/bin/bash
pushd build
#rm -rf *
cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
make -j $(nproc)
popd    