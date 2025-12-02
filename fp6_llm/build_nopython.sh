#!/bin/bash
mkdir -p build_nopython
pushd build_nopython
rm -rf *
cmake -DWITH_PYTORCH=OFF -DCMAKE_CUDA_ARCHITECTURES=80 ..
make -j $(nproc)
popd    
