# DUAL-TRITS

need to install libmpfr-dev libmpfrc++-dev libgmp-dev

# phases

## phase 0
Implement this format in C++ and CUDA, support basic arithmetic operations, support convert to and convert from standard formats (FP32, FP16, FP4), benchmark the software implementation speed impact compared to FP4.

## phase 1

Implement a PyTorch/CUDA layer that:Stores weights in dual-trit format (compressed),Decodes on-demand to FP8/FP16 during forward pass,Caches decoded weights if memory allows.

## phase 2

Compare accuracy and memory usage of deep learning networks using both formats.
## phase 3

Quantization-Aware Training (QAT)

1. Fine-tune pre-trained models with dual-trit simulation
1. Compare convergence vs FP4-QAT
1. Comparison with SOTA Methods
