#!/bin/bash
export PATH=/usr/local/cuda-11.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
echo "✅ CUDA 11.0 환경 활성화됨"
nvcc --version
