#!/bin/bash

uv pip3 install --upgrade pip
uv pip3 install pillow transformers accelerate modelscope opencv-python tqdm addict simplejson sortedcontainers
uv pip install numpy==1.24.4
uv pip install "pyarrow==12.0.1"
uv pip install "modelscope[datasets]"
git clone https://github.com/facebookresearch/dinov3.git
cd dinov3
# Create working directory
mkdir test_dinov3
cd test_dinov3
# Edit test code
vim dinov3_vision_test.py