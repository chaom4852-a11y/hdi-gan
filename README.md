# HDI-PRNet Unofficial Reproduction

This is an unofficial PyTorch reproduction of **HDI-PRNet: A Progressive Image Restoration Network for High-order Degradation Imaging in Remote Sensing**.

The official code is not public, so this implementation follows the paper's architecture diagrams and descriptions as closely as possible while making practical assumptions where details are missing.

## Implemented

- Progressive high-order restoration network
- Denoising module with 3-scale RCAB encoder-decoder
- SR module with Conv + bilinear interpolation + Conv
- Deblurring module with truncated Neumann expansion
- Dual-domain degradation learning block using spatial and frequency branches
- CIA/SIA/DEA feature interaction
- Reconstruction loss + intermediate supervision
- High-order synthetic degradation: blur -> resize -> noise, repeated k times
- Training, validation, synthetic testing, and real-image inference scripts

## Install

```bash
conda create -n hdi-prnet python=3.10 -y
conda activate hdi-prnet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python tqdm pyyaml numpy