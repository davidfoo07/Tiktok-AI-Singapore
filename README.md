# AISG

## Setting up the environment
1. Run on Linux system only (due to **flash attention 2** support)

2. Install Python 3.11.*\
 [Download Python 3.11.*](https://www.python.org/downloads/)

3. Install CUDA 12.4.1 (Tested on **2x H200 SXM** 141GB VRAM GPU)\
 [Download CUDA 12.4.1](https://developer.nvidia.com/cuda-12-4-1-download-archive)

4. Install PyTorch 2.4.0
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

5. Run **InternVL3.ipynb**