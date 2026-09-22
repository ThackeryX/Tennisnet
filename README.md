# Tennisnet: Temporal Action Detection for Tennis Videos

<p align="left">
<a href="LICENSE" alt="license">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
<a href="https://huggingface.co/datasets/Tang1166/TennisDB" alt="HuggingFace">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TennisDB-yellow" /></a>
<a href="https://github.com/sming256/OpenTAD" alt="OpenTAD">
    <img src="https://img.shields.io/badge/Based%20on-OpenTAD-orange" /></a>
</p>

**Tennisnet** is a temporal action detection framework specifically tailored for tennis videos, integrating State Space Models (**MambaBMN**) to capture long-range action context.

> 📢 **Acknowledgement**: This project is built upon the [OpenTAD](https://github.com/sming256/OpenTAD) benchmark. We thank the authors for their well-engineered foundation.

---

## 🛠️ Step 1: Complete Environment Setup

Run the following commands in your terminal to set up the entire environment from scratch:

```bash
# 1. Create and activate conda environment
conda create -n tennisnet python=3.10 -y
conda activate tennisnet

# 2. Install PyTorch matching your CUDA version (CUDA 11.8 example)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# 3. Install build tools for Mamba compilation
pip install ninja packaging wheel setuptools

# 4. Install Mamba dependencies (Selective Scan CUDA kernels)
pip install causal-conv1d>=1.2.0 --no-build-isolation
pip install mamba-ssm>=1.2.0 --no-build-isolation

# 5. Clone Tennisnet and install OpenTAD components
git clone https://github.com/ThackeryX/Tennisnet.git
cd Tennisnet
pip install -r requirements.txt
pip install -v -e .

```

---

## 📦 Step 2: Automatic Data Preparation
All extracted features and annotations for TennisDB are hosted on Hugging Face: Tang1166/TennisDB.

You can automatically download and place the entire dataset into the data/tennis directory using this command:
# Install Hugging Face Hub CLI & client
```bash
pip install huggingface_hub

# Download TennisDB directly into the data folder
python -c "
from huggingface_hub import snapshot_download
import os

print('Downloading TennisDB from Hugging Face...')
snapshot_download(
    repo_id='Tang1166/TennisDB',
    repo_type='dataset',
    local_dir='data/tennis',
    local_dir_use_symlinks=False
)
print('Dataset downloaded successfully into data/tennis/')
"

Verify that your directory matches the following structure:
Tennisnet/
├── configs/
│   └── tennisnet/
│       └── tennisdb.py
├── data/
│   └── tennisdb/
│       ├── features/
│       └── classifiers/
│           └── tennisdb_scores.npy
├── opentad/
└── tools/
```

---
## 🚀 Step 3: Training & Evaluation
All commands follow standard OpenTAD execution protocols.
```bash
1. Training
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py configs/tennisnet/tennisdb.py
2. Testing & Evaluation
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/tennisdb/tennisdb.py \
    --checkpoint exps/tennisdb/tennisdb/gpu4_id0/checkpoint/epoch_13.pth
```

---
## 🖊️ Citation & References
This codebase is developed on top of the OpenTAD project. If you find this repository helpful for your research, please cite OpenTAD:
@article{liu2025opentad,
  title={OpenTAD: A Unified Framework and Comprehensive Study of Temporal Action Detection},
  author={Liu, Shuming and Zhao, Chen and Zohra, Fatimah and Soldan, Mattia and Pardo, Alejandro and Xu, Mengmeng and Alssum, Lama and Ramazanova, Merey and Alcázar, Juan León and Cioppa, Anthony and Giancola, Silvio and Hinojosa, Carlos and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2502.20361},
  year={2025}
}


