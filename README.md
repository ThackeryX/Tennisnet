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
