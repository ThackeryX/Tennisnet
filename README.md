# Tennisnet: Temporal Action Detection for Tennis Videos

<p align="left">
<a href="LICENSE" alt="license">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
<a href="https://huggingface.co/datasets/Tang1166/TennisDB" alt="HuggingFace">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TennisDB-yellow" /></a>
<a href="https://github.com/sming256/OpenTAD" alt="OpenTAD">
    <img src="https://img.shields.io/badge/Based%20on-OpenTAD-orange" /></a>
</p>

**Tennisnet** is a specialized temporal action detection (TAD) framework for tennis videos, integrating State Space Models (**MambaBMN**) to effectively capture long-range temporal dependencies in athletic actions.

> 📢 **Acknowledgement**: This project is developed based on the open-source toolbox [OpenTAD](https://github.com/sming256/OpenTAD). We sincerely appreciate the authors for their well-designed modular codebase.

---

## 🛠️ Installation

Our environment setup inherits directly from OpenTAD. Please refer to [install.md](docs/en/install.md) for full system requirements and compilation instructions.

### 1. Basic OpenTAD Setup
```bash
# Clone the repository
git clone https://github.com/ThackeryX/Tennisnet.git
cd Tennisnet

# Install dependencies and build OpenTAD in editable mode
pip install -r requirements.txt
pip install -v -e .
