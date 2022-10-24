
## 1. Environment setup 🔧

## 🐍 main.py

[![Python 3.9.7](https://img.shields.io/badge/python-%E2%89%A5_3.9.7-blue.svg)](https://www.python.org/downloads/release/python-397/)
[![Conda 3.10.6](https://img.shields.io/badge/miniconda-%E2%89%A5_3.10.6-green.svg)](https://docs.conda.io/en/latest/miniconda.html)

### Install environment 📶

```bash
mamba env create --file kaggle_reseau.yml
conda env create --file kaggle_reseau.yml
conda activate reseau
```

### Module list 📝

| **Module**   | **Version** |
| :----------- | :---------: |
| keras        |   `2.7.0`   |
| matplotlib   |   `3.6.1`   |
| numpy        |  `1.23.3`   |
| pip          |   `22.3`    |
| python       |  `3.10.6`   |
| scikit-learn |   `1.1.2`   |
| scipy        |   `1.9.1`   |
| tensorflow   |   `2.7.0`   |


## 🐍 RNABERT

[![Python 3.6.4](https://img.shields.io/badge/python-_3.6.5-blue.svg)](https://www.python.org/downloads/release/python-365/)
[![PyTorch 1.4.0](https://img.shields.io/badge/PyTorch-%E2%89%A5_1.4.0-blue.svg)](https://pytorch.org/get-started/previous-versions/#v140)
[![Biopython 1.76](https://img.shields.io/badge/biopython-%E2%89%A5_1.76-blue.svg)](https://biopython.org/wiki/Download)

### Install environment 📶

```bash
mamba env create --file env_RNABERT.yaml
conda env create --file env_RNABERT.yaml
conda activate env_RNABERT_emb
```

### Module list 📝

| **Module**       | **Version** |
| :--------------- | :---------: |
| attrdict         |   `2.0.1`   |
| matplotlib       |   `3.3.4`   |
| numpy            |  `1.19.5`   |
| pip              |  `21.3.1`   |
| python           |  `3.6.15`   |
| pytorch          |   `1.9.1`   |
| scikit-learn     |  `0.24.2`   |
| torchvision      |  `0.10.1`   |
| biopython **\*** |   `1.76`    |

**\*** is used with `pip`.
