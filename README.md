# 🦙 KAGGLE_OPENVACCINE_MRNA 🦙

✍ Authors:

**BEL Alexis** - **BELAKTIB Anas** - **OUSSAREN Mohamed** - **ROUAUD Lucas**

Master 2 Bio-informatics at *Univerité de Paris*.

[![Python 3.9.7](https://img.shields.io/badge/python-%E2%89%A5_3.9.7-blue.svg)](https://www.python.org/downloads/release/python-397/)
[![Conda 3.10.6](https://img.shields.io/badge/miniconda-%E2%89%A5_3.10.6-green.svg)](https://docs.conda.io/en/latest/miniconda.html)
[![GitHub last commit](https://img.shields.io/github/last-commit/FilouPlains/KAGGLE_OPENVACCINE_MRNA.svg)](https://github.com/FilouPlains/KAGGLE_OPENVACCINE_MRNA)
![GitHub stars](https://img.shields.io/github/stars/FilouPlains/KAGGLE_OPENVACCINE_MRNA.svg?style=social)

## 🔎 Interesting path
- 📑 Report: `doc/report`
- 📢 Oral presentation: `doc/presentation`
- 🖥 Main: `src/main.py`

## 🤔 Context

> This project is actually trying to answer to this Kaggle project: [**OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction**](https://www.kaggle.com/competitions/stanford-covid-vaccine).

The main problematic is the synthesis of a stable mRNA vaccine. Because of the molecular nature of it, the vaccine degraded itself easily and quickly. To counter that, it is necessary to synthesize a mRNA stable.

To do so, the product mRNA have to be tested. And this is where this project take place: creating a neural network to predict the stability of a given sequence.

## 🧐 Methods Implemented

Because this project is done to validate a course, one mandatory criterion is create two neural network approches. Here, we've done:
- Three different embedding:
  - One from `keras`.
  - One recode by ourselves.
  - One using `RNABERT` transformer.
- Two neural networks:
  - One Convutional Neural Network (CNN).
  - One Google Inception.

## 🚀 Launching this program

### 🐍 Conda environment

To use this program, you will need to create a conda environment like so:

```bash
mamba env create --file kaggle_reseau.yml
conda env create --file kaggle_reseau.yaml
conda activate reseau
```

### ⚙️ General method

To launch this program, simply use the next commands (after the activation of the conda environment):

```bash
python3 src/main.py --help
```

### 🔎 Parameters description

Next, the parameters are described:



|        **Parameters**        | **Parameters name**                                  | **Usage**                                               |
| :--------------------------: | :--------------------------------------------------- | :------------------------------------------------------ |
|     **\***`-i, --input`      | Input `X` + `Y` data/neural network trained          | Add an `.npy` data file or a `.h5` neural network file. |
|     **\***`-o, --output`     | Output data `Y`/neural network finish to be trained. | Add an `.npy` data file or a `.h5`   neural network file. |
|   `-pred, --predict_data`    | Output predicted `Y` data.                           | Add an `.npy` data file.                                |
|           `--cnn`            | Convolutional Neuronal Network.                      | Add like a `True`.                                      |
|`--ginc, --google_inception`  | Google inception's neural network.                   | Add like a `True`.                                      |
|  `--ke, --keras_embedding`   | Using classical `keras` embedding method.            | Add like a `True`.                                      |
|   `--owe, --own_embedding`   | Using our compute pre-embedding.                     | Add like a `True`.                                      |
| `--re, --rnabert_embedding`  | Using embedding compute by `RNABERT` transformer.    | Add like a `True`.                                      |

### 🧠 One program, two usage

#### **If you don't use the parameters `-pred, --predict_data`:**

You actually said to the program that you want to train neural network. To do that, give to `-i, --input` a dataset to learn and to `-o, --output` the neural network to reuse. Do not forget to indicate a type of neural network (`-ginc, --google_inception` or `-ke, --keras_embedding`) to use and a type of an input embedding (`-ke, --keras_embedding`, `-hme, --homemade_embedding` or `-re, --rnabert_embedding`).

#### **If you use the parameters `-pred, --predict_data`:**

You actually said to the program that you already have a train neural network. So you want to predict `Y` data base on `X` data. To do that, give to `-i, --input` a trained neural network, to `-o, --output` how to write the `Y` predict data and to `-pred, --predict_data` the `X` data. Do not forget to indicate the good input embedding (`-ke, --keras_embedding`, `-hme, --homemade_embedding` or `-re, --rnabert_embedding`).
