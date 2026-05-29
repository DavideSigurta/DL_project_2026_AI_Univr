# ISIC Melanoma Project

Binary melanoma classification (ISIC 2018 & ISIC 2020).

## Setup

### 1) Create environment

```bash
conda env create -f environment.yml
conda activate isic-dl
```

### 2) Configure Kaggle credentials

Place your Kaggle API token in your home directory (never commit it):

```bash
mkdir -p ~/.kaggle && echo your_token_here > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```

Get your token from https://www.kaggle.com/settings/api

### 3) Download datasets

Open and run:

```
notebooks/00_setup.ipynb
```

This downloads ISIC 2018 & ISIC 2020 and generates processed CSVs.

