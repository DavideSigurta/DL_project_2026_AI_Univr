# ISIC Melanoma Project

Binary melanoma classification comparing supervised, semi-supervised, and domain adaptation methods under label scarcity and domain shift.

**Key challenge:** ISIC 2018 (source) has 10.98% malignant prevalence; ISIC 2020 (target) has 1.7% — a 6.5× prior shift. Target test set has only ~58 positives, giving AUC comparisons ±7.6pp 95% CI.

**Methods compared:**
- **E1/E2** — Supervised baseline (ResNet18) + label budget ablation (1%–100%)
- **E3** — SimCLR self-supervised pretraining + fine-tuning
- **E4** — Pseudo-labeling on unlabeled target
- **E5** — Mean Teacher consistency regularization
- **E6** — Domain Adversarial Neural Network (DANN)

All methods evaluated on both source (ISIC 2018) and target (ISIC 2020) test sets across 6 label fractions.

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

## Key Results

| Method | Source (ISIC 2018) | Target (ISIC 2020) |
|--------|-------------------|-------------------|
| **Supervised baseline** (full labels) | AUC 0.928 | AUC 0.698 |
| **SimCLR SSL** (full labels) | AUC 0.934 (+0.6pp) | AUC 0.694 (—) |
| **SimCLR SSL** (1% labels) | AUC 0.845 (+3.1pp) | AUC **0.798 (+13.7pp)** |
| Pseudo-labeling | No improvement | Degrades at 100% (−5pp) |
| Mean Teacher | Degrades at 1% (−7.4pp) | Collapses at 1% (−11pp) |
| DANN | Neutral | Degrades under prior shift |

**Takeaway:** Three semi-supervised/DA methods (pseudo-label, Mean Teacher, DANN) fail on this dataset, each for a different reason rooted in the 6.5× prior shift. SimCLR SSL is the only method that improves target AUC at low label budgets, because it makes no assumptions about target label distribution.
