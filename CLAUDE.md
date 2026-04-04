# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis (TFM) developing a Generative AI framework to produce synthetic tabular data demonstrating mathematical anonymity under GDPR. Uses the **Diabetes 130-US Hospitals (1999-2008)** dataset from UCI.

**Goal**: Prove synthetic data crosses the legal boundary to become "anonymous data" (GDPR-exempt) while maintaining statistical utility for ML in healthcare/finance.

## Development Commands

```bash
# Build and start Docker container
docker compose up --build -d

# Verify GPU detection (AMD RX9070)
docker compose exec tfm python3 -c "import torch; print(torch.cuda.is_available())"

# Access container shell
docker compose exec tfm /bin/bash

# Jupyter Lab: http://localhost:8888
```

## Architecture & Data Flow

```
compartida/
├── data/
│   ├── diabetic_data.csv          # Raw: 101,766 records × 50 vars
│   ├── diabetic_data_clean.csv    # Processed: 99,340 records × 39 vars
│   └── IDS_mapping.csv            # ICD-9 diagnosis code mappings
├── notebooks/
│   ├── 01_eda_detallado.ipynb     # Exploratory analysis
│   └── 02_limpieza_ingenieria.ipynb # Cleaning & feature engineering
└── docs/                          # Spanish reports (EDA, cleaning, process log)
```

**Pipeline**: Raw CSV → EDA (notebook 01) → Cleaning/Feature Engineering (notebook 02) → Clean CSV → Generative Models (Phase 3) → Evaluation (Phase 4) → Privacy Attacks (Phase 5)

## Tech Stack

- **Python 3.10**, PyTorch, Docker (Python-slim base)
- **Data**: pandas, numpy, scipy, scikit-learn
- **Generative**: sdv, ctgan, rdt (Tabular Diffusion/CTGAN/TVAE)
- **Viz**: matplotlib, seaborn
- **Dev**: Jupyter Lab (port 8888)

## Key Data Transformations (Completed)

- **Excluded**: 2,423 deceased/hospice patients (`discharge_disposition_id` in [11,13,14,19,20,21])
- **Removed**: `weight` (97% null), `payer_code` (40% null), low-variance medications
- **ICD-9 Grouping**: 100s of codes → 9 clinical categories (Circulatory, Respiratory, Digestive, Diabetes, Injury, Genitourinary, Musculoskeletal, Neoplasms, Other)
- **New Features**: `prior_visits`, `any_med_change`
- **Target**: `readmitted` (<30, >30, NO) — 11.1% minority class

## Phase 3 Scripts (Generative Engine)

```
compartida/scripts/
├── config.py              # Centralized hyperparameters, paths, column metadata
├── data_loader.py         # Loads clean CSV, casts types, returns SDV metadata
├── check_environment.py   # Verifies GPU, libs, datasets before training
├── train_ctgan.py         # CTGAN training + synthetic data generation
├── train_tvae.py          # TVAE training + latent space diagnostics
├── train_tabddpm.py       # TabDDPM from scratch (PyTorch): MLP denoiser + DDPM sampling
└── run_all_models.py      # Orchestrator: runs all three models sequentially
```

**Execution order:**
```bash
python scripts/check_environment.py          # Verify setup
python scripts/run_all_models.py --quick     # Fast test (50 epochs)
python scripts/run_all_models.py             # Full training
```

**Outputs:**
- `compartida/outputs/synthetic_ctgan.csv`
- `compartida/outputs/synthetic_tvae.csv`
- `compartida/outputs/synthetic_tabddpm.csv`
- `compartida/models/` — saved model checkpoints

## Next Implementation Files

- `compartida/notebooks/03_model_implementation.ipynb` — orchestrates the scripts above
- `compartida/notebooks/04_evaluation.ipynb` — Fidelity metrics, TSTR validation
- `compartida/notebooks/05_privacy_attacks.ipynb` — DCR, MIA tests

## Conventions

- Spanish for notebook markdown (documentation language)
- English for variable names and code comments
- All work in `compartida/` (Docker mounted volume)
