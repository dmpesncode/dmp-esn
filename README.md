# Dynamical Mode Pruning for Echo State Networks

This repository contains the code for **Dynamical Mode Pruning (DMP)**, a pruning method for Echo State Networks (ESNs). DMP identifies important reservoir neurons using dominant dynamical modes extracted from a trajectory-averaged Jacobian Gram matrix.

The code includes implementations for the proposed DMP method and several ESN baselines used in the experiments.

## Overview

Echo State Networks use a fixed recurrent reservoir and train only the output readout. However, randomly initialized reservoirs can be over-parameterized. This repository studies reservoir pruning methods that reduce reservoir size while preserving forecasting performance.

DMP scores reservoir neurons according to their contribution to dominant state-transition modes. Low-scoring neurons are removed, the pruned reservoir is stabilized, and the readout is retrained.

## Repository Structure

```text
.
├── data/                  # Time-series datasets
├── results/               # Saved experimental outputs
├── src/
│   ├── dmp_model/          # DMP implementation and result saving
│   ├── sota_models/        # Baseline ESN models
│   └── utils/              # Data loading and table formatting utilities
├── static/
│   └── config.yml          # Experiment configuration
├── main.py                 # Main training script
└── README.md
````

## Models

The repository includes:

* **Base ESN**
* **Leaky ESN**
* **Deep ESN**
* **Betweenness-pruned ESN**
* **Closeness-pruned ESN**
* **Dynamical Mode Pruning (DMP)**

## Data

The experiments use five one-dimensional time-series datasets:

* Mackey--Glass
* Electricity
* Temperature
* Wind
* Solar

The datasets are stored as single-column CSV files under `data/`. The data sources follow the descriptions and citations provided in the paper. The datasets are publicly available and can be easily obtained from the cited sources.

## Configuration

All main experiment settings are controlled by:

```text
static/config.yml
```

This file defines:

* datasets
* reservoir sizes
* forecasting horizons
* pruning ratios
* random seeds
* ESN hyperparameters
* result-saving paths

## Running Experiments

Install the required Python packages, then run:

```bash
python main_train.py
```

The script trains all enabled models and saves metrics, predictions, plots, and LaTeX tables.

## Outputs

Results are saved under:

```text
results/
```

Each run stores:

* `metrics.csv`
* prediction CSV files
* true-vs-prediction plots
* error heatmaps
* pruning diagnostics, when applicable

LaTeX tables are generated automatically under:

```text
results/<results_template>/latex_tables/
```