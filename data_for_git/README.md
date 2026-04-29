# Data

This directory contains the time-series datasets used in the experiments.

The experiments use five benchmark datasets:

| Dataset | File | Description |
|---|---|---|
| Mackey--Glass | `mackey_glass.csv` | Synthetic chaotic time-series benchmark generated from the Mackey--Glass delay differential equation. |
| Electricity | `electricity_consumption.csv` | Electricity consumption time series used for forecasting experiments. |
| Temperature | `temperature.csv` | Real-world temperature time series used to evaluate forecasting performance. |
| Wind | `wind.csv` | Wind-related time series used to evaluate robustness on weather-driven temporal dynamics. |
| Solar | `solar_consumption.csv` | Solar consumption / generation-related time series used for forecasting experiments. |

## Data Sources

The datasets used in this repository follow the sources described in the paper. They are standard, publicly available time-series datasets and can be easily obtained from the cited benchmark/data repositories.

For reproducibility, each dataset is stored as a single-column CSV file in this directory. The training script reads the first column of each CSV file and applies the same preprocessing pipeline used in the experiments.

Expected structure:

```text
data/
├── electricity.csv
├── mackey.csv
├── solar.csv
├── temperature.csv
├── wind.csv
└── README.md