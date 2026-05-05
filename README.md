# rMet: Transit Latency Prediction

**Thomas Bakewell · Ethan Klose · David Onks IV** — CS 4774, University of Virginia

Predicts next-stop bus delay (seconds) for NYC MTA buses using two models: a Random Forest baseline and a two-layer LSTM. Trained on the [MTA SIRI dataset](https://www.kaggle.com/datasets/stoney71/new-york-city-transport-statistics) (~26 M stop records across June, August, October, and December 2017).

The best LSTM run achieved a test MAE of **153.74 s**, beating the no-change baseline (187.42 s) by 18%.

---

## Repository layout

```
.
├── sort_cycles.py        # Step 1 — sort raw CSVs into route cycles
├── engineer_features.py  # Step 2 — compute delay_s, speed, time encodings, lag features
├── data_pipeline.py      # Trip reconstruction, temporal split, tensor builders
├── rnn_model.py          # Two-layer LSTM + MAE trainer
├── random_forest_model.py# RF baseline + random-search hyperparameter tuning
├── train.py              # Unified CLI entry point (RNN, RF, or both)
├── visualize.py          # Training curves and prediction plots
├── run_slurm.sh          # SLURM job script
└── requirements.txt
```

---

## Data pipeline

Raw data is not included in this repo. Download the four monthly CSVs from Kaggle and run the two preprocessing steps:

```
data/
  transit/        ← place raw CSVs here
    mta_1706.csv
    mta_1708.csv
    mta_1710.csv
    mta_1712.csv
  sorted/         ← produced by sort_cycles.py
  parsed/         ← produced by engineer_features.py
```

**Step 1 — sort into route cycles:**
```bash
python sort_cycles.py data/transit/mta_1706.csv data/sorted/mta_1706_sorted.csv
# repeat for each month
```

**Step 2 — engineer features:**
```bash
python engineer_features.py data/sorted/mta_1706_sorted.csv data/parsed/mta_1706_parsed.csv
# optionally include weather (NYC_Weather_2016_2022.csv from Kaggle):
python engineer_features.py data/sorted/mta_1706_sorted.csv data/parsed/mta_1706_parsed.csv data/NYC_Weather_2016_2022.csv
# repeat for each month
```

---

## Training

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run locally (RNN only):**
```bash
python train.py --csv data/parsed/mta_1706_parsed.csv \
                      data/parsed/mta_1708_parsed.csv \
                      data/parsed/mta_1710_parsed.csv \
                      data/parsed/mta_1712_parsed.csv \
                --model rnn
```

**Run locally (both models):**
```bash
python train.py --csv data/parsed/*.csv --model both
```

**Cache the processed trips to skip reloading on re-runs:**
```bash
python train.py --csv data/parsed/*.csv --cache-dir trips_cache --model rnn
```

**On SLURM (GPU partition):**
```bash
sbatch --export=ALL,DATA_DIR=/path/to/data/parsed run_slurm.sh

# Override hyperparameters:
sbatch --export=ALL,DATA_DIR=/path/to/data/parsed,EPOCHS=50,HIDDEN=256 run_slurm.sh
```

### Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--csv` | required | Parsed CSV files |
| `--model` | `both` | `rnn`, `rf`, or `both` |
| `--hidden` | `128` | LSTM hidden size |
| `--layers` | `2` | LSTM layers |
| `--lr` | `1e-3` | Learning rate |
| `--batch` | `64` | Batch size |
| `--epochs` | `100` | Max epochs |
| `--patience` | `10` | Early stopping patience |
| `--rf-search` | `20` | RF random-search trials |
| `--cache-dir` | none | Directory to cache processed trips |
| `--output-dir` | `outputs` | Where to write results |

---

## Features

**LSTM** (10 features — no lag features; the LSTM learns temporal context directly):

| Feature | Description |
|---|---|
| `delay_s` | Current stop delay in seconds (positive = late) |
| `speed` | Approximate bus speed (m/s) |
| `hour_sin`, `hour_cos` | Cyclical hour-of-day encoding |
| `day_sin`, `day_cos` | Cyclical day-of-week encoding |
| `rush_hour` | Binary flag (08:00–09:00 or 17:00–18:00) |
| `DistanceFromStop` | Distance to next stop (metres) |
| `stop_idx_norm` | Normalised position in route (0 = first stop, 1 = last) |
| `line_mean_delay` | Mean delay for this bus line (computed from training data) |

**Random Forest** (12 features — adds lag features to emulate temporal context):

Same as above minus `stop_idx_norm` and `line_mean_delay`, plus `delay_lag1–3` and `speed_lag1–3`.

---

## Outputs

```
outputs/<job-id>/
  best_rnn.pt              # Best RNN checkpoint (lowest val MAE)
  rnn_scaler.joblib        # Feature scaler (needed for inference)
  rf_model.joblib          # Serialised Random Forest
  rnn_training_curves.png  # Train/val MAE and RMSE per epoch
  rnn_pred_vs_actual.png   # Predicted vs actual scatter plot
  rnn_residuals.png        # Residual distribution
  results.json             # Final test MAE/RMSE for each model
```

---

## Results

All results use a **temporal split**: first 2/3 of each month → train, next 1/6 → val, last 1/6 → test.

| Model | Features | Test MAE | Test RMSE | vs. no-change baseline |
|---|---|---|---|---|
| No-change baseline | — | 187.42 s | 304.73 s | — |
| Random Forest | 12 | 194.67 s* | 335.04 s* | −3.9% (worse) |
| LSTM | 10 | **153.74 s** | **265.53 s** | **−18.0%** |

\* RF val-set result from an incomplete run (timed out before test evaluation).
