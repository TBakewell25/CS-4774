# rMet: Transit Latency Prediction

**Thomas Bakewell · Ethan Klose · David Onks IV** — CS 4774, University of Virginia

Predicts the **change in bus delay** (Δdelay, seconds) from one stop to the next for NYC MTA buses using a two-layer LSTM. Trained on the [MTA SIRI dataset](https://www.kaggle.com/datasets/stoney71/new-york-city-transport-statistics) (~26 M stop records across June, August, October, and December 2017).

The best run achieved a test MAE of **153.74 s**, beating the no-change baseline (187.42 s) by 18%.

---

## Repository layout

```
.
├── sort_cycles.py        # Step 1 — sort raw CSVs into route cycles
├── engineer_features.py  # Step 2 — compute delay_s, speed, time encodings, lag features
├── train_rnn.py          # LSTM training script
├── submit_rnn.sh         # SLURM job script
└── requirements.txt
```

---

## Data pipeline

Raw data is not included in this repo. Download the four monthly CSVs from Kaggle and run the two preprocessing steps:

```
data/
  transit/        ← place raw CSVs here (mta_1706.csv … mta_1712.csv)
  sorted/         ← produced by sort_cycles.py
  parsed/         ← produced by engineer_features.py  (used for training)
```

**Step 1 — sort into route cycles:**
```bash
python sort_cycles.py data/transit/mta_1706.csv data/sorted/mta_1706_sorted.csv
# repeat for each month
```

**Step 2 — engineer features:**
```bash
# Without weather:
python engineer_features.py data/sorted/mta_1706_sorted.csv data/parsed/mta_1706_parsed.csv

# With weather (NYC_Weather_2016_2022.csv, also on Kaggle):
python engineer_features.py data/sorted/mta_1706_sorted.csv data/parsed/mta_1706_parsed.csv data/NYC_Weather_2016_2022.csv

# Repeat for each month.
```

---

## Training

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run locally:**
```bash
python train_rnn.py --csv data/parsed/mta_1706_parsed.csv \
                          data/parsed/mta_1708_parsed.csv \
                          data/parsed/mta_1710_parsed.csv \
                          data/parsed/mta_1712_parsed.csv

# With weather:
python train_rnn.py --csv data/parsed/*.csv --weather data/NYC_Weather_2016_2022.csv

# Subsample for a quick test run:
python train_rnn.py --csv data/parsed/*.csv --max-cycles 5000
```

**On SLURM (GPU partition):**
```bash
# Required: DATA_DIR points to the folder of parsed CSVs.
sbatch --export=ALL,DATA_DIR=/path/to/data/parsed submit_rnn.sh

# With weather:
sbatch --export=ALL,DATA_DIR=/path/to/data/parsed,WEATHER=/path/to/NYC_Weather_2016_2022.csv submit_rnn.sh
```

### Key CLI flags (`train_rnn.py`)

| Flag | Default | Description |
|---|---|---|
| `--csv` | required | Parsed CSV files (one or more) |
| `--weather` | none | Hourly weather CSV (optional) |
| `--hidden-dim` | `64` | LSTM hidden state size |
| `--lr` | `1e-3` | Adam learning rate |
| `--epochs` | `20` | Max training epochs |
| `--batch-size` | `64` | Batch size |
| `--patience` | `5` | Early stopping patience |
| `--max-cycles` | all | Subsample to N cycles (for testing) |
| `--out-dir` | `.` | Where to write outputs |

---

## Features

16 features total: 13 base + 3 encoded (computed from training data to avoid leakage).

| Feature | Description |
|---|---|
| `delay_s` | Current stop delay in seconds (positive = late) |
| `speed` | Approximate bus speed (m/s) |
| `hour_sin`, `hour_cos` | Cyclical hour-of-day encoding |
| `day_sin`, `day_cos` | Cyclical day-of-week encoding |
| `rush_hour` | Binary flag (08:00–09:00 or 17:00–18:00) |
| `DistanceFromStop` | Distance to next stop (metres) |
| `DirectionRef` | Route direction (encoded numerically) |
| `temperature_c` | Hourly temperature °C (requires weather CSV) |
| `precipitation_mm` | Hourly precipitation mm (requires weather CSV) |
| `rain_mm` | Hourly rainfall mm (requires weather CSV) |
| `windspeed_kmh` | Hourly wind speed km/h (requires weather CSV) |
| `stop_mean_Δdelay` | Mean Δdelay for this stop (from training set) |
| `line_mean_Δdelay` | Mean Δdelay for this bus line (from training set) |
| `stop_idx_norm` | Normalised position in route (0 = first stop, 1 = last) |

Without weather, the four weather columns are zero-filled and the model trains on 12 effective features.

---

## Target

```
Δdelay = clip(delay_s[k+1], ±30 min) − clip(delay_s[k], ±30 min)
```

Positive = delay growing, negative = delay recovering. The **no-change baseline** predicts Δdelay = 0 everywhere (i.e., delay stays constant stop-to-stop).

---

## Outputs

```
<out-dir>/
  rnn_best.pt           # Best checkpoint (lowest val MAE)
  learning_curve.png    # Train/val MAE per epoch
  pred_vs_actual.png    # Predicted vs actual scatter (test set)
  residuals.png         # Residual distribution (test set)
  y_test.npy            # Ground-truth Δdelay values
  y_pred.npy            # Model predictions
```

---

## Results

Temporal split: first 2/3 of each month → train, next 1/6 → val, last 1/6 → test.

| Model | Features | Test MAE | Test RMSE | vs. no-change baseline |
|---|---|---|---|---|
| No-change baseline | — | 187.42 s | 304.73 s | — |
| LSTM (no weather) | 12 | 163.83 s | 271.39 s | −12.6% |
| LSTM (with weather) | 16 | **153.74 s** | **265.53 s** | **−18.0%** |
