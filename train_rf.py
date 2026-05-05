"""
Random Forest training for MTA bus delay prediction.

Target: CHANGE in delay from current stop to the next stop (Δdelay, seconds).
  Positive = delay growing, negative = delay recovering.
  Full prediction at inference: delay_s + predicted_Δdelay.

Features: current + lagged delay/speed, cyclical time encodings, rush-hour flag,
          distance to next stop, direction, hourly weather, stop/line mean Δdelay
          encodings, normalized stop position within cycle.

Temporal split (per proposal §6.4):
  - First 2/3 of dates  -> train
  - Middle 1/6          -> val
  - Last 1/6            -> test

Usage:
  python train_rf.py [--max-rows N] [--n-iter K] [--out-dir DIR] [--weather PATH]
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ── paths ─────────────────────────────────────────────────────────────────────

PARSED_DIR   = os.path.join(os.path.dirname(__file__), "data", "parsed")
WEATHER_PATH = os.path.join(os.path.dirname(__file__), "data",
                            "NYC_Weather_2016_2022.csv")
PARSED_FILES = [
    "mta_1706_parsed.csv",
    "mta_1708_parsed.csv",
    "mta_1710_parsed.csv",
    "mta_1712_parsed.csv",
]

# ── columns ──────────────────────────────────────────────────────────────────

# Columns read directly from the parsed CSV.
BASE_FEATURE_COLS = [
    "delay_s",
    "delay_lag1", "delay_lag2", "delay_lag3",
    "speed",
    "speed_lag1", "speed_lag2", "speed_lag3",
    "hour_sin", "hour_cos",
    "day_sin",  "day_cos",
    "rush_hour",
    "DistanceFromStop",
    "DirectionRef",
    # weather (joined at load time)
    "temperature_c",
    "precipitation_mm",
    "rain_mm",
    "windspeed_kmh",
]

# Full model feature set (BASE + derived encodings added after split).
FEATURE_COLS = BASE_FEATURE_COLS + [
    "stop_mean_delay",   # mean Δdelay for this stop (from train)
    "line_mean_delay",   # mean Δdelay for this line  (from train)
    "stop_idx_norm",     # fractional position in cycle [0, 1)
]

CYCLE_GROUP = ["PublishedLineName", "DirectionRef", "VehicleRef", "CycleNumber"]

DELAY_CLIP = 1800   # ±30 min — clip before computing delta


# ── helpers ───────────────────────────────────────────────────────────────────

def load_weather(path: str) -> pd.DataFrame | None:
    """Load hourly weather CSV keyed on (Date str, hour int)."""
    if not os.path.exists(path):
        print(f"  Warning: weather file not found at {path} — skipping.")
        return None
    w = pd.read_csv(path, parse_dates=["time"])
    w["Date"] = w["time"].dt.strftime("%Y-%m-%d")
    w["hour"] = w["time"].dt.hour
    w = w.rename(columns={
        "temperature_2m (°C)":  "temperature_c",
        "precipitation (mm)":   "precipitation_mm",
        "rain (mm)":            "rain_mm",
        "windspeed_10m (km/h)": "windspeed_kmh",
    })
    return w[["Date", "hour", "temperature_c", "precipitation_mm",
              "rain_mm", "windspeed_kmh"]]


def temporal_split(df: pd.DataFrame):
    """Split rows by date: first 2/3 train, next 1/6 val, last 1/6 test."""
    dates = np.sort(df["Date"].unique())
    n = len(dates)
    train_dates = set(dates[:int(n * 2 / 3)])
    val_dates   = set(dates[int(n * 2 / 3):int(n * 5 / 6)])
    test_dates  = set(dates[int(n * 5 / 6):])

    train = df[df["Date"].isin(train_dates)].copy()
    val   = df[df["Date"].isin(val_dates)].copy()
    test  = df[df["Date"].isin(test_dates)].copy()
    return train, val, test


def load_and_prepare(paths: list[str], max_rows: int | None = None,
                     weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load parsed CSVs, join weather, compute Δdelay target and stop_idx_norm."""
    rows_per_file = None
    if max_rows is not None:
        rows_per_file = max_rows // len(paths)

    # Always load RecordedAtTime — needed for the weather hour join.
    load_cols = list(set(
        BASE_FEATURE_COLS + CYCLE_GROUP + ["Date", "NextStopPointName", "RecordedAtTime"]
    ))
    # Remove weather cols from load_cols — they're not in the CSV yet, added via join.
    weather_cols = ["temperature_c", "precipitation_mm", "rain_mm", "windspeed_kmh"]
    load_cols = [c for c in load_cols if c not in weather_cols]

    splits = []
    for p in paths:
        print(f"  Loading {os.path.basename(p)} …")

        if rows_per_file is not None:
            with open(p) as fh:
                total = sum(1 for _ in fh) - 1
            keep = max(rows_per_file, 1)
            if keep < total:
                skip_idx = np.sort(
                    np.random.default_rng(42).choice(total, size=total - keep,
                                                     replace=False) + 1
                )
                df = pd.read_csv(p, skiprows=skip_idx, usecols=load_cols,
                                 low_memory=False)
            else:
                df = pd.read_csv(p, usecols=load_cols, low_memory=False)
        else:
            df = pd.read_csv(p, usecols=load_cols, low_memory=False)

        print(f"    {len(df):,} rows read")

        # Sort (stable: within-cycle timestamp order preserved since the parsed
        # CSV was written in (cycle_group, RecordedAtTime) order).
        df = df.sort_values(CYCLE_GROUP)

        # Weather join on (Date, hour).
        if weather is not None:
            df["hour"] = pd.to_datetime(df["RecordedAtTime"]).dt.hour
            df = df.merge(weather, on=["Date", "hour"], how="left")
            df.drop(columns=["hour"], inplace=True)
            n_nan = df[weather_cols].isna().any(axis=1).sum()
            if n_nan:
                print(f"    Warning: {n_nan:,} rows missing weather (filling 0)")
                df[weather_cols] = df[weather_cols].fillna(0)
        else:
            for c in weather_cols:
                df[c] = 0.0

        # Fractional stop position within the cycle.
        df["stop_idx"] = df.groupby(CYCLE_GROUP).cumcount()
        max_idx = (df.groupby(CYCLE_GROUP)["stop_idx"]
                     .transform("max")
                     .clip(lower=1))
        df["stop_idx_norm"] = df["stop_idx"] / max_idx

        # Target: change in delay from current stop to the next (both clipped).
        next_delay = df.groupby(CYCLE_GROUP)["delay_s"].shift(-1).clip(-DELAY_CLIP, DELAY_CLIP)
        curr_delay = df["delay_s"].clip(-DELAY_CLIP, DELAY_CLIP)
        df["target"] = next_delay - curr_delay

        df = df.dropna(subset=["target"] + BASE_FEATURE_COLS)
        splits.append(df)

    df = pd.concat(splits, ignore_index=True)
    print(f"  Total rows after cleaning: {len(df):,}")
    return df


def encode_mean_targets(train_df: pd.DataFrame,
                        val_df:   pd.DataFrame,
                        test_df:  pd.DataFrame) -> None:
    """Add stop/line mean Δdelay columns (target-encoded from train only)."""
    global_mean = float(train_df["target"].mean())
    stop_mean = train_df.groupby("NextStopPointName")["target"].mean()
    line_mean = train_df.groupby("PublishedLineName")["target"].mean()

    for df in (train_df, val_df, test_df):
        df["stop_mean_delay"] = (df["NextStopPointName"]
                                   .map(stop_mean)
                                   .fillna(global_mean))
        df["line_mean_delay"] = (df["PublishedLineName"]
                                   .map(line_mean)
                                   .fillna(global_mean))


def scores(y_true, y_pred, label: str):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    print(f"  {label:<14s}  MAE={mae:.2f}s   RMSE={rmse:.2f}s")
    return mae, rmse


def plot_results(imp, trial_records, y_test, y_pred_test, out_dir: str):
    """Save four diagnostic plots to out_dir."""

    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — Feature Importances")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)

    trials_df = pd.DataFrame(trial_records)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trials_df["trial"], trials_df["val_mae"], marker="o", linewidth=1.5)
    ax.axhline(trials_df["val_mae"].min(), color="red", linestyle="--",
               linewidth=1, label=f"best={trials_df['val_mae'].min():.2f}s")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Val MAE (s)")
    ax.set_title("Hyperparameter Search — Val MAE per Trial")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "hp_search.png"), dpi=150)
    plt.close(fig)

    n_sample = min(5_000, len(y_test))
    rng = np.random.default_rng(1)
    idx = rng.choice(len(y_test), size=n_sample, replace=False)
    y_s, yp_s = y_test[idx], y_pred_test[idx]
    lims = (min(y_s.min(), yp_s.min()), max(y_s.max(), yp_s.max()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_s, yp_s, alpha=0.3, s=8, color="steelblue", rasterized=True)
    ax.plot(lims, lims, "r--", linewidth=1, label="perfect")
    ax.set_xlabel("Actual Δdelay (s)")
    ax.set_ylabel("Predicted Δdelay (s)")
    ax.set_title(f"Predicted vs Actual — Test Set (n={n_sample:,})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "pred_vs_actual.png"), dpi=150)
    plt.close(fig)

    residuals = y_pred_test - y_test
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=80, color="steelblue", edgecolor="none", density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (predicted − actual Δdelay, s)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution — Test Set")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "residuals.png"), dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {out_dir}/")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--weather", default=WEATHER_PATH,
                        help="Path to hourly weather CSV.")
    args = parser.parse_args()

    # ── load weather ──────────────────────────────────────────────────────────
    print("\n[0/4] Loading weather …")
    weather = load_weather(args.weather)
    if weather is not None:
        print(f"  {len(weather):,} hourly weather records loaded.")

    # ── load data ─────────────────────────────────────────────────────────────
    print("\n[1/4] Loading transit data …")
    paths = [os.path.join(PARSED_DIR, f) for f in PARSED_FILES]
    df = load_and_prepare(paths, max_rows=args.max_rows, weather=weather)

    # ── split ────────────────────────────────────────────────────────────────
    print("\n[2/4] Splitting data …")
    train_df, val_df, test_df = temporal_split(df)
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # ── persistence baseline: predict Δdelay = 0 (no change) ─────────────────
    print("\n  Baseline (predict no change in delay):")
    scores(test_df["target"].values, np.zeros(len(test_df)), "no-change")

    # ── stop/line mean encodings (from train only) ────────────────────────────
    print("\n  Computing stop/line mean-Δdelay encodings …")
    encode_mean_targets(train_df, val_df, test_df)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["target"].values
    X_val   = val_df[FEATURE_COLS].values
    y_val   = val_df["target"].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["target"].values

    print(f"  Features: {len(FEATURE_COLS)}  "
          f"(base={len(BASE_FEATURE_COLS)}, encoded=3)")

    # ── hyperparameter search ────────────────────────────────────────────────
    print(f"\n[3/4] Random search over {args.n_iter} trials …")
    rng = np.random.default_rng(0)

    n_estimators_grid     = [50, 100, 200, 300, 500]
    max_depth_grid        = [5, 10, 15, 20, 30, None]
    min_samples_leaf_grid = [1, 5, 10, 20]

    best_mae    = float("inf")
    best_params = {}
    best_model  = None
    trial_records = []

    for i in range(args.n_iter):
        n_est  = int(rng.choice(n_estimators_grid))
        depth  = rng.choice(max_depth_grid)
        min_sl = int(rng.choice(min_samples_leaf_grid))

        print(f"  Trial {i+1}/{args.n_iter}: n_estimators={n_est}, "
              f"max_depth={depth}, min_samples_leaf={min_sl} …",
              end=" ", flush=True)
        t0 = time.time()

        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=depth,
            min_samples_leaf=min_sl,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_train, y_train)

        val_mae = mean_absolute_error(y_val, rf.predict(X_val))
        elapsed = time.time() - t0
        print(f"val MAE={val_mae:.2f}s  ({elapsed:.1f}s)")
        trial_records.append({
            "trial": i + 1,
            "n_estimators": n_est,
            "max_depth": depth if depth is not None else "None",
            "min_samples_leaf": min_sl,
            "val_mae": val_mae,
        })

        if val_mae < best_mae:
            best_mae    = val_mae
            best_params = {"n_estimators": n_est, "max_depth": depth,
                           "min_samples_leaf": min_sl}
            best_model  = rf

    print(f"\n  Best params: {best_params}  (val MAE={best_mae:.2f}s)")

    # ── final evaluation ─────────────────────────────────────────────────────
    print("\n[4/4] Final evaluation …")
    scores(y_train, best_model.predict(X_train), "train")
    scores(y_val,   best_model.predict(X_val),   "val")
    scores(y_test,  best_model.predict(X_test),  "test")

    imp = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
    imp = imp.sort_values(ascending=False)
    print("\n  Feature importances:")
    for feat, val in imp.items():
        print(f"    {feat:<22s} {val:.4f}")

    out_path = os.path.join(args.out_dir, "rf_feature_importance.csv")
    imp.to_csv(out_path, header=["importance"])
    print(f"\n  Saved feature importances to {out_path}")
    print(f"  Best hyperparameters: {best_params}")

    print("\n[+] Generating plots …")
    y_pred_test = best_model.predict(X_test)
    plot_results(imp, trial_records, y_test, y_pred_test, args.out_dir)
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test)
    np.save(os.path.join(args.out_dir, "y_pred.npy"), y_pred_test)
    print(f"  Predictions saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
