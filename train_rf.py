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

# ── columns ──────────────────────────────────────────────────────────────────

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
]

FEATURE_COLS = BASE_FEATURE_COLS + [
    "stop_mean_delay",
    "line_mean_delay",
    "stop_idx_norm",
]

CYCLE_GROUP = ["PublishedLineName", "DirectionRef", "VehicleRef", "CycleNumber"]

DELAY_CLIP = 1800   # 30 min delay in seconds

# ── helpers ───────────────────────────────────────────────────────────────────

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


def load_and_prepare(paths: list[str], max_rows: int | None = None) -> pd.DataFrame:
    """Load parsed CSVs, compute Δdelay target and stop_idx_norm."""
    rows_per_file = None
    if max_rows is not None:
        rows_per_file = max_rows // len(paths)

    load_cols = list(set(
        BASE_FEATURE_COLS + CYCLE_GROUP + ["Date", "NextStopPointName", "RecordedAtTime"]
    ))

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

        df = df.sort_values(CYCLE_GROUP)

        df["stop_idx"] = df.groupby(CYCLE_GROUP).cumcount()
        max_idx = (df.groupby(CYCLE_GROUP)["stop_idx"]
                     .transform("max")
                     .clip(lower=1))
        df["stop_idx_norm"] = df["stop_idx"] / max_idx

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
    parser.add_argument("--csv", nargs="+", required=True,
                        help="Paths to parsed MTA CSV files")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--n-iter",   type=int, default=10)
    parser.add_argument("--out-dir",  default=".")
    args = parser.parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    print("\n[1/3] Loading transit data …")
    df = load_and_prepare(args.csv, max_rows=args.max_rows)

    # ── split ────────────────────────────────────────────────────────────────
    print("\n[2/3] Splitting data …")
    train_df, val_df, test_df = temporal_split(df)
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    print("\n  Baseline (predict no change in delay):")
    scores(test_df["target"].values, np.zeros(len(test_df)), "no-change")

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
    print(f"\n[3/3] Random search over {args.n_iter} trials …")
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
    print("\nFinal evaluation …")
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
