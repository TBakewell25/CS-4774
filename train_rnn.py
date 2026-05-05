"""
LSTM training for MTA bus delay prediction (proposal §6.5–6.8).

Architecture:
  - 2-layer LSTM with linear output head (scalar regression per stop)

Target: CHANGE in delay from current stop to the next (Δdelay, seconds).
  Positive = delay growing, negative = delay recovering.
  Full prediction at inference: delay_s + predicted_Δdelay.

Input: one sequence per route cycle, each step = one bus stop.
  - Features: current delay, speed, time encodings, rush-hour, distance,
              direction, hourly weather, stop/line mean-Δdelay encodings,
              stop_idx_norm.
  - Target: Δdelay at the *next* stop within the same cycle.

Usage:
  python train_rnn.py [--max-cycles N] [--hidden-dim H] [--lr LR]
                      [--epochs E] [--batch-size B] [--out-dir DIR]
                      [--weather PATH]
"""

import argparse
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

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

# ── constants ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "delay_s",
    "speed",
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

CYCLE_GROUP = ["PublishedLineName", "DirectionRef", "VehicleRef", "CycleNumber"]

DELAY_CLIP = 1800   # ±30 min — clip before computing delta


# ── data helpers ──────────────────────────────────────────────────────────────

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


def temporal_split_dates(dates: np.ndarray):
    dates = np.sort(np.unique(dates))
    n = len(dates)
    return (
        set(dates[:int(n * 2 / 3)]),
        set(dates[int(n * 2 / 3):int(n * 5 / 6)]),
        set(dates[int(n * 5 / 6):]),
    )


def load_cycles(paths: list[str], max_cycles: int | None = None,
                weather: pd.DataFrame | None = None) -> list[tuple]:
    """Load parsed CSVs and return a list of 5-tuples per route cycle:
      (date, X, y, stop_names, line_name)

    X : float32 (L-1, k) — features at stops 0..L-2
    y : float32 (L-1,)   — Δdelay at stops 1..L-1  (clip(delay[k+1]) - clip(delay[k]))
    """
    weather_cols = ["temperature_c", "precipitation_mm", "rain_mm", "windspeed_kmh"]
    needed = list(set(
        FEATURE_COLS + CYCLE_GROUP
        + ["Date", "RecordedAtTime", "NextStopPointName"]
    ))
    # Weather cols not yet in CSV; remove so read_csv doesn't error.
    needed = [c for c in needed if c not in weather_cols]

    rows_per_file = (max_cycles * 43 * 10 // len(paths)
                     if max_cycles is not None else None)

    all_dfs = []
    for p in paths:
        print(f"  Loading {os.path.basename(p)} …")
        df = pd.read_csv(p, usecols=needed, low_memory=False,
                         nrows=rows_per_file)
        all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"  {len(df):,} rows loaded")

    df["RecordedAtTime"] = pd.to_datetime(df["RecordedAtTime"])
    df = df.sort_values(CYCLE_GROUP + ["RecordedAtTime"])

    # Weather join on (Date, hour).
    if weather is not None:
        df["hour"] = df["RecordedAtTime"].dt.hour
        df = df.merge(weather, on=["Date", "hour"], how="left")
        df.drop(columns=["hour"], inplace=True)
        n_nan = df[weather_cols].isna().any(axis=1).sum()
        if n_nan:
            print(f"  Warning: {n_nan:,} rows missing weather (filling 0)")
            df[weather_cols] = df[weather_cols].fillna(0)
    else:
        for c in weather_cols:
            df[c] = 0.0

    df = df.dropna(subset=FEATURE_COLS)
    print(f"  {len(df):,} rows after dropping NaN")

    cycles = []
    for keys, grp in df.groupby(CYCLE_GROUP, sort=False):
        if len(grp) < 2:
            continue
        feats      = grp[FEATURE_COLS].values.astype(np.float32)   # (L, k)
        delay      = grp["delay_s"].values.astype(np.float32)      # (L,)
        date       = grp["Date"].iloc[0]
        stop_names = grp["NextStopPointName"].values                # (L,)
        line_name  = keys[0]                                        # PublishedLineName

        # Δdelay target: change in clipped delay from one stop to the next.
        clipped = np.clip(delay, -DELAY_CLIP, DELAY_CLIP)
        target  = clipped[1:] - clipped[:-1]

        cycles.append((date, feats[:-1], target, stop_names[:-1], line_name))

    print(f"  {len(cycles):,} cycles built")

    if max_cycles is not None and len(cycles) > max_cycles:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(cycles), size=max_cycles, replace=False)
        cycles = [cycles[i] for i in idx]
        print(f"  Subsampled to {max_cycles:,} cycles")

    return cycles


def split_cycles(cycles: list[tuple]):
    all_dates = np.array([c[0] for c in cycles])
    train_d, val_d, test_d = temporal_split_dates(all_dates)
    train = [c for c in cycles if c[0] in train_d]
    val   = [c for c in cycles if c[0] in val_d]
    test  = [c for c in cycles if c[0] in test_d]
    return train, val, test


def compute_stop_line_means(train_cycles: list[tuple]):
    """Mean Δdelay per stop name and per line from training cycles."""
    stop_sum = defaultdict(float)
    stop_cnt = defaultdict(int)
    line_sum = defaultdict(float)
    line_cnt = defaultdict(int)

    for _, _, targets, stop_names, line_name in train_cycles:
        for t, s in zip(targets, stop_names):
            stop_sum[s] += float(t)
            stop_cnt[s] += 1
        line_sum[line_name] += float(targets.sum())
        line_cnt[line_name] += len(targets)

    stop_means = {s: stop_sum[s] / stop_cnt[s] for s in stop_sum}
    line_means = {l: line_sum[l] / line_cnt[l] for l in line_sum}
    global_mean = sum(stop_sum.values()) / max(sum(stop_cnt.values()), 1)
    return stop_means, line_means, global_mean


def add_encoded_features(cycles: list[tuple], stop_means: dict,
                         line_means: dict, global_mean: float) -> list[tuple]:
    """Append stop_mean_delay, line_mean_delay, stop_idx_norm.
    Converts 5-tuples to 3-tuples."""
    new_cycles = []
    for date, feats, targets, stop_names, line_name in cycles:
        L = len(feats)
        stop_enc = np.array([stop_means.get(s, global_mean) for s in stop_names],
                            dtype=np.float32).reshape(-1, 1)
        line_enc = np.full((L, 1), line_means.get(line_name, global_mean),
                           dtype=np.float32)
        stop_idx_norm = np.linspace(0.0, 1.0, L, dtype=np.float32).reshape(-1, 1)
        feats_new = np.concatenate([feats, stop_enc, line_enc, stop_idx_norm], axis=1)
        new_cycles.append((date, feats_new, targets))
    return new_cycles


def fit_scaler(cycles: list[tuple]) -> StandardScaler:
    X = np.concatenate([c[1] for c in cycles], axis=0)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def apply_scaler(cycles: list[tuple], scaler: StandardScaler) -> list[tuple]:
    return [(date, scaler.transform(X).astype(np.float32), y)
            for date, X, y in cycles]


# ── dataset / dataloader ──────────────────────────────────────────────────────

class CycleDataset(Dataset):
    def __init__(self, cycles: list[tuple]):
        self.data = [(torch.from_numpy(X), torch.from_numpy(y))
                     for _, X, y in cycles]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    xs_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys_pad = nn.utils.rnn.pad_sequence(ys, batch_first=True)
    return xs_pad, ys_pad, lengths


# ── model ─────────────────────────────────────────────────────────────────────

class BusDelayRNN(nn.Module):
    """Two-layer LSTM with linear output head."""

    def __init__(self, input_size: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x_pad, lengths):
        packed = pack_padded_sequence(x_pad, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        out_pad, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return self.output(out_pad).squeeze(-1)   # (B, T_max)


# ── training helpers ──────────────────────────────────────────────────────────

def masked_mae(preds, targets, lengths):
    total_loss = 0.0
    total_n = 0
    for i, L in enumerate(lengths):
        total_loss += torch.abs(preds[i, :L] - targets[i, :L]).sum()
        total_n += L
    return total_loss / total_n


@torch.no_grad()
def evaluate_v2(model, loader, device):
    model.eval()
    sum_abs = torch.tensor(0.0)
    sum_sq  = torch.tensor(0.0)
    n_total = 0
    for x, y, lengths in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x, lengths)
        for i, L in enumerate(lengths.tolist()):
            diff = preds[i, :L] - y[i, :L]
            sum_abs += diff.abs().sum().cpu()
            sum_sq  += (diff ** 2).sum().cpu()
            n_total += L
    return (sum_abs / n_total).item(), torch.sqrt(sum_sq / n_total).item()


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for x, y, lengths in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x, lengths)
        for i, L in enumerate(lengths.tolist()):
            all_preds.append(preds[i, :L].cpu().numpy())
            all_targets.append(y[i, :L].cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def plot_results(history, y_test, y_pred_test, out_dir: str):
    epochs = range(1, len(history["train_mae"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_mae"], label="train MAE", linewidth=1.5)
    ax.plot(epochs, history["val_mae"],   label="val MAE",   linewidth=1.5)
    best_epoch = int(np.argmin(history["val_mae"])) + 1
    ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1,
               label=f"best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (s)")
    ax.set_title("LSTM Training — Learning Curve (Δdelay target)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curve.png"), dpi=150)
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
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--weather", default=WEATHER_PATH,
                        help="Path to hourly weather CSV.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── load weather ──────────────────────────────────────────────────────────
    print("\n[0/6] Loading weather …")
    weather = load_weather(args.weather)
    if weather is not None:
        print(f"  {len(weather):,} hourly weather records loaded.")

    # ── load cycles ───────────────────────────────────────────────────────────
    print("\n[1/6] Loading data …")
    paths  = [os.path.join(PARSED_DIR, f) for f in PARSED_FILES]
    cycles = load_cycles(paths, max_cycles=args.max_cycles, weather=weather)

    # ── split ────────────────────────────────────────────────────────────────
    print("\n[2/6] Splitting …")
    train_c5, val_c5, test_c5 = split_cycles(cycles)
    print(f"  Train: {len(train_c5):,} cycles  "
          f"Val: {len(val_c5):,}  Test: {len(test_c5):,}")

    # ── persistence baseline: predict Δdelay = 0 (no change) ─────────────────
    persist_targets = np.concatenate([c[2] for c in test_c5])
    persist_mae  = float(np.mean(np.abs(persist_targets)))
    persist_rmse = float(np.sqrt(np.mean(persist_targets ** 2)))
    print(f"\n  No-change baseline (test):  "
          f"MAE={persist_mae:.2f}s  RMSE={persist_rmse:.2f}s")

    # ── stop/line mean encodings ──────────────────────────────────────────────
    print("\n[3/6] Computing stop/line encodings …")
    stop_means, line_means, global_mean = compute_stop_line_means(train_c5)
    train_c = add_encoded_features(train_c5, stop_means, line_means, global_mean)
    val_c   = add_encoded_features(val_c5,   stop_means, line_means, global_mean)
    test_c  = add_encoded_features(test_c5,  stop_means, line_means, global_mean)
    k = train_c[0][1].shape[1]
    print(f"  Feature dim: {len(FEATURE_COLS)} base + 3 encoded = {k} total")

    # ── normalize ────────────────────────────────────────────────────────────
    print("\n[4/6] Fitting scaler on train …")
    scaler  = fit_scaler(train_c)
    train_c = apply_scaler(train_c, scaler)
    val_c   = apply_scaler(val_c,   scaler)
    test_c  = apply_scaler(test_c,  scaler)

    train_loader = DataLoader(CycleDataset(train_c), batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(CycleDataset(val_c),   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(CycleDataset(test_c),  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ── model ────────────────────────────────────────────────────────────────
    print("\n[5/6] Training …")
    model = BusDelayRNN(input_size=k, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_val_mae = float("inf")
    patience_ctr = 0
    ckpt_path    = os.path.join(args.out_dir, "rnn_best.pt")
    history      = {"train_mae": [], "val_mae": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for x, y, lengths in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x, lengths)
            loss  = masked_mae(preds, y, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        train_mae_epoch = epoch_loss / len(train_loader)
        val_mae, val_rmse = evaluate_v2(model, val_loader, device)
        elapsed = time.time() - t0

        history["train_mae"].append(train_mae_epoch)
        history["val_mae"].append(val_mae)

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train_mae={train_mae_epoch:.2f}s  "
              f"val_mae={val_mae:.2f}s  val_rmse={val_rmse:.2f}s  "
              f"({elapsed:.1f}s)")

        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs).")
                break

    # ── final evaluation ─────────────────────────────────────────────────────
    print("\n[6/6] Final evaluation (best checkpoint) …")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    train_mae, train_rmse = evaluate_v2(model, train_loader, device)
    val_mae,   val_rmse   = evaluate_v2(model, val_loader,   device)
    test_mae,  test_rmse  = evaluate_v2(model, test_loader,  device)

    print(f"  {'Split':<12}  {'MAE':>10}  {'RMSE':>10}")
    print(f"  {'no-change':<12}  {persist_mae:>10.2f}s  {persist_rmse:>10.2f}s")
    print(f"  {'train':<12}  {train_mae:>10.2f}s  {train_rmse:>10.2f}s")
    print(f"  {'val':<12}  {val_mae:>10.2f}s  {val_rmse:>10.2f}s")
    print(f"  {'test':<12}  {test_mae:>10.2f}s  {test_rmse:>10.2f}s")
    print(f"\n  Checkpoint: {ckpt_path}")
    print(f"  hidden_dim={args.hidden_dim}  lr={args.lr}  "
          f"batch_size={args.batch_size}  features={k}")

    print("\n[+] Generating plots …")
    y_pred_test, y_test_flat = collect_preds(model, test_loader, device)
    plot_results(history, y_test_flat, y_pred_test, args.out_dir)
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test_flat)
    np.save(os.path.join(args.out_dir, "y_pred.npy"), y_pred_test)
    print(f"  Predictions saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
