# Train both models (default)
# python train.py --csv data1.csv data2.csv data3.csv data4.csv

# RNN only with custom hyperparameters
# python train.py --csv *.csv --model rnn --hidden 256 --epochs 60 --lr 0.0005

# RF only with random-search tuning
# python train.py --csv *.csv --model rf --rf-search 30

# Cache processed trips so re-runs skip loading
# python train.py --csv *.csv --cache-dir trips_cache

import argparse
import joblib
import os
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from data_pipeline import MTADataPipeline, RNN_FEATURE_COLS, RF_FEATURE_COLS
from rnn_model import BusDelayRNN, RNNTrainer, TripDataset
from random_forest_model import BusDelayRandomForest
from visualize import plot_training_curves, plot_predictions, plot_pred_vs_actual, plot_residuals

# CLI 
def parse_args():
    p = argparse.ArgumentParser(description="MTA Bus Delay Prediction")
    p.add_argument("--csv",        nargs="+", required=True,
                   help="Paths to MTA SIRI CSV files")
    p.add_argument("--model",      choices=["both", "rnn", "rf"], default="both")
    p.add_argument("--cache-dir",  default=None,
                   help="Directory to cache processed trips (skip reloading)")
    p.add_argument("--output-dir", default="outputs")
    # RNN hyperparameters
    p.add_argument("--hidden",     type=int,   default=128)
    p.add_argument("--layers",     type=int,   default=2)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch",      type=int,   default=64)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--patience",   type=int,   default=7)
    # RF hyperparameters
    p.add_argument("--rf-search",      type=int,   default=20,
                   help="Number of random-search iterations for RF")
    p.add_argument("--rf-max-samples", type=float, default=0.5,
                   help="Fraction of training rows each tree bootstraps (limits per-tree memory)")
    # Trip filtering
    p.add_argument("--min-trip-len", type=int, default=10,
                   help="Drop trips shorter than this many stops (removes truncated end-of-file sequences)")
    # Device
    p.add_argument("--device",     default=None,
                   help="'cpu' | 'cuda' | 'mps'  (auto-detected if not set)")
    return p.parse_args()

# Device detection 
def detect_device(requested):
    import torch
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"

# Data preparation 
def prepare_data(args):
    cache_file = Path(args.cache_dir) / "splits.pkl" if args.cache_dir else None

    if cache_file and cache_file.exists():
        print(f"Loading cached splits from {cache_file} ...")
        try:
            return joblib.load(cache_file)
        except Exception as e:
            print(f"  WARNING: cache corrupt ({type(e).__name__}: {e}), deleting and rebuilding ...")
            cache_file.unlink()

    pipeline = (
        MTADataPipeline(args.csv)
        .load()
        .reconstruct_trips(min_len=args.min_trip_len)
    )
    splits = pipeline.temporal_split()

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        tmp_file = cache_file.with_suffix(".pkl.tmp")
        joblib.dump(splits, tmp_file)
        tmp_file.rename(cache_file)  # atomic: never leaves a half-written pkl visible
        print(f"Cached splits -> {cache_file}")

    return splits

# RNN training
def train_rnn(args, train_trips, val_trips, test_trips, out_dir, *, _free_trips=True):
    print("\n" + "="*60)
    print(" TRAINING RNN")
    print("="*60)

    train_max_len = int(np.percentile([len(t) for t in train_trips], 99))
    X_tr, y_tr, m_tr = MTADataPipeline.trips_to_rnn_tensors(train_trips, RNN_FEATURE_COLS, max_len=train_max_len)
    X_v,  y_v,  m_v  = MTADataPipeline.trips_to_rnn_tensors(val_trips,   RNN_FEATURE_COLS, max_len=train_max_len)
    X_te, y_te, m_te = MTADataPipeline.trips_to_rnn_tensors(test_trips,  RNN_FEATURE_COLS, max_len=train_max_len)
    if _free_trips:
        train_trips.clear(); val_trips.clear(); test_trips.clear()

    print(f"Tensor shapes  X: {X_tr.shape}  y: {y_tr.shape}")

    # Fit scaler on valid (non-padded) train timesteps only, apply to all splits
    k = X_tr.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_tr[m_tr])
    # Transform in-place to avoid allocating a second full-size array per split
    for arr in (X_tr, X_v, X_te):
        flat = arr.reshape(-1, k)
        scaler.transform(flat, copy=False)  # writes into flat, which aliases arr
    joblib.dump(scaler, str(out_dir / "rnn_scaler.joblib"))
    print(f"Scaler fitted and saved -> {out_dir}/rnn_scaler.joblib")

    train_loader = DataLoader(TripDataset(X_tr, y_tr, m_tr),
                              batch_size=args.batch, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(TripDataset(X_v,  y_v,  m_v),
                              batch_size=args.batch, shuffle=False, num_workers=2)
    test_loader  = DataLoader(TripDataset(X_te, y_te, m_te),
                              batch_size=args.batch, shuffle=False, num_workers=2)

    device = detect_device(args.device)
    print(f"Device: {device}")

    model = BusDelayRNN(
        input_size  = X_tr.shape[2],
        hidden_size = args.hidden,
        num_layers  = args.layers,
        dropout     = args.dropout,
    )
    trainer = RNNTrainer(model, learning_rate=args.lr, device=device)

    save_path = str(out_dir / "best_rnn.pt")
    trainer.fit(train_loader, val_loader,
                max_epochs=args.epochs, patience=args.patience,
                save_path=save_path)

    print("\n-- Test Set --")
    mae, rmse = trainer.evaluate(test_loader)

    plot_training_curves(trainer.history, out_dir / "rnn_training_curves.png")

    # Collect all test predictions for residual plot and saved arrays
    preds_all   = trainer.predict(X_te)        # (N, T)
    y_pred_flat = preds_all[m_te]              # valid timesteps only
    y_true_flat = y_te[m_te]
    plot_pred_vs_actual(y_true_flat, y_pred_flat, out_dir / "rnn_pred_vs_actual.png")
    plot_residuals(y_true_flat, y_pred_flat,      out_dir / "rnn_residuals.png")
    np.save(out_dir / "rnn_y_test.npy",  y_true_flat)
    np.save(out_dir / "rnn_y_pred.npy",  y_pred_flat)
    print(f"Predictions saved -> {out_dir}/rnn_y_{{test,pred}}.npy")

    X_sample, y_sample, m_sample = next(iter(test_loader))
    preds_np = trainer.predict(X_sample.numpy())
    plot_predictions(
        y_true  = y_sample[0][m_sample[0]].numpy(),
        y_pred  = preds_np[0][m_sample[0].numpy()],
        title   = "RNN - sample trip prediction",
        save_to = out_dir / "rnn_sample_prediction.png",
    )
    return mae, rmse


# RF training
def train_rf(args, train_trips, val_trips, test_trips, out_dir):
    print("\n" + "="*60)
    print(" TRAINING RANDOM FOREST")
    print("="*60)

    X_tr, y_tr = MTADataPipeline.trips_to_rf_features(train_trips, RF_FEATURE_COLS)
    train_trips.clear()
    X_v,  y_v  = MTADataPipeline.trips_to_rf_features(val_trips,   RF_FEATURE_COLS)
    val_trips.clear()

    print(f"RF feature matrix  X: {X_tr.shape}  y: {y_tr.shape}")

    best_rf = BusDelayRandomForest.random_search(
        X_tr, y_tr, X_v, y_v, n_iter=args.rf_search, max_samples=args.rf_max_samples
    )
    del X_v, y_v
    best_rf.feature_names = RF_FEATURE_COLS
    best_rf.feature_importances()

    X_te, y_te = MTADataPipeline.trips_to_rf_features(test_trips, RF_FEATURE_COLS)
    test_trips.clear()

    print("\n-- Test Set --")
    results = best_rf.evaluate(X_te, y_te, split_name="test")
    best_rf.save(str(out_dir / "rf_model.joblib"))
    return results["mae"], results["rmse"]

# Entry point
def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_trips, val_trips, test_trips = prepare_data(args)
    train_trips, val_trips, test_trips = MTADataPipeline.encode_line_stats(
        train_trips, val_trips, test_trips
    )

    results = {}
    if args.model in ("both", "rnn"):
        # Free trips inside train_rnn only when RF won't need them afterward
        rnn_mae, rnn_rmse = train_rnn(args, train_trips, val_trips, test_trips, out_dir,
                                      _free_trips=(args.model == "rnn"))
        results["rnn"] = {"mae": rnn_mae, "rmse": rnn_rmse}

    if args.model in ("both", "rf"):
        rf_mae, rf_rmse = train_rf(args, train_trips, val_trips, test_trips, out_dir)
        results["rf"] = {"mae": rf_mae, "rmse": rf_rmse}

    del train_trips, val_trips, test_trips

    print("\n" + "="*60)
    print(" FINAL RESULTS")
    print("="*60)
    for name, scores in results.items():
        print(f"  {name.upper():4s}  MAE: {scores['mae']:.2f}s   RMSE: {scores['rmse']:.2f}s")

    import json
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
