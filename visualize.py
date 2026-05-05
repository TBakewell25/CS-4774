from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg") # headless for server/SLURM
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed - plots will be skipped.")


def plot_training_curves(history: dict, save_to: str | Path = "training_curves.png"):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("mae", "rmse")):
        ax.plot(history[f"train_{metric}"], label="train")
        ax.plot(history[f"val_{metric}"],   label="val")
        ax.set_title(metric.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Seconds")
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_to, dpi=120)
    plt.close(fig)
    print(f"Training curves saved -> {save_to}")

def plot_predictions(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    title:   str = "Predictions vs Ground Truth",
    save_to: str | Path = "predictions.png",
):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Time-series view
    ax = axes[0]
    ax.plot(y_true, label="actual",    marker="o", linewidth=1.5)
    ax.plot(y_pred, label="predicted", marker="x", linewidth=1.5, linestyle="--")
    ax.set_title(f"{title} - sequence")
    ax.set_xlabel("Stop index")
    ax.set_ylabel("Latency (s)")
    ax.legend()
    ax.grid(True)

    # Scatter view
    ax = axes[1]
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.scatter(y_true, y_pred, alpha=0.5, s=15)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_title(f"{title} - scatter")
    ax.set_xlabel("Actual (s)")
    ax.set_ylabel("Predicted (s)")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_to, dpi=120)
    plt.close(fig)
    print(f"Prediction plot saved -> {save_to}")

def plot_pred_vs_actual(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    save_to: str | Path = "pred_vs_actual.png",
):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(y_true, y_pred, gridsize=80, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Count")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "b--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("Actual delay (s)")
    ax.set_ylabel("Predicted delay (s)")
    ax.set_title(f"Predicted vs Actual — Test Set  (n={len(y_true):,})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_to, dpi=120)
    plt.close(fig)
    print(f"Pred vs actual plot saved -> {save_to}")


def plot_residuals(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    save_to: str | Path = "residuals.png",
):
    if not HAS_MPL:
        return
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=80, color="steelblue", edgecolor="none", density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (predicted − actual delay, s)")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distribution — Test Set  (n={len(residuals):,})")
    fig.tight_layout()
    fig.savefig(save_to, dpi=120)
    plt.close(fig)
    print(f"Residual plot saved -> {save_to}")


def plot_feature_importances(
    importances: dict,
    top_n:  int  = 15,
    save_to: str | Path = "feature_importances.png",
):
    if not HAS_MPL:
        return
    items  = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names  = [i[0] for i in items]
    scores = [i[1] for i in items]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names[::-1], scores[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_to, dpi=120)
    plt.close(fig)
    print(f"Feature importance plot saved -> {save_to}")
