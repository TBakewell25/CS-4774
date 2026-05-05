"""
Two-layer LSTM with hidden_size=128

Architecture
    i_t  = sigma(W_ii * x_t + W_hi * h_{t-1} + b_i)    input gate
    f_t  = sigma(W_if * x_t + W_hf * h_{t-1} + b_f)    forget gate
    g_t  = tanh(W_ig  * x_t + W_hg * h_{t-1} + b_g)    cell gate
    o_t  = sigma(W_io * x_t + W_ho * h_{t-1} + b_o)    output gate
    c_t  = f_t (x) c_{t-1} + i_t (x) g_t    cell state
    h_t  = o_t (x) tanh(c_t)    hidden state
    y_hat = W_o . h_t + b     linear regression head

Loss: MAE
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


# Dataset 

class TripDataset(Dataset):
    # Wraps padded tensors and masks for DataLoader

    def __init__(self, X: np.ndarray, y: np.ndarray, masks: np.ndarray):
        self.X     = torch.from_numpy(X)
        self.y     = torch.from_numpy(y)
        self.masks = torch.from_numpy(masks)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks[idx]

class BusDelayRNN(nn.Module):
    
    # LSTM
    def __init__(
        self,
        input_size:  int,
        hidden_size: int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.rnn = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        zeros = lambda: torch.zeros(
            self.num_layers, x.size(0), self.hidden_size,
            device=x.device, dtype=x.dtype
        )
        if h0 is None:
            h0 = zeros()
        if c0 is None:
            c0 = zeros()

        out, (h_n, c_n) = self.rnn(x, (h0, c0))
        preds = self.output_layer(out).squeeze(-1)
        return preds, (h_n, c_n)


class RNNTrainer:
    # Full training loop

    def __init__(
        self,
        model:         BusDelayRNN,
        learning_rate: float = 1e-3,
        clip_grad:     float = 1.0,
        device:        str   = "cpu",
    ):
        self.model    = model.to(device)
        self.device   = device
        self.clip_grad= clip_grad
        self.criterion= nn.L1Loss(reduction="none")   # MAE, element-wise
        self.optimizer= torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        self.history = {"train_mae": [], "val_mae": [], "train_rmse": [], "val_rmse": []}

    # Single epoch
    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        self.model.train(train)
        total_mae = 0.0
        total_mse = 0.0
        n_elements = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for X_batch, y_batch, mask_batch in loader:
                X_batch    = X_batch.to(self.device)
                y_batch    = y_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)

                preds, _ = self.model(X_batch)

                # Mask away padded positions before computing loss
                mae_elements = self.criterion(preds, y_batch)
                masked_mae = mae_elements * mask_batch.float()
                loss = masked_mae.sum() / mask_batch.float().sum()

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    self.optimizer.step()

                # Accumulate metrics on non-padded positions
                valid_preds = preds[mask_batch]
                valid_y = y_batch[mask_batch]
                n  = valid_y.numel()
                total_mae += (valid_preds - valid_y).abs().sum().item()
                total_mse += ((valid_preds - valid_y) ** 2).sum().item()
                n_elements += n

        mae  = total_mae / max(n_elements, 1)
        rmse = (total_mse / max(n_elements, 1)) ** 0.5
        return mae, rmse

    # Full training loop 
    def fit(
        self,
        train_loader:  DataLoader,
        val_loader:    DataLoader,
        max_epochs:    int   = 50,
        patience:      int   = 7,
        save_path:     str   = "best_rnn.pt",
    ) -> "RNNTrainer":
        # Train until convergence or max_epochs, using early stopping on val MAE
        best_val_mae   = float("inf")
        epochs_no_impr = 0

        for epoch in range(1, max_epochs + 1):
            tr_mae,  tr_rmse  = self._run_epoch(train_loader, train=True)
            val_mae, val_rmse = self._run_epoch(val_loader,   train=False)
            self.scheduler.step(val_mae)

            self.history["train_mae"].append(tr_mae)
            self.history["val_mae"].append(val_mae)
            self.history["train_rmse"].append(tr_rmse)
            self.history["val_rmse"].append(val_rmse)

            print(
                f"Epoch {epoch:3d} | "
                f"Train MAE {tr_mae:8.2f}s  RMSE {tr_rmse:8.2f}s | "
                f"Val   MAE {val_mae:8.2f}s  RMSE {val_rmse:8.2f}s"
            )

            if val_mae < best_val_mae:
                best_val_mae   = val_mae
                epochs_no_impr = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"  New best ({best_val_mae:.2f}s) - checkpoint saved.")
            else:
                epochs_no_impr += 1
                if epochs_no_impr >= patience:
                    print(f"Early stopping after {patience} epochs without improvement.")
                    break

        # Restore best weights
        self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        return self

    # Evaluation
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        #Return (MAE, RMSE) on an arbitrary split
        mae, rmse = self._run_epoch(loader, train=False)
        print(f"Evaluation → MAE: {mae:.2f}s   RMSE: {rmse:.2f}s")
        return mae, rmse

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        squeeze = X.ndim == 2
        if squeeze:
            X = X[np.newaxis]
        tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        preds, _ = self.model(tensor)
        out = preds.cpu().numpy()
        return out[0] if squeeze else out
