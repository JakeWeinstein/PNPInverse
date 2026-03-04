"""Training loop and ensemble training for the NN surrogate model.

Features:
- AdamW optimizer with CosineAnnealingWarmRestarts scheduler
- Early stopping with configurable patience
- Checkpointing (best model + periodic snapshots)
- CSV logging: epoch, train_loss, val_loss, val_cd_rmse, val_pc_rmse, lr, elapsed_s
- Loss curve plotting
- Ensemble training (multiple models with different seeds and splits)
- Physics-informed regularization (monotonicity + smoothness penalties)
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from Surrogate.nn_model import (
    NNSurrogateModel,
    ResNetMLP,
    ZScoreNormalizer,
    _check_torch,
)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NNTrainingConfig:
    """Configuration for NN surrogate training.

    Attributes
    ----------
    epochs : int
        Maximum training epochs.
    lr : float
        Initial learning rate for AdamW.
    weight_decay : float
        AdamW weight decay.
    patience : int
        Early stopping patience (epochs without val improvement).
    batch_size : int or None
        Mini-batch size. None means full-batch training.
    checkpoint_interval : int
        Save checkpoint every N epochs.
    T_0 : int
        CosineAnnealingWarmRestarts T_0 parameter.
    T_mult : int
        CosineAnnealingWarmRestarts T_mult parameter.
    eta_min : float
        Minimum learning rate for cosine scheduler.
    hidden : int
        Hidden layer width.
    n_blocks : int
        Number of residual blocks.
    monotonicity_weight : float
        Weight for monotonicity penalty (0 = disabled).
    smoothness_weight : float
        Weight for smoothness penalty (0 = disabled).
    """

    epochs: int = 5000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 500
    batch_size: int | None = None
    checkpoint_interval: int = 500
    T_0: int = 500
    T_mult: int = 2
    eta_min: float = 1e-6
    hidden: int = 128
    n_blocks: int = 4
    monotonicity_weight: float = 0.0
    smoothness_weight: float = 0.0


# ---------------------------------------------------------------------------
# Physics-informed regularization
# ---------------------------------------------------------------------------

def _monotonicity_penalty(
    model: Any,
    X_batch: Any,
    n_eta: int,
    phi_applied: np.ndarray,
) -> Any:
    """Penalise non-monotonic CD predictions (should decrease with eta).

    Computes a ReLU penalty on positive differences in CD along the
    voltage dimension.  Only applied to the CD portion of the output.

    Parameters
    ----------
    model : ResNetMLP
        The neural network.
    X_batch : torch.Tensor (B, 4)
        Normalized input batch.
    n_eta : int
        Number of voltage points.
    phi_applied : np.ndarray (n_eta,)
        Voltage grid (used to determine sort order).

    Returns
    -------
    torch.Tensor
        Scalar penalty.
    """
    _check_torch()
    pred = model(X_batch)
    cd_pred = pred[:, :n_eta]  # CD portion

    # If voltage is ascending, CD should be decreasing (negative slope)
    # Penalise positive differences: cd[i+1] - cd[i] > 0
    diffs = cd_pred[:, 1:] - cd_pred[:, :-1]
    penalty = torch.relu(diffs).pow(2).mean()
    return penalty


def _smoothness_penalty(
    model: Any,
    X_batch: Any,
    n_eta: int,
) -> Any:
    """Penalise non-smooth outputs (second-order finite difference).

    Parameters
    ----------
    model : ResNetMLP
        The neural network.
    X_batch : torch.Tensor (B, 4)
        Normalized input batch.
    n_eta : int
        Number of voltage points.

    Returns
    -------
    torch.Tensor
        Scalar penalty.
    """
    _check_torch()
    pred = model(X_batch)
    # Second-order differences for both CD and PC
    d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    penalty = d2.pow(2).mean()
    return penalty


# ---------------------------------------------------------------------------
# Single model training with full logging
# ---------------------------------------------------------------------------

def train_nn_surrogate(
    parameters: np.ndarray,
    current_density: np.ndarray,
    peroxide_current: np.ndarray,
    phi_applied: np.ndarray,
    *,
    val_parameters: np.ndarray | None = None,
    val_cd: np.ndarray | None = None,
    val_pc: np.ndarray | None = None,
    config: NNTrainingConfig | None = None,
    seed: int = 0,
    output_dir: str | None = None,
    model_tag: str = "model",
    verbose: bool = True,
) -> Tuple[NNSurrogateModel, Dict[str, Any]]:
    """Train a single NN surrogate model with full logging and checkpointing.

    Parameters
    ----------
    parameters : np.ndarray (N, 4)
        Training parameters [k0_1, k0_2, alpha_1, alpha_2].
    current_density : np.ndarray (N, n_eta)
        Training CD curves.
    peroxide_current : np.ndarray (N, n_eta)
        Training PC curves.
    phi_applied : np.ndarray (n_eta,)
        Voltage grid.
    val_parameters, val_cd, val_pc : optional
        Validation data. If None, 15% split from training data.
    config : NNTrainingConfig or None
        Training configuration.
    seed : int
        Random seed.
    output_dir : str or None
        Directory for checkpoints, logs, and plots.
    model_tag : str
        Name tag for this model (used in filenames).
    verbose : bool
        Print progress.

    Returns
    -------
    (NNSurrogateModel, dict)
        Fitted model and training history dict.
    """
    _check_torch()

    if config is None:
        config = NNTrainingConfig()

    N = parameters.shape[0]
    n_eta = phi_applied.shape[0]
    n_out = 2 * n_eta

    # Auto-split validation if not provided
    if val_parameters is None:
        rng = np.random.default_rng(seed + 7777)
        n_val = max(1, int(N * 0.15))
        perm = rng.permutation(N)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        val_parameters = parameters[val_idx]
        val_cd = current_density[val_idx]
        val_pc = peroxide_current[val_idx]
        parameters = parameters[train_idx]
        current_density = current_density[train_idx]
        peroxide_current = peroxide_current[train_idx]
        N = parameters.shape[0]

    # Build log-space inputs
    X_log = parameters.copy()
    X_log[:, 0] = np.log10(np.maximum(X_log[:, 0], 1e-30))
    X_log[:, 1] = np.log10(np.maximum(X_log[:, 1], 1e-30))

    Y = np.concatenate([current_density, peroxide_current], axis=1)

    # Normalizers
    input_norm = ZScoreNormalizer.from_data(X_log)
    output_norm = ZScoreNormalizer.from_data(Y)

    X_norm = input_norm.transform(X_log)
    Y_norm = output_norm.transform(Y)

    # Validation
    X_val_log = val_parameters.copy()
    X_val_log[:, 0] = np.log10(np.maximum(X_val_log[:, 0], 1e-30))
    X_val_log[:, 1] = np.log10(np.maximum(X_val_log[:, 1], 1e-30))
    X_val_norm = input_norm.transform(X_val_log)
    Y_val = np.concatenate([val_cd, val_pc], axis=1)
    Y_val_norm = output_norm.transform(Y_val)

    # Tensors
    device = torch.device("cpu")
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_norm, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
    Y_val_t = torch.tensor(Y_val_norm, dtype=torch.float32, device=device)

    # Build model
    torch.manual_seed(seed)
    net = ResNetMLP(
        n_in=4, n_out=n_out,
        hidden=config.hidden, n_blocks=config.n_blocks,
    ).to(device)

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min,
    )

    # Output dir setup
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{model_tag}_log.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "epoch", "train_loss", "val_loss",
            "val_cd_rmse", "val_pc_rmse", "lr", "elapsed_s",
        ])
    else:
        csv_file = None
        csv_writer = None

    batch_size = config.batch_size if config.batch_size is not None else N
    if batch_size > N:
        batch_size = N

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history: Dict[str, List[float]] = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_cd_rmse": [], "val_pc_rmse": [], "lr": [],
    }
    t_start = time.time()

    if verbose:
        print(f"\n  Training {model_tag} (seed={seed}, N_train={N}, "
              f"N_val={len(val_parameters)}, n_out={n_out})", flush=True)

    for epoch in range(1, config.epochs + 1):
        # --- Train ---
        net.train()
        perm_t = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm_t[start : start + batch_size]
            xb = X_t[idx]
            yb = Y_t[idx]

            pred = net(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)

            # Physics regularization
            if config.monotonicity_weight > 0:
                loss = loss + config.monotonicity_weight * _monotonicity_penalty(
                    net, xb, n_eta, phi_applied,
                )
            if config.smoothness_weight > 0:
                loss = loss + config.smoothness_weight * _smoothness_penalty(
                    net, xb, n_eta,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / n_batches

        # --- Validate ---
        net.eval()
        with torch.no_grad():
            val_pred_norm = net(X_val_t).cpu().numpy()
            val_loss_t = torch.nn.functional.mse_loss(
                torch.tensor(val_pred_norm), Y_val_t.cpu(),
            ).item()

        # Inverse-transform for physical-space RMSE
        val_pred_phys = output_norm.inverse_transform(val_pred_norm)
        val_cd_pred = val_pred_phys[:, :n_eta]
        val_pc_pred = val_pred_phys[:, n_eta:]
        val_cd_rmse = float(np.sqrt(np.mean((val_cd_pred - val_cd) ** 2)))
        val_pc_rmse = float(np.sqrt(np.mean((val_pc_pred - val_pc) ** 2)))

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_start

        # Log
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss_t)
        history["val_cd_rmse"].append(val_cd_rmse)
        history["val_pc_rmse"].append(val_pc_rmse)
        history["lr"].append(current_lr)

        if csv_writer is not None:
            csv_writer.writerow([
                epoch, f"{train_loss:.8e}", f"{val_loss_t:.8e}",
                f"{val_cd_rmse:.8e}", f"{val_pc_rmse:.8e}",
                f"{current_lr:.6e}", f"{elapsed:.1f}",
            ])

        # Early stopping
        if val_loss_t < best_val_loss:
            best_val_loss = val_loss_t
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Checkpointing
        if output_dir is not None and epoch % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(output_dir, f"{model_tag}_epoch{epoch}.pt")
            torch.save(net.state_dict(), ckpt_path)
            # Plot loss curves
            _plot_loss_curves(history, output_dir, model_tag, epoch)

        if verbose and (epoch % 500 == 0 or epoch == 1):
            print(
                f"  [{model_tag}] Epoch {epoch:>5d}/{config.epochs}  "
                f"train={train_loss:.6e}  val={val_loss_t:.6e}  "
                f"cd_rmse={val_cd_rmse:.4e}  pc_rmse={val_pc_rmse:.4e}  "
                f"lr={current_lr:.2e}  dt={elapsed:.0f}s",
                flush=True,
            )

        if epochs_no_improve >= config.patience:
            if verbose:
                print(
                    f"  [{model_tag}] Early stopping at epoch {epoch} "
                    f"(patience={config.patience})",
                    flush=True,
                )
            break

    # Restore best
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    # Save best model
    if output_dir is not None:
        best_path = os.path.join(output_dir, f"{model_tag}_best.pt")
        torch.save(net.state_dict(), best_path)
        _plot_loss_curves(history, output_dir, model_tag, epoch)

    if csv_file is not None:
        csv_file.close()

    # Build NNSurrogateModel wrapper
    surrogate = NNSurrogateModel(
        hidden=config.hidden, n_blocks=config.n_blocks,
        seed=seed, device="cpu",
    )
    surrogate._model = net
    surrogate._input_normalizer = input_norm
    surrogate._output_normalizer = output_norm
    surrogate._phi_applied = phi_applied.copy()
    surrogate._n_eta = n_eta
    surrogate._is_fitted = True
    surrogate.training_bounds = {
        "k0_1": (float(parameters[:, 0].min()), float(parameters[:, 0].max())),
        "k0_2": (float(parameters[:, 1].min()), float(parameters[:, 1].max())),
        "alpha_1": (float(parameters[:, 2].min()), float(parameters[:, 2].max())),
        "alpha_2": (float(parameters[:, 3].min()), float(parameters[:, 3].max())),
    }

    total_time = time.time() - t_start
    if verbose:
        print(
            f"  [{model_tag}] Done: best_val_loss={best_val_loss:.6e}, "
            f"total_time={total_time:.1f}s, final_epoch={epoch}",
            flush=True,
        )

    return surrogate, {
        "history": history,
        "best_val_loss": best_val_loss,
        "final_epoch": epoch,
        "total_time_s": total_time,
    }


# ---------------------------------------------------------------------------
# Loss curve plotting
# ---------------------------------------------------------------------------

def _plot_loss_curves(
    history: Dict[str, List[float]],
    output_dir: str,
    model_tag: str,
    current_epoch: int,
) -> None:
    """Plot training and validation loss curves, save to output_dir.

    Parameters
    ----------
    history : dict
        Training history with 'epoch', 'train_loss', 'val_loss' keys.
    output_dir : str
        Directory to save the plot.
    model_tag : str
        Model name tag for the filename.
    current_epoch : int
        Current epoch number (used in title).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib not available, skip plotting

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = history["epoch"]

    # Loss curves
    ax = axes[0]
    ax.semilogy(epochs, history["train_loss"], label="Train", alpha=0.8)
    ax.semilogy(epochs, history["val_loss"], label="Val", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalized)")
    ax.set_title(f"{model_tag}: Loss (epoch {current_epoch})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Physical RMSE
    ax = axes[1]
    ax.semilogy(epochs, history["val_cd_rmse"], label="CD RMSE", alpha=0.8)
    ax.semilogy(epochs, history["val_pc_rmse"], label="PC RMSE", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (physical units)")
    ax.set_title(f"{model_tag}: Validation RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_tag}_loss_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ensemble training
# ---------------------------------------------------------------------------

def train_nn_ensemble(
    parameters: np.ndarray,
    current_density: np.ndarray,
    peroxide_current: np.ndarray,
    phi_applied: np.ndarray,
    *,
    n_ensemble: int = 5,
    config: NNTrainingConfig | None = None,
    output_dir: str = "nn_ensemble",
    base_seed: int = 42,
    val_fraction: float = 0.15,
    verbose: bool = True,
) -> Tuple[List[NNSurrogateModel], Dict[str, Any]]:
    """Train an ensemble of NN surrogate models with different seeds and splits.

    Each ensemble member uses a different random seed (for weight
    initialization) and a different train/val split.

    Parameters
    ----------
    parameters : np.ndarray (N, 4)
        All training parameters.
    current_density : np.ndarray (N, n_eta)
        All CD curves.
    peroxide_current : np.ndarray (N, n_eta)
        All PC curves.
    phi_applied : np.ndarray (n_eta,)
        Voltage grid.
    n_ensemble : int
        Number of ensemble members.
    config : NNTrainingConfig or None
        Shared training configuration.
    output_dir : str
        Output directory for all models.
    base_seed : int
        Base random seed (each member gets base_seed + i).
    val_fraction : float
        Fraction of data reserved for validation per member.
    verbose : bool
        Print progress.

    Returns
    -------
    (list of NNSurrogateModel, dict)
        List of fitted models and ensemble-level metadata.
    """
    _check_torch()

    if config is None:
        config = NNTrainingConfig()

    os.makedirs(output_dir, exist_ok=True)
    N = parameters.shape[0]
    n_val = max(1, int(N * val_fraction))

    models: List[NNSurrogateModel] = []
    all_histories: List[Dict[str, Any]] = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"  NN ENSEMBLE TRAINING")
        print(f"  N_data={N}, n_ensemble={n_ensemble}, "
              f"val_fraction={val_fraction}")
        print(f"  Output: {output_dir}")
        print(f"{'='*70}\n")

    for i in range(n_ensemble):
        member_seed = base_seed + i
        member_tag = f"member_{i}"
        member_dir = os.path.join(output_dir, member_tag)

        # Generate unique train/val split for this member
        rng = np.random.default_rng(member_seed)
        perm = rng.permutation(N)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        if verbose:
            print(f"\n--- Ensemble member {i+1}/{n_ensemble} "
                  f"(seed={member_seed}) ---", flush=True)

        model, info = train_nn_surrogate(
            parameters=parameters[train_idx],
            current_density=current_density[train_idx],
            peroxide_current=peroxide_current[train_idx],
            phi_applied=phi_applied,
            val_parameters=parameters[val_idx],
            val_cd=current_density[val_idx],
            val_pc=peroxide_current[val_idx],
            config=config,
            seed=member_seed,
            output_dir=member_dir,
            model_tag=member_tag,
            verbose=verbose,
        )

        # Save model
        model.save(os.path.join(member_dir, "saved_model"))
        models.append(model)
        all_histories.append(info)

    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"  ENSEMBLE TRAINING COMPLETE")
        print(f"{'='*70}")
        for i, info in enumerate(all_histories):
            print(
                f"  Member {i}: best_val={info['best_val_loss']:.6e}, "
                f"epoch={info['final_epoch']}, "
                f"time={info['total_time_s']:.1f}s",
            )
        print(f"{'='*70}\n")

    return models, {
        "n_ensemble": n_ensemble,
        "histories": all_histories,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Ensemble prediction (mean + std)
# ---------------------------------------------------------------------------

def predict_ensemble(
    models: Sequence[NNSurrogateModel],
    parameters: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Predict with an ensemble and return mean + std.

    Parameters
    ----------
    models : sequence of NNSurrogateModel
        Ensemble members.
    parameters : np.ndarray (M, 4)
        Parameter sets to predict.

    Returns
    -------
    dict with keys:
        'current_density_mean' : np.ndarray (M, n_eta)
        'current_density_std' : np.ndarray (M, n_eta)
        'peroxide_current_mean' : np.ndarray (M, n_eta)
        'peroxide_current_std' : np.ndarray (M, n_eta)
        'phi_applied' : np.ndarray (n_eta,)
    """
    cd_preds = []
    pc_preds = []

    for m in models:
        result = m.predict_batch(parameters)
        cd_preds.append(result["current_density"])
        pc_preds.append(result["peroxide_current"])

    cd_stack = np.stack(cd_preds, axis=0)  # (E, M, n_eta)
    pc_stack = np.stack(pc_preds, axis=0)

    return {
        "current_density_mean": cd_stack.mean(axis=0),
        "current_density_std": cd_stack.std(axis=0),
        "peroxide_current_mean": pc_stack.mean(axis=0),
        "peroxide_current_std": pc_stack.std(axis=0),
        "phi_applied": models[0].phi_applied,
    }
