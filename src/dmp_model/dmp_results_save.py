"""
Result saving utilities for Dynamical Mode Pruning (DMP).

This module saves the artifacts produced by a DMPESN run:

    - metrics.csv
    - prediction CSV files
    - true-vs-prediction plots
    - prediction error heatmaps
    - DMP neuron scores
    - kept / removed neuron lists
    - Gramian eigenspectrum diagnostics
    - reservoir/readout/state heatmaps

The code is intentionally independent from the pruning implementation. It only
expects the runner object to expose diagnostic attributes such as:

    runner.last_base_esn
    runner.last_pruned_esn
    runner.last_pruned_y_test
    runner.last_scores
    runner.keep_idx or runner.last_keep_idx
    runner.last_pruner

These attributes are provided by DMPESN in dmp_esn.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


# ==============================================================
# Global save settings
# ==============================================================

FIG_DPI = 320
CSV_FLOAT = "%.8f"


# ==============================================================
# Small helpers
# ==============================================================

def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_series(x) -> np.ndarray:
    """Convert an array-like object to a flat NumPy float array."""
    return np.asarray(x, dtype=float).reshape(-1)


def spectral_radius(matrix: torch.Tensor) -> float:
    """Compute the exact spectral radius of a square matrix."""
    if matrix.ndim != 2:
        return 0.0

    if matrix.shape[0] != matrix.shape[1]:
        return 0.0

    if matrix.numel() == 0:
        return 0.0

    eigvals = torch.linalg.eigvals(matrix.detach().to(torch.complex128))
    return float(torch.max(torch.abs(eigvals)).real.item())


# ==============================================================
# Prediction reconstruction
# ==============================================================

def compute_test_target(esn, data: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct the multi-step test target used by the ESN.

    The returned tensor has shape:

        (outSize * multi_step, effective_test_len)

    For one-step plotting, the first row is used later.
    """
    test_start = esn.initLen + esn.trainLen
    effective_test_len = min(
        esn.testLen,
        len(data) - test_start - esn.tau * esn.multi_step,
    )

    if effective_test_len <= 0:
        raise ValueError("Test length is too short for target reconstruction.")

    target = torch.zeros(
        (esn.outSize * esn.multi_step, effective_test_len),
        device=esn.device,
        dtype=getattr(esn, "dtype", torch.float64),
    )

    for step in range(esn.multi_step):
        s = test_start + esn.tau * (step + 1)
        e = s + effective_test_len

        target[
            step * esn.outSize : (step + 1) * esn.outSize,
            :,
        ] = data[s:e].reshape(esn.outSize, -1)

    return target


def prediction_arrays(esn, y_test: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert normalized ESN predictions and targets back to original scale.

    Returns
    -------
    y_true:
        One-dimensional target array.

    y_pred:
        One-dimensional prediction array.
    """
    target_test = compute_test_target(esn, esn.data)

    y_true = esn.denorm(target_test).detach().cpu().numpy()
    y_pred = esn.denorm(y_test).detach().cpu().numpy()

    y_true = y_true[0, :] if y_true.ndim == 2 else y_true.reshape(-1)
    y_pred = y_pred[0, :] if y_pred.ndim == 2 else y_pred.reshape(-1)

    return y_true, y_pred


# ==============================================================
# Prediction plots
# ==============================================================

def plot_real_vs_prediction(
    y_true,
    y_pred,
    save_path: Path,
    title: str,
    max_points: int = 500,
) -> None:
    """Save a true-vs-prediction line plot."""
    y_true = clean_series(y_true)
    y_pred = clean_series(y_pred)

    n = int(min(len(y_true), len(y_pred), max_points))

    if n <= 1:
        return

    x = np.arange(n)
    abs_error = np.abs(y_true[:n] - y_pred[:n])

    mse = float(np.mean(abs_error ** 2))
    rmse = float(np.sqrt(mse))

    plt.figure(figsize=(13.2, 4.8))
    plt.plot(x, y_true[:n], label="True", linewidth=2.2)
    plt.plot(x, y_pred[:n], label="Predicted", linewidth=2.0, alpha=0.9)
    plt.fill_between(x, y_true[:n], y_pred[:n], alpha=0.08, linewidth=0)

    plt.title(f"{title} | RMSE={rmse:.4g} | MSE={mse:.4g}")
    plt.xlabel("Test time index")
    plt.ylabel("Value")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_prediction_error_heatmap(
    y_true,
    y_pred,
    save_path: Path,
    title: str,
    max_points: int = 1200,
) -> None:
    """Save a one-row heatmap of absolute prediction error."""
    y_true = clean_series(y_true)
    y_pred = clean_series(y_pred)

    n = int(min(len(y_true), len(y_pred), max_points))

    if n <= 1:
        return

    abs_error = np.abs(y_true[:n] - y_pred[:n])

    if np.isfinite(abs_error).any():
        vmax = float(np.nanpercentile(abs_error, 99))
    else:
        vmax = 1.0

    plt.figure(figsize=(13.2, 2.8))
    im = plt.imshow(
        abs_error.reshape(1, -1),
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        vmin=0.0,
        vmax=max(vmax, 1e-12),
    )

    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label("Absolute error")

    plt.title(f"{title} | max={np.max(abs_error):.4g}")
    plt.xlabel("Test time index")
    plt.yticks([0], ["|e|"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


# ==============================================================
# DMP score plots
# ==============================================================

def plot_score_scatter(
    values,
    save_path: Path,
    title: str = "DMP score scatter",
    pruned_idx: Optional[np.ndarray] = None,
) -> None:
    """Save a scatter plot of neuron scores, marking kept and pruned neurons."""
    values = clean_series(values)
    x = np.arange(len(values))

    keep_mask = np.ones(len(values), dtype=bool)

    if pruned_idx is not None and len(pruned_idx) > 0:
        keep_mask[np.asarray(pruned_idx, dtype=int)] = False

    plt.figure(figsize=(10.8, 4.2))

    if keep_mask.any():
        plt.scatter(
            x[keep_mask],
            values[keep_mask],
            s=14,
            alpha=0.8,
            label="Kept",
        )

    if (~keep_mask).any():
        plt.scatter(
            x[~keep_mask],
            values[~keep_mask],
            s=16,
            alpha=0.9,
            label="Pruned",
        )

    plt.xlabel("Neuron index")
    plt.ylabel("DMP score")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def save_score_bar_plot(
    save_path: Path,
    scores: np.ndarray,
    kept_mask: np.ndarray,
    title: str,
) -> None:
    """Save a bar plot of DMP scores by neuron index."""
    plt.figure(figsize=(11.0, 4.8))

    # Use default Matplotlib colors through separate calls.
    x = np.arange(len(scores))

    kept = kept_mask.astype(bool)
    pruned = ~kept

    if kept.any():
        plt.bar(x[kept], scores[kept], width=0.9, label="Kept")

    if pruned.any():
        plt.bar(x[pruned], scores[pruned], width=0.9, label="Pruned")

    plt.xlabel("Neuron index")
    plt.ylabel("DMP score")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def save_sorted_score_plot(
    save_path: Path,
    scores: np.ndarray,
    num_kept: int,
    title: str,
) -> None:
    """Save sorted DMP scores with a vertical line at the pruning boundary."""
    order = np.argsort(scores)[::-1]

    plt.figure(figsize=(9.0, 4.8))
    plt.plot(scores[order], linewidth=2.0)

    if num_kept > 0:
        plt.axvline(num_kept - 1, linestyle="--", linewidth=1.8)

    plt.xlabel("Rank")
    plt.ylabel("DMP score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def save_kept_removed_histogram(
    save_path: Path,
    kept_scores: np.ndarray,
    removed_scores: np.ndarray,
    title: str,
) -> None:
    """Save score distributions for kept and removed neurons."""
    kept_scores = kept_scores[np.isfinite(kept_scores)]
    removed_scores = removed_scores[np.isfinite(removed_scores)]

    plt.figure(figsize=(8.5, 4.8))

    if len(kept_scores) > 0:
        plt.hist(kept_scores, bins=20, alpha=0.75, label="Kept")

    if len(removed_scores) > 0:
        plt.hist(removed_scores, bins=20, alpha=0.72, label="Removed")

    plt.xlabel("DMP score")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


# ==============================================================
# Matrix and state heatmaps
# ==============================================================

def save_matrix_heatmap(matrix: torch.Tensor, save_path: Path, title: str) -> None:
    """Save a heatmap for a matrix-like tensor."""
    values = matrix.detach().cpu().numpy()
    values = np.atleast_2d(values)

    vmax = float(np.max(np.abs(values))) if values.size else 1.0
    vmax = max(vmax, 1e-12)

    plt.figure(figsize=(8.0, 5.6))
    plt.imshow(
        values,
        cmap="coolwarm",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def save_state_heatmap(states: torch.Tensor, save_path: Path, title: str) -> None:
    """Save a heatmap of reservoir states over time."""
    values = states.detach().cpu().numpy()

    plt.figure(figsize=(9.2, 5.8))
    plt.imshow(values.T, cmap="viridis", aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def save_base_diagnostic_plots(
    save_dir: Path,
    scores: np.ndarray,
    keep_idx: np.ndarray,
    W: np.ndarray,
    states: np.ndarray,
    Wout: Optional[np.ndarray] = None,
    title_prefix: str = "DMP",
) -> None:
    """
    Save diagnostic plots for the original full reservoir.

    These plots are useful for appendix/debugging:
        - sorted DMP scores
        - original reservoir W heatmap
        - original reservoir state heatmap
        - optional readout magnitude heatmap
    """
    save_dir = ensure_dir(Path(save_dir))

    scores = clean_series(scores)
    order = np.argsort(scores)[::-1]

    plt.figure(figsize=(9.2, 4.2))
    plt.plot(np.arange(len(scores)), scores[order], linewidth=2.0)

    if keep_idx is not None:
        k = len(np.asarray(keep_idx, dtype=int))
        if k > 0:
            plt.axvline(k - 1, linestyle="--", linewidth=1.6)

    plt.xlabel("Rank")
    plt.ylabel("DMP score")
    plt.title(f"{title_prefix} | Sorted scores")
    plt.tight_layout()
    plt.savefig(save_dir / "diagnostic_scores_sorted.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

    W = np.asarray(W, dtype=float)

    plt.figure(figsize=(7.6, 6.0))
    vmax = max(float(np.max(np.abs(W))) if W.size else 1.0, 1e-12)
    plt.imshow(W, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"{title_prefix} | Reservoir W")
    plt.tight_layout()
    plt.savefig(save_dir / "diagnostic_W_heatmap.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

    states = np.asarray(states, dtype=float)

    if states.ndim == 2 and states.size > 0:
        plt.figure(figsize=(8.6, 5.0))
        plt.imshow(states.T, cmap="viridis", aspect="auto")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel("Time")
        plt.ylabel("Neuron")
        plt.title(f"{title_prefix} | Reservoir states")
        plt.tight_layout()
        plt.savefig(save_dir / "diagnostic_states_heatmap.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close()

    if Wout is not None:
        Wout = np.asarray(Wout, dtype=float)

        if Wout.ndim >= 2 and Wout.size > 0:
            plt.figure(figsize=(8.0, 4.0))
            plt.imshow(np.abs(Wout), cmap="magma", aspect="auto")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f"{title_prefix} | |Wout|")
            plt.tight_layout()
            plt.savefig(save_dir / "diagnostic_wout_abs.png", dpi=FIG_DPI, bbox_inches="tight")
            plt.close()


# ==============================================================
# Gramian eigenspectrum plots
# ==============================================================

def save_spectrum_plot(
    save_path: Path,
    eigen_df: pd.DataFrame,
    title: str,
) -> None:
    """Save the Gramian eigenvalue spectrum."""
    plt.figure(figsize=(9.0, 4.8))
    plt.plot(eigen_df["mode"], eigen_df["eigenvalue"], linewidth=2.0)
    plt.xlabel("Mode rank")
    plt.ylabel("Eigenvalue")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def save_cumulative_energy_plot(
    save_path: Path,
    eigen_df: pd.DataFrame,
    rank_r: int,
    energy_tau: float,
    title: str,
) -> None:
    """Save cumulative normalized Gramian energy."""
    plt.figure(figsize=(9.0, 4.8))
    plt.plot(eigen_df["mode"], eigen_df["cumulative_energy"], linewidth=2.0)

    plt.axhline(
        energy_tau,
        linestyle="--",
        linewidth=1.8,
        label=f"energy_tau={energy_tau:.2f}",
    )

    if rank_r > 0:
        plt.axvline(
            rank_r,
            linestyle="--",
            linewidth=1.8,
            label=f"r={rank_r}",
        )

    plt.xlabel("Mode rank")
    plt.ylabel("Cumulative energy")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


# ==============================================================
# Main artifact saving function
# ==============================================================

def save_dmp_artifacts(
    run_dir: Path,
    scores: torch.Tensor,
    keep_idx: torch.Tensor,
    base_esn,
    pruned_esn,
    dataset_name: str,
    res_size: int,
    seed: int,
    pruner=None,
) -> None:
    """
    Save all DMP pruning artifacts for one run.

    Parameters
    ----------
    run_dir:
        Directory where artifacts are saved.

    scores:
        Full-reservoir DMP scores. If a local/final score vector is given,
        the function maps it defensively where possible.

    keep_idx:
        Indices of kept neurons in the original reservoir indexing.

    base_esn:
        Original unpruned ESN.

    pruned_esn:
        Final pruned ESN.

    pruner:
        Optional DynamicalModePruner object with eigenspectrum diagnostics.
    """
    run_dir = ensure_dir(Path(run_dir))

    scores_cpu = scores.detach().cpu().to(torch.float64).reshape(-1)
    keep_idx_cpu = keep_idx.detach().cpu().to(torch.long)

    base_n = int(base_esn.resSize)

    kept_mask = np.zeros(base_n, dtype=int)
    kept_mask[keep_idx_cpu.numpy()] = 1
    removed_idx = np.where(kept_mask == 0)[0]

    score_np = scores_cpu.numpy()

    # Defensive fallback: map local scores back to full reservoir shape.
    if score_np.shape[0] != base_n:
        score_full = np.full(base_n, np.nan, dtype=float)

        if score_np.shape[0] == int(keep_idx_cpu.numel()):
            score_full[keep_idx_cpu.numpy()] = score_np
        else:
            n_copy = min(base_n, int(score_np.shape[0]))
            score_full[:n_copy] = score_np[:n_copy]

        score_np = score_full

    score_plot = np.nan_to_num(
        score_np,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # ----------------------------------------------------------
    # CSV files: neuron scores and selection
    # ----------------------------------------------------------
    pd.DataFrame(
        {
            "neuron_id": np.arange(base_n, dtype=int),
            "dmp_score": score_np,
            "kept": kept_mask,
        }
    ).to_csv(
        run_dir / "dmp_scores.csv",
        index=False,
        float_format=CSV_FLOAT,
    )

    pd.DataFrame(
        {"kept_neuron_id": keep_idx_cpu.numpy()}
    ).to_csv(
        run_dir / "kept_neurons.csv",
        index=False,
    )

    pd.DataFrame(
        {"removed_neuron_id": removed_idx}
    ).to_csv(
        run_dir / "removed_neurons.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Pruning summary
    # ----------------------------------------------------------
    selected_rank = None if pruner is None else getattr(pruner, "last_rank", None)
    energy_tau = None if pruner is None else getattr(pruner, "energy_tau", None)
    gramian_total_energy = None if pruner is None else getattr(pruner, "last_total_energy", None)
    jacobian_timesteps = None if pruner is None else getattr(pruner, "last_timesteps", None)

    target_size = int(keep_idx_cpu.numel())

    pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "reservoir_size": int(base_esn.resSize),
                "final_reservoir_size": int(target_size),
                "pruned_count": int(base_esn.resSize - target_size),
                "pruned_fraction": float(
                    (base_esn.resSize - target_size) / max(base_esn.resSize, 1)
                ),
                "spectral_radius_pruned": spectral_radius(pruned_esn.W),
                "energy_tau": None if energy_tau is None else float(energy_tau),
                "selected_rank_r": None if selected_rank is None else int(selected_rank),
                "gramian_total_energy": (
                    None if gramian_total_energy is None else float(gramian_total_energy)
                ),
                "jacobian_timesteps": (
                    None if jacobian_timesteps is None else int(jacobian_timesteps)
                ),
            }
        ]
    ).to_csv(
        run_dir / "pruning_summary.csv",
        index=False,
        float_format=CSV_FLOAT,
    )

    # ----------------------------------------------------------
    # Gramian eigenspectrum diagnostics
    # ----------------------------------------------------------
    if pruner is not None and getattr(pruner, "last_eigenvalues", None) is not None:
        eigen_df = pd.DataFrame(
            {
                "mode": np.arange(
                    1,
                    len(pruner.last_eigenvalues) + 1,
                    dtype=int,
                ),
                "eigenvalue": pruner.last_eigenvalues.detach().cpu().numpy(),
                "normalized_eigenvalue": (
                    pruner.last_normalized_eigenvalues.detach().cpu().numpy()
                ),
                "cumulative_energy": (
                    pruner.last_cumulative_energy.detach().cpu().numpy()
                ),
            }
        )

        eigen_df["retained"] = (
            eigen_df["mode"] <= int(selected_rank or 0)
        ).astype(int)

        eigen_df.to_csv(
            run_dir / "gramian_eigenspectrum.csv",
            index=False,
            float_format=CSV_FLOAT,
        )

        save_spectrum_plot(
            run_dir / "gramian_eigenvalue_spectrum.png",
            eigen_df,
            f"Gramian eigenvalue spectrum | RES={res_size} seed={seed}",
        )

        save_cumulative_energy_plot(
            run_dir / "gramian_cumulative_energy.png",
            eigen_df,
            rank_r=0 if selected_rank is None else int(selected_rank),
            energy_tau=0.0 if energy_tau is None else float(energy_tau),
            title=f"Gramian cumulative energy | RES={res_size} seed={seed}",
        )

    # ----------------------------------------------------------
    # Score visualizations
    # ----------------------------------------------------------
    plot_score_scatter(
        score_plot,
        run_dir / "dmp_selection_scatter.png",
        title=f"DMP scores | RES={res_size} seed={seed} | {dataset_name}",
        pruned_idx=removed_idx,
    )

    save_score_bar_plot(
        run_dir / "dmp_scores_bar.png",
        scores=score_plot,
        kept_mask=kept_mask,
        title=f"DMP scores by neuron | RES={res_size} seed={seed}",
    )

    save_sorted_score_plot(
        run_dir / "dmp_scores_sorted.png",
        scores=score_plot,
        num_kept=len(keep_idx_cpu),
        title=f"Sorted DMP scores | RES={res_size} seed={seed}",
    )

    kept_scores = score_np[keep_idx_cpu.numpy()]
    removed_scores = score_np[removed_idx]

    save_kept_removed_histogram(
        run_dir / "dmp_kept_vs_removed_hist.png",
        kept_scores=kept_scores,
        removed_scores=removed_scores,
        title=f"DMP score distribution | RES={res_size} seed={seed}",
    )

    # ----------------------------------------------------------
    # Pruned ESN matrices and states
    # ----------------------------------------------------------
    save_matrix_heatmap(
        pruned_esn.W,
        run_dir / "pruned_W_heatmap.png",
        "Pruned reservoir W",
    )

    save_matrix_heatmap(
        pruned_esn.Win,
        run_dir / "pruned_Win_heatmap.png",
        "Pruned input matrix Win",
    )

    if pruned_esn.Wout is not None:
        save_matrix_heatmap(
            pruned_esn.Wout,
            run_dir / "pruned_Wout_heatmap.png",
            "Pruned readout Wout",
        )

    if pruned_esn.reservoir_states is not None:
        pd.DataFrame(
            pruned_esn.reservoir_states.detach().cpu().numpy()
        ).to_csv(
            run_dir / "pruned_states.csv",
            index=False,
            float_format=CSV_FLOAT,
        )

        save_state_heatmap(
            pruned_esn.reservoir_states,
            run_dir / "pruned_states_heatmap.png",
            f"Pruned reservoir states | RES={res_size} seed={seed}",
        )

    # ----------------------------------------------------------
    # Full base ESN diagnostics
    # ----------------------------------------------------------
    if base_esn.reservoir_states is not None:
        base_wout = None

        if base_esn.Wout is not None:
            base_wout = (
                base_esn.Wout[..., -base_esn.resSize :]
                .detach()
                .cpu()
                .numpy()
            )

        save_base_diagnostic_plots(
            save_dir=run_dir,
            scores=score_plot,
            keep_idx=keep_idx_cpu.numpy(),
            W=base_esn.W.detach().cpu().numpy(),
            states=base_esn.reservoir_states.detach().cpu().numpy(),
            Wout=base_wout,
            title_prefix=f"DMP | RES={res_size} seed={seed}",
        )


# ==============================================================
# Saver class
# ==============================================================

@dataclass
class DMPResultSaveConfig:
    """
    Output directory configuration for DMP result saving.

    Example directory structure
    ---------------------------
    results_root /
        results_root_template /
            dataset /
                h_20 /
                    DMP /
                        prune_20pct /
                            1000 /
                                seed_0 /
    """

    results_root: str
    results_root_template: str
    model_name: str = "DMP"


class DMPResultsSaver:
    """
    Save all CSV and PNG artifacts after a DMPESN run.

    The expected row dictionary is the metrics dictionary returned by DMPESN.run.
    """

    def __init__(self, config: DMPResultSaveConfig):
        self.cfg = config

    def run_dir(self, row: Dict) -> Path:
        """Build the output directory path for a single run."""
        ratio = float(row.get("prune_ratio", 0.0))

        dataset_name = row.get("dataset", row.get("data"))
        if dataset_name is None:
            raise KeyError("row must include 'dataset' or 'data'.")

        ratio_tag = f"prune_{int(round(ratio * 100))}pct"

        return (
            Path(self.cfg.results_root)
            / self.cfg.results_root_template
            / str(dataset_name)
            / f"h_{int(row['horizon'])}"
            / self.cfg.model_name
            / ratio_tag
            / str(int(row["reservoir"]))
            / f"seed_{int(row['seed'])}"
        )

    def save(self, row: Dict, runner) -> Path:
        """
        Save all available artifacts from a finished DMPESN run.

        If some diagnostics are unavailable, the method still saves metrics.csv
        and returns the run directory.
        """
        run_dir = ensure_dir(self.run_dir(row))

        pd.DataFrame([row]).to_csv(
            run_dir / "metrics.csv",
            index=False,
            float_format=CSV_FLOAT,
        )

        base_esn = getattr(runner, "last_base_esn", None)
        pruned_esn = getattr(runner, "last_pruned_esn", None)
        y_pruned = getattr(runner, "last_pruned_y_test", None)
        scores = getattr(runner, "last_scores", None)
        keep_idx = getattr(runner, "keep_idx", None)

        if keep_idx is None:
            keep_idx = getattr(runner, "last_keep_idx", None)

        pruner = getattr(runner, "last_pruner", None)

        required = [base_esn, pruned_esn, y_pruned, scores, keep_idx]

        if any(item is None for item in required):
            return run_dir

        # Save denormalized prediction arrays and plots.
        y_true_pruned, y_pred_pruned = prediction_arrays(pruned_esn, y_pruned)

        pd.DataFrame(
            {
                "y_true": y_true_pruned,
                "y_pred": y_pred_pruned,
            }
        ).to_csv(
            run_dir / "pruned_test_pred_1step.csv",
            index=False,
            float_format=CSV_FLOAT,
        )

        plot_real_vs_prediction(
            y_true_pruned,
            y_pred_pruned,
            run_dir / "pruned_test_true_vs_pred.png",
            title=(
                f"DMP ESN | RES={row['reservoir']} "
                f"seed={row['seed']} | {row['dataset']}"
            ),
        )

        plot_prediction_error_heatmap(
            y_true_pruned,
            y_pred_pruned,
            run_dir / "pruned_test_error_heatmap.png",
            title=f"DMP ESN error | RES={row['reservoir']} seed={row['seed']}",
        )

        save_dmp_artifacts(
            run_dir=run_dir,
            scores=scores,
            keep_idx=keep_idx,
            base_esn=base_esn,
            pruned_esn=pruned_esn,
            dataset_name=str(row["dataset"]),
            res_size=int(row["reservoir"]),
            seed=int(row["seed"]),
            pruner=pruner,
        )

        return run_dir


# ==============================================================
# Backward-compatible aliases
# ==============================================================

# These aliases allow older experiment scripts to keep working while the public
# code uses DMP naming.
OursResultSaveConfig = DMPResultSaveConfig
OursJacobianResultsSaver = DMPResultsSaver
save_jacobian_artifacts = save_dmp_artifacts


__all__ = [
    "DMPResultSaveConfig",
    "DMPResultsSaver",
    "save_dmp_artifacts",

    # Backward-compatible aliases
    "OursResultSaveConfig",
    "OursJacobianResultsSaver",
    "save_jacobian_artifacts",
]
