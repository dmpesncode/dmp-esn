#!/usr/bin/env python3
"""
Standalone ESN pruned once with betweenness centrality.

Pure model module:
- build one ESN
- bulk prune once by betweenness
- train once
- return train/test metrics and optional artifacts
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import psutil
import torch

from .base_esn import EchoStateNetwork, identity, spectral_radius_power
from src.utils.data_cache import hydrate_esn_series


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cpu_rss_mb() -> float:
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))


def gpu_peak_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return 0.0


def rmse_from_mse(mse: float) -> float:
    return float(np.sqrt(mse))


def build_nx_graph_from_reservoir(W: torch.Tensor) -> nx.DiGraph:
    """
    Map nonzero recurrent weights to a directed graph.

    In the ESN update W @ x, W[i, j] means neuron j contributes to neuron i.
    Therefore, the graph edge is stored as j -> i to represent signal flow.
    """
    W_np = W.detach().cpu().numpy()

    graph = nx.DiGraph()
    graph.add_nodes_from(range(W_np.shape[0]))

    rows, cols = np.nonzero(W_np)

    for i, j in zip(rows.tolist(), cols.tolist()):
        weight = float(W_np[i, j])
        strength = abs(weight)

        if strength <= 1e-12:
            continue

        graph.add_edge(
            j,
            i,
            weight=strength,
            distance=1.0 / (strength + 1e-12),
        )

    return graph


def compute_betweenness_scores(W: torch.Tensor) -> torch.Tensor:
    """Return one betweenness-centrality pruning score per reservoir node."""
    graph = build_nx_graph_from_reservoir(W)

    scores = nx.betweenness_centrality(
        graph,
        normalized=True,
        weight="distance",
    )

    values = np.array(
        [scores.get(i, 0.0) for i in range(W.shape[0])],
        dtype=np.float64,
    )

    return torch.from_numpy(values).to(device=W.device, dtype=W.dtype)


def prediction_arrays(esn: EchoStateNetwork, y_test: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    target_test = compute_test_target(esn, esn.data)

    y_true = esn.denorm(target_test).detach().cpu().numpy()
    y_pred = esn.denorm(y_test).detach().cpu().numpy()

    y_true = y_true[0, :] if y_true.ndim == 2 else y_true.reshape(-1)
    y_pred = y_pred[0, :] if y_pred.ndim == 2 else y_pred.reshape(-1)

    return y_true, y_pred


@dataclass
class BetweennessPruningConfig:
    # ESN hyperparameters
    tau: int
    multi_step: int
    leaky_rate: float
    spectral_radius: float
    ridge_alpha: float
    sparsity: float
    input_scaling: float
    bias_scaling: float
    feedback_scaling: float
    normalize_states: bool
    use_feedback: bool

    # Pruning hyperparameters
    prune_ratio: float
    prune_sr_iters: int

    # Runtime and artifact options
    device: str

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive.")
        if self.multi_step <= 0:
            raise ValueError("multi_step must be positive.")
        if not 0.0 < self.leaky_rate <= 1.0:
            raise ValueError("leaky_rate must be in (0, 1].")
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError("sparsity must be in [0, 1).")
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative.")
        if not 0.0 <= self.prune_ratio < 1.0:
            raise ValueError("prune_ratio must be in [0, 1).")
        if self.prune_sr_iters <= 0:
            raise ValueError("prune_sr_iters must be positive.")


class BetweennessPruningESN(EchoStateNetwork):
    """Single ESN that applies one bulk betweenness-pruning step before training."""

    def __init__(
        self,
        res_size: int,
        seed: int,
        init_len: int,
        train_len: int,
        test_len: int,
        config: BetweennessPruningConfig,
        data_path: Optional[str],
    ):
        super().__init__(
            trainLen=train_len,
            testLen=test_len,
            initLen=init_len,
            tau=config.tau,
            resSize=res_size,
            inSize=1,
            outSize=1,
            a=config.leaky_rate,
            spectral_radius=config.spectral_radius,
            data_path=data_path,
            use_ridge=True,
            ridge_alpha=config.ridge_alpha,
            sparsity=config.sparsity,
            activation=torch.tanh,
            output_activation=identity,
            normalize_states=config.normalize_states,
            use_feedback=config.use_feedback,
            store_states=False,
            input_scaling=config.input_scaling,
            bias_scaling=config.bias_scaling,
            feedback_scaling=config.feedback_scaling,
            multi_step=config.multi_step,
            seed=seed,
            device=config.device,
        )

        self.cfg = config
        self.original_reservoir_size = int(self.resSize)
        self.final_reservoir_size = int(self.resSize)

        self.node_scores: Optional[torch.Tensor] = None
        self.pruned_node_ids: List[int] = []
        self.keep_idx: Optional[torch.Tensor] = None

        # Saved for optional inspection by experiment scripts.
        self.last_y_test: Optional[torch.Tensor] = None
        self.last_train_mse: Optional[float] = None
        self.last_test_mse: Optional[float] = None

    def _one_shot_prune_count(self, res_size: int) -> int:
        if self.cfg.prune_ratio <= 0.0:
            return 0

        return min(
            max(1, int(round(self.cfg.prune_ratio * res_size))),
            res_size - 1,
        )

    def prune_reservoir(self) -> None:
        """Prune once, then rescale recurrent matrix back to target spectral radius."""
        if self.W is None or self.Win is None:
            raise ValueError("Call generate_reservoir() before pruning.")

        self.original_reservoir_size = int(self.W.shape[0])
        self.node_scores = compute_betweenness_scores(self.W)

        prune_count = self._one_shot_prune_count(self.original_reservoir_size)

        if prune_count == 0:
            self.final_reservoir_size = self.original_reservoir_size
            self.pruned_node_ids = []
            self.keep_idx = torch.arange(self.original_reservoir_size, device=self.W.device)
            return

        # Low betweenness nodes are removed first.
        remove_idx = torch.argsort(self.node_scores)[:prune_count]

        keep_mask = torch.ones(
            self.original_reservoir_size,
            device=self.W.device,
            dtype=torch.bool,
        )
        keep_mask[remove_idx] = False
        keep_idx = torch.where(keep_mask)[0]

        self.keep_idx = keep_idx.detach().clone()
        self.pruned_node_ids = remove_idx.detach().cpu().numpy().astype(int).tolist()

        self.W = self.W.index_select(0, keep_idx).index_select(1, keep_idx)
        self.Win = self.Win.index_select(0, keep_idx)

        if self.Wfb is not None:
            self.Wfb = self.Wfb.index_select(0, keep_idx)

        rho = spectral_radius_power(self.W, n_iter=self.cfg.prune_sr_iters)

        if rho > 0.0:
            self.W *= self.spectral_radius / rho

        self.resSize = int(self.W.shape[0])
        self.final_reservoir_size = int(self.resSize)

    def generate_reservoir(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        super().generate_reservoir()
        self.prune_reservoir()
        return self.W, self.Win, self.Wfb

    def _save_artifacts(self, y_test: torch.Tensor, out_dir: str, title_prefix: str, row: Dict) -> None:
        ensure_dir(out_dir)

        y_true, y_pred = prediction_arrays(self, y_test)

        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
            os.path.join(out_dir, "test_pred_1step.csv"),
            index=False,
            float_format="%.8f",
        )

        plot_real_vs_prediction(
            y_true,
            y_pred,
            os.path.join(out_dir, "test_true_vs_pred.png"),
            title=title_prefix,
        )

        plot_prediction_error_heatmap(
            y_true,
            y_pred,
            os.path.join(out_dir, "test_error_heatmap.png"),
            title=f"{title_prefix} | error",
        )

        if self.node_scores is not None:
            node_ids = np.arange(int(self.original_reservoir_size), dtype=int)
            pruned_lookup = np.isin(
                node_ids,
                np.array(self.pruned_node_ids, dtype=int),
            ).astype(int)

            kept_lookup = 1 - pruned_lookup

            pd.DataFrame(
                {
                    "node_index": node_ids,
                    "betweenness_score": self.node_scores.detach().cpu().numpy(),
                    "kept": kept_lookup,
                    "pruned": pruned_lookup,
                }
            ).to_csv(
                os.path.join(out_dir, "node_scores.csv"),
                index=False,
                float_format="%.8f",
            )

        if self.keep_idx is not None:
            pd.DataFrame(
                {
                    "kept_node_id": self.keep_idx.detach().cpu().numpy().astype(int),
                }
            ).to_csv(
                os.path.join(out_dir, "kept_nodes.csv"),
                index=False,
            )

        pd.DataFrame(
            {
                "pruned_node_id": np.array(self.pruned_node_ids, dtype=int),
            }
        ).to_csv(
            os.path.join(out_dir, "pruned_nodes.csv"),
            index=False,
        )

        with open(os.path.join(out_dir, "pruned_nodes.txt"), "w", encoding="utf-8") as handle:
            handle.write(", ".join(map(str, self.pruned_node_ids)))

        pd.DataFrame([row]).to_csv(
            os.path.join(out_dir, "metrics.csv"),
            index=False,
            float_format="%.8f",
        )

    def run(
        self,
        series_raw_cpu: Optional[torch.Tensor],
        out_dir: Optional[str],
        dataset_name: str,
    ) -> Dict:
        """Run pure bulk-betweenness training and return required outputs."""
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        cpu0 = cpu_rss_mb()

        if series_raw_cpu is None:
            self.load_data()
        else:
            hydrate_esn_series(self, series_raw_cpu)

        self.generate_reservoir()
        self.train_esn(self.data)

        y_test, train_mse, test_mse = self.err(self.data, print_mse=False)

        runtime_ms = (time.time() - t0) * 1000.0
        cpu1 = cpu_rss_mb()
        gmem = gpu_peak_mb()

        self.last_y_test = y_test
        self.last_train_mse = float(train_mse)
        self.last_test_mse = float(test_mse)

        row = {
            "dataset": dataset_name,
            "reservoir": int(self.original_reservoir_size),
            "final_reservoir_size": int(self.final_reservoir_size),
            "pruned_count": int(self.original_reservoir_size - self.final_reservoir_size),
            "seed": int(self.seed),
            "train_mse": float(train_mse),
            "train_rmse": rmse_from_mse(float(train_mse)),
            "test_mse": float(test_mse),
            "test_rmse": rmse_from_mse(float(test_mse)),
            "total_time_ms": float(runtime_ms),
            "total_time_s": float(runtime_ms / 1000.0),
            "cpu_rss_mb": float(cpu1),
            "cpu_rss_delta_mb": float(cpu1 - cpu0),
            "gpu_peak_mem_mb": float(gmem),
            "horizon": int(self.cfg.multi_step),
            "tau": int(self.cfg.tau),
            "leaky_rate": float(self.cfg.leaky_rate),
            "spectral_radius": float(self.cfg.spectral_radius),
            "sparsity": float(self.cfg.sparsity),
            "ridge_alpha": float(self.cfg.ridge_alpha),
            "input_scaling": float(self.cfg.input_scaling),
            "bias_scaling": float(self.cfg.bias_scaling),
            "feedback_scaling": float(self.cfg.feedback_scaling),
            "normalize_states": bool(self.cfg.normalize_states),
            "use_feedback": bool(self.cfg.use_feedback),
            "prune_ratio": float(self.cfg.prune_ratio),
            "prune_sr_iters": int(self.cfg.prune_sr_iters),
            "device": str(self.device),
        }

        if out_dir:
            self._save_artifacts(
                y_test=y_test,
                out_dir=out_dir,
                title_prefix=(
                    f"Betweenness-pruned ESN | RES={self.original_reservoir_size} "
                    f"| seed={self.seed}"
                ),
                row=row,
            )

        return row


def _clean_series(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def compute_test_target(esn, data: torch.Tensor) -> torch.Tensor:
    test_start = esn.initLen + esn.trainLen
    effective_test_len = min(esn.testLen, len(data) - test_start - esn.tau * esn.multi_step)

    if effective_test_len <= 0:
        raise ValueError("Test length too short for target reconstruction.")

    target = torch.zeros(
        (esn.outSize * esn.multi_step, effective_test_len),
        device=esn.device,
        dtype=getattr(esn, "dtype", torch.float64),
    )

    for step in range(esn.multi_step):
        s = test_start + esn.tau * (step + 1)
        e = s + effective_test_len

        target[
            step * esn.outSize:(step + 1) * esn.outSize,
            :,
        ] = data[s:e].reshape(esn.outSize, -1)

    return target


def plot_real_vs_prediction(y_true, y_pred, save_path: str, title: str, max_points: int = 500) -> None:
    y_true = _clean_series(y_true)
    y_pred = _clean_series(y_pred)

    n = int(min(len(y_true), len(y_pred), max_points))
    if n <= 1:
        return

    x = np.arange(n)
    abs_error = np.abs(y_true[:n] - y_pred[:n])

    mse = float(np.mean(abs_error ** 2))
    rmse = float(np.sqrt(mse))

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    ax.plot(x, y_true[:n], label="True", linewidth=2.2, color="#1f77b4")
    ax.plot(x, y_pred[:n], label="Predicted", linewidth=2.0, alpha=0.9, color="#d62728")
    ax.fill_between(x, y_true[:n], y_pred[:n], color="#d62728", alpha=0.08, linewidth=0)

    ax.set_title(f"{title} | RMSE={rmse:.4g} | MSE={mse:.4g}")
    ax.set_xlabel("Test time index")
    ax.set_ylabel("Value")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_error_heatmap(y_true, y_pred, save_path: str, title: str, max_points: int = 1200) -> None:
    y_true = _clean_series(y_true)
    y_pred = _clean_series(y_pred)

    n = int(min(len(y_true), len(y_pred), max_points))
    if n <= 1:
        return

    abs_error = np.abs(y_true[:n] - y_pred[:n])
    vmax = float(np.nanpercentile(abs_error, 99)) if np.isfinite(abs_error).any() else 1.0

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(13.2, 2.8))
    im = ax.imshow(
        abs_error.reshape(1, -1),
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        vmin=0.0,
        vmax=max(vmax, 1e-12),
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Absolute error")

    ax.set_title(f"{title} | max={np.max(abs_error):.4g}")
    ax.set_xlabel("Test time index")
    ax.set_yticks([0])
    ax.set_yticklabels(["|e|"])

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)