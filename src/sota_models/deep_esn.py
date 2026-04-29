#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_rss_mb() -> float:
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))


def gpu_peak_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return 0.0


def rmse_from_mse(mse: float) -> float:
    return float(math.sqrt(mse))


def minmax_to_pm1(x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    x_min = float(x.min())
    x_max = float(x.max())

    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min) * 2.0 - 1.0

    return x, x_min, x_max


def pm1_to_minmax(x_pm1: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    if x_max > x_min:
        return (x_pm1 + 1.0) * 0.5 * (x_max - x_min) + x_min

    return x_pm1


@torch.no_grad()
def spectral_radius_power(W: torch.Tensor, n_iter: int = 30) -> float:
    """Fast spectral-radius estimate used to scale the reservoir."""
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.numel() == 0:
        return 0.0

    v = torch.randn(W.shape[0], device=W.device, dtype=W.dtype)
    v = v / (torch.norm(v) + 1e-12)

    for _ in range(n_iter):
        v = W @ v
        v = v / (torch.norm(v) + 1e-12)

    return float(torch.norm(W @ v).item())


def save_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: str,
    dpi: int,
) -> None:
    n = min(len(y_true), len(y_pred))
    if n <= 1:
        return

    x = np.arange(n)
    err = np.abs(np.asarray(y_true[:n]) - np.asarray(y_pred[:n]))

    mse = float(np.mean(err ** 2))
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

    fig, ax = plt.subplots(figsize=(13.0, 4.8))
    ax.plot(x, y_true[:n], label="True", linewidth=2.2, color="#1f77b4")
    ax.plot(x, y_pred[:n], label="Predicted", linewidth=2.0, alpha=0.92, color="#d62728")
    ax.fill_between(x, y_true[:n], y_pred[:n], color="#d62728", alpha=0.08, linewidth=0)

    ax.set_title(f"{title} | RMSE={rmse:.4g} | MSE={mse:.4g}")
    ax.set_xlabel("Test time index")
    ax.set_ylabel("Value")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_error_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: str,
    dpi: int,
    max_points: int = 1200,
) -> None:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

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

    fig, ax = plt.subplots(figsize=(13.0, 2.8))

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
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


@dataclass
class DeepESNConfig:
    # Model hyperparameters
    multi_step: int
    tau: int
    n_layers: int
    leaky_rate: float
    spectral_radius: float
    sparsity_zero_frac: float
    ridge_alpha: float
    input_scaling: float
    inter_layer_scaling: float
    bias_scaling: float

    # Runtime options
    device: str
    dtype: torch.dtype

    # Artifact options
    plot_first_n: int
    plot_dpi: int

    def __post_init__(self) -> None:
        if self.multi_step <= 0:
            raise ValueError("multi_step must be positive.")
        if self.tau <= 0:
            raise ValueError("tau must be positive.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if not 0.0 < self.leaky_rate <= 1.0:
            raise ValueError("leaky_rate must be in (0, 1].")
        if not 0.0 <= self.sparsity_zero_frac < 1.0:
            raise ValueError("sparsity_zero_frac must be in [0, 1).")
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative.")
        if self.plot_first_n <= 0:
            raise ValueError("plot_first_n must be positive.")
        if self.plot_dpi <= 0:
            raise ValueError("plot_dpi must be positive.")

    @property
    def torch_device(self) -> torch.device:
        return torch.device(str(self.device))


class DeepESN:
    """
    Deep ESN with stacked reservoir layers and multi-step readout.

    Layer-1 update:
      x1 <- (1-a)x1 + a*tanh(Win1*[1;u] + W1*x1 + b1)

    Layer-l update (l > 1):
      xl <- (1-a)xl + a*tanh(Winl*x(l-1) + Wl*xl + bl)

    Readout:
      y_multi <- Wout*[1;u;concat(x1..xL)]
    """

    def __init__(self, res_size: int, seed: int, config: DeepESNConfig):
        if int(res_size) <= 0:
            raise ValueError("res_size must be positive.")

        self.res_size = int(res_size)
        self.seed = int(seed)
        self.cfg = config
        self.device = config.torch_device
        self.dtype = config.dtype

        self.W_layers: List[torch.Tensor] = []
        self.Win_layers: List[torch.Tensor] = []
        self.b_layers: List[torch.Tensor] = []
        self.Wout: Optional[torch.Tensor] = None
        self.one = torch.ones((1, 1), device=self.device, dtype=self.dtype)

        # Saved for optional inspection by experiment scripts.
        self.last_train_len_eff: Optional[int] = None
        self.last_train_mse: Optional[float] = None
        self.last_test_mse: Optional[float] = None
        self.last_y_true: Optional[np.ndarray] = None
        self.last_y_pred: Optional[np.ndarray] = None

    def _zero_states(self) -> List[torch.Tensor]:
        return [
            torch.zeros((self.res_size, 1), device=self.device, dtype=self.dtype)
            for _ in range(self.cfg.n_layers)
        ]

    @torch.no_grad()
    def _step(self, u: torch.Tensor, states: List[torch.Tensor]) -> List[torch.Tensor]:
        new_states: List[torch.Tensor] = []

        for layer in range(self.cfg.n_layers):
            x = states[layer]
            W = self.W_layers[layer]
            b = self.b_layers[layer]

            if layer == 0:
                inp = torch.vstack((self.one, u))
                pre = self.Win_layers[layer] @ inp + W @ x + b
            else:
                # Keep previous-time-step coupling across layers for consistency.
                prev = states[layer - 1]
                pre = self.Win_layers[layer] @ prev + W @ x + b

            x_new = torch.tanh(pre)
            x_upd = (1.0 - self.cfg.leaky_rate) * x + self.cfg.leaky_rate * x_new

            new_states.append(x_upd)

        return new_states

    @torch.no_grad()
    def _collect_states(self, data_pm1: torch.Tensor, init_len: int, run_len: int) -> torch.Tensor:
        states = self._zero_states()

        for t in range(init_len):
            u = data_pm1[t].view(1, 1)
            states = self._step(u, states)

        Xs = torch.zeros(
            (self.cfg.n_layers * self.res_size, run_len),
            device=self.device,
            dtype=self.dtype,
        )

        for t in range(run_len):
            u = data_pm1[init_len + t].view(1, 1)
            states = self._step(u, states)
            Xs[:, t] = torch.vstack(states)[:, 0]

        return Xs

    def _build_targets(self, data_pm1: torch.Tensor, start: int, length: int) -> torch.Tensor:
        Yt = torch.zeros((self.cfg.multi_step, length), device=self.device, dtype=self.dtype)

        for k in range(self.cfg.multi_step):
            s = start + self.cfg.tau * (k + 1)
            e = s + length
            Yt[k, :] = data_pm1[s:e]

        return Yt

    def init_weights(self) -> None:
        set_seed(self.seed)

        self.W_layers = []
        self.Win_layers = []
        self.b_layers = []

        for layer in range(self.cfg.n_layers):
            W = torch.rand(self.res_size, self.res_size, device=self.device, dtype=self.dtype) - 0.5
            zero_mask = torch.rand(self.res_size, self.res_size, device=self.device) < self.cfg.sparsity_zero_frac
            W[zero_mask] = 0.0

            sr_est = spectral_radius_power(W, n_iter=30) + 1e-12
            W *= self.cfg.spectral_radius / sr_est

            if layer == 0:
                Win = (
                    torch.rand(self.res_size, 2, device=self.device, dtype=self.dtype) - 0.5
                ) * self.cfg.input_scaling
            else:
                Win = (
                    torch.rand(self.res_size, self.res_size, device=self.device, dtype=self.dtype) - 0.5
                ) * self.cfg.inter_layer_scaling

            b = (
                torch.rand(self.res_size, 1, device=self.device, dtype=self.dtype) - 0.5
            ) * self.cfg.bias_scaling

            self.W_layers.append(W)
            self.Win_layers.append(Win)
            self.b_layers.append(b)

    def fit_readout(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        XXt = X @ X.T
        rhs = (Y @ X.T).T
        eye = torch.eye(XXt.shape[0], device=self.device, dtype=self.dtype)

        trace_scale = float(torch.trace(XXt).abs().item()) / max(int(XXt.shape[0]), 1)
        trace_scale = max(trace_scale, 1.0)

        base_reg = float(self.cfg.ridge_alpha)

        # Numerical guard: some seeds can make XXt nearly singular in float32.
        self.Wout = None
        for mult in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
            reg = (base_reg + mult * trace_scale) * eye
            try:
                self.Wout = torch.linalg.solve((XXt + reg).T, rhs).T
                return
            except RuntimeError:
                self.Wout = None

        # Last fallback on CPU with stronger diagonal jitter to avoid CUDA SVD failures.
        A_cpu = XXt.detach().to(device="cpu", dtype=torch.float64)
        B_cpu = rhs.detach().to(device="cpu", dtype=torch.float64)
        eye_cpu = torch.eye(A_cpu.shape[0], device="cpu", dtype=torch.float64)

        for mult in (1e-2, 1e-1, 1.0):
            add_reg = base_reg + mult * trace_scale
            try:
                wout_cpu = torch.linalg.solve((A_cpu + add_reg * eye_cpu).T, B_cpu).T
                self.Wout = wout_cpu.to(device=self.device, dtype=self.dtype)
                return
            except RuntimeError:
                continue

        add_reg = base_reg + 1.0 * trace_scale
        wout_cpu = torch.linalg.lstsq((A_cpu + add_reg * eye_cpu).T, B_cpu).solution.T
        self.Wout = wout_cpu.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def train_readout(self, data_pm1: torch.Tensor, init_len: int, train_len: int) -> int:
        max_len = len(data_pm1) - init_len - self.cfg.tau * self.cfg.multi_step
        train_len_eff = min(int(train_len), int(max_len))

        if train_len_eff <= 0:
            raise ValueError("train_len is too short for init_len/tau/multi_step.")

        Yt = self._build_targets(data_pm1, start=init_len, length=train_len_eff)
        Xs = self._collect_states(data_pm1, init_len=init_len, run_len=train_len_eff)

        X = torch.zeros(
            (1 + 1 + self.cfg.n_layers * self.res_size, train_len_eff),
            device=self.device,
            dtype=self.dtype,
        )

        for t in range(train_len_eff):
            u = data_pm1[init_len + t].view(1, 1)
            X[:, t] = torch.vstack((self.one, u, Xs[:, t].view(-1, 1)))[:, 0]

        self.fit_readout(X, Yt)
        self.last_train_len_eff = int(train_len_eff)

        return train_len_eff

    @torch.no_grad()
    def eval_train_block(self, data_pm1: torch.Tensor, init_len: int, train_len: int) -> float:
        if self.Wout is None:
            raise ValueError("Call train_readout() before eval_train_block().")

        max_len = len(data_pm1) - init_len - self.cfg.tau * self.cfg.multi_step
        train_len_eff = min(int(train_len), int(max_len))

        if train_len_eff <= 0:
            raise ValueError("train_len is too short for init_len/tau/multi_step.")

        target = self._build_targets(data_pm1, start=init_len, length=train_len_eff)
        Xs = self._collect_states(data_pm1, init_len=init_len, run_len=train_len_eff)

        X = torch.zeros(
            (1 + 1 + self.cfg.n_layers * self.res_size, train_len_eff),
            device=self.device,
            dtype=self.dtype,
        )

        for t in range(train_len_eff):
            u = data_pm1[init_len + t].view(1, 1)
            X[:, t] = torch.vstack((self.one, u, Xs[:, t].view(-1, 1)))[:, 0]

        Y = self.Wout @ X

        return float(torch.mean((target - Y) ** 2))

    @torch.no_grad()
    def eval_test_block(
        self,
        data_pm1: torch.Tensor,
        init_len: int,
        train_len: int,
        test_len: int,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        if self.Wout is None:
            raise ValueError("Call train_readout() before eval_test_block().")

        start = init_len + train_len
        max_len = len(data_pm1) - start - self.cfg.tau * self.cfg.multi_step
        test_len_eff = min(int(test_len), int(max_len))

        if test_len_eff <= 0:
            raise ValueError("test_len is too short for init_len/train_len/tau/multi_step.")

        target = self._build_targets(data_pm1, start=start, length=test_len_eff)

        states = self._zero_states()
        for t in range(start):
            u = data_pm1[t].view(1, 1)
            states = self._step(u, states)

        Y = torch.zeros((self.cfg.multi_step, test_len_eff), device=self.device, dtype=self.dtype)

        for t in range(test_len_eff):
            u = data_pm1[start + t].view(1, 1)
            states = self._step(u, states)

            feat = torch.vstack((self.one, u, torch.vstack(states)))
            y_multi = self.Wout @ feat

            Y[:, t] = y_multi[:, 0]

        mse_multi = float(torch.mean((target - Y) ** 2))

        y_true_1 = target[0, :].detach().cpu().numpy()
        y_pred_1 = Y[0, :].detach().cpu().numpy()

        return mse_multi, y_true_1, y_pred_1

    def run(
        self,
        series_raw_cpu: torch.Tensor,
        init_len: int,
        train_len: int,
        test_len: int,
        out_dir: Optional[str],
        dataset_name: str,
    ) -> Dict:
        """Train once, evaluate train/test, and optionally save artifacts."""
        if torch.cuda.is_available() and str(self.cfg.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        cpu0 = cpu_rss_mb()

        data_pm1, dmin, dmax = minmax_to_pm1(
            series_raw_cpu.clone().to(self.cfg.torch_device, dtype=self.cfg.dtype)
        )

        self.init_weights()

        train_len_eff = self.train_readout(
            data_pm1,
            init_len=init_len,
            train_len=train_len,
        )

        train_mse = self.eval_train_block(
            data_pm1,
            init_len=init_len,
            train_len=train_len_eff,
        )

        test_mse, y_true_pm1, y_pred_pm1 = self.eval_test_block(
            data_pm1,
            init_len=init_len,
            train_len=train_len_eff,
            test_len=test_len,
        )

        runtime_ms = (time.time() - t0) * 1000.0
        cpu1 = cpu_rss_mb()
        gmem = gpu_peak_mb()

        y_true = pm1_to_minmax(y_true_pm1, dmin, dmax)
        y_pred = pm1_to_minmax(y_pred_pm1, dmin, dmax)

        self.last_train_mse = float(train_mse)
        self.last_test_mse = float(test_mse)
        self.last_y_true = y_true
        self.last_y_pred = y_pred

        row = {
            "dataset": dataset_name,
            "reservoir": int(self.res_size),
            "seed": int(self.seed),
            "n_layers": int(self.cfg.n_layers),
            "total_nodes": int(self.cfg.n_layers * self.res_size),
            "train_mse": float(train_mse),
            "train_rmse": rmse_from_mse(train_mse),
            "test_mse": float(test_mse),
            "test_rmse": rmse_from_mse(test_mse),
            "total_time_ms": float(runtime_ms),
            "total_time_s": float(runtime_ms / 1000.0),
            "gpu_peak_mem_mb": float(gmem),
            "cpu_rss_mb": float(cpu1),
            "cpu_rss_delta_mb": float(cpu1 - cpu0),
            "horizon": int(self.cfg.multi_step),
            "tau": int(self.cfg.tau),
            "leaky_rate": float(self.cfg.leaky_rate),
            "spectral_radius": float(self.cfg.spectral_radius),
            "sparsity_zero_frac": float(self.cfg.sparsity_zero_frac),
            "ridge_alpha": float(self.cfg.ridge_alpha),
            "input_scaling": float(self.cfg.input_scaling),
            "inter_layer_scaling": float(self.cfg.inter_layer_scaling),
            "bias_scaling": float(self.cfg.bias_scaling),
            "dtype": str(self.cfg.dtype).replace("torch.", ""),
            "device": str(self.cfg.device),
        }

        if out_dir:
            ensure_dir(out_dir)

            pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
                os.path.join(out_dir, "test_pred_1step.csv"),
                index=False,
                float_format="%.8f",
            )

            save_plot(
                y_true,
                y_pred,
                os.path.join(out_dir, "test_true_vs_pred_full.png"),
                title=(
                    f"Deep ESN | L={self.cfg.n_layers} | RES={self.res_size} "
                    f"| seed={self.seed} | test 1-step"
                ),
                dpi=self.cfg.plot_dpi,
            )

            n = min(self.cfg.plot_first_n, len(y_true))
            save_plot(
                y_true[:n],
                y_pred[:n],
                os.path.join(out_dir, "test_true_vs_pred_firstN.png"),
                title=(
                    f"Deep ESN | L={self.cfg.n_layers} | RES={self.res_size} "
                    f"| seed={self.seed} | first {n} 1-step"
                ),
                dpi=self.cfg.plot_dpi,
            )

            save_error_heatmap(
                y_true,
                y_pred,
                os.path.join(out_dir, "test_error_heatmap.png"),
                title=f"Deep ESN Error | L={self.cfg.n_layers} | RES={self.res_size} | seed={self.seed}",
                dpi=self.cfg.plot_dpi,
            )

            pd.DataFrame([row]).to_csv(
                os.path.join(out_dir, "metrics.csv"),
                index=False,
                float_format="%.8f",
            )

        return row