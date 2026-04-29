# -*- coding: utf-8 -*-
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
from matplotlib import pyplot as plt

from src.utils.data_cache import hydrate_esn_series


def identity(x):
    # Default output activation: keep the readout linear.
    return x


@torch.no_grad()
def spectral_radius_power(W: torch.Tensor, n_iter: int = 30) -> float:
    """Fast spectral-radius estimate used to scale the reservoir."""
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.numel() == 0:
        return 0.0

    x = torch.randn(W.shape[0], device=W.device, dtype=W.dtype)
    x = x / (torch.norm(x) + 1e-12)

    for _ in range(n_iter):
        x = W @ x
        x = x / (torch.norm(x) + 1e-12)

    return float(torch.norm(W @ x).item())


class EchoStateNetwork:
    """ESN for 1D series with optional feedback and stacked multi-horizon targets."""

    def __init__(
        self,
        trainLen: int,
        testLen: int,
        initLen: int,
        tau: int,
        resSize: int,
        inSize: int,
        outSize: int,
        a: float,
        spectral_radius: float,
        data_path: Optional[str],
        use_ridge: bool,
        ridge_alpha: float,
        sparsity: float,
        activation: Callable,
        output_activation: Callable,
        normalize_states: bool,
        use_feedback: bool,
        store_states: bool,
        input_scaling: float,
        bias_scaling: float,
        feedback_scaling: float,
        multi_step: int,
        seed: Optional[int],
        device: str,
    ):
        self.logger = logging.getLogger("main.EchoStateNetwork")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)

        # Static experiment settings.
        self.trainLen = int(trainLen)
        self.testLen = int(testLen)
        self.initLen = int(initLen)
        self.tau = int(tau)
        self.resSize = int(resSize)
        self.inSize = int(inSize)
        self.outSize = int(outSize)
        self.a = float(a)
        self.spectral_radius = float(spectral_radius)
        self.data_path = data_path
        self.use_ridge = bool(use_ridge)
        self.ridge_alpha = float(ridge_alpha)
        self.sparsity = float(sparsity)
        self.activation = activation
        self.output_activation = output_activation
        self.normalize_states = bool(normalize_states)
        self.use_feedback = bool(use_feedback)
        self.store_states = bool(store_states)
        self.input_scaling = float(input_scaling)
        self.bias_scaling = float(bias_scaling)
        self.feedback_scaling = float(feedback_scaling)
        self.multi_step = max(1, int(multi_step))
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = torch.float32 if self.device.type == "cuda" else torch.float64

        # Runtime values are filled after data loading / training.
        self.data = None
        self.data_min = None
        self.data_max = None
        self.reservoir_states = None
        self.W = None
        self.Win = None
        self.Wfb = None
        self.Wout = None
        self.output_bias = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.one = torch.ones((1, 1), device=self.device, dtype=self.dtype)
        self.zero_scalar = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        self._validate_params()
        self._set_seed(self.seed)

        self.logger.info(
            f"ESN init: resSize={self.resSize}, a={self.a}, rho={self.spectral_radius}, "
            f"sparsity={self.sparsity}, feedback={self.use_feedback}, "
            f"multi_step={self.multi_step}, device={self.device}"
        )

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return

        # Keep reservoir generation and readout fitting reproducible.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.logger.info(f"Seed set to {seed}")

    def _validate_params(self) -> None:
        # Fail early on configuration errors so later tensor code stays simple.
        if self.resSize <= 0:
            raise ValueError("resSize must be positive")
        if min(self.trainLen, self.testLen, self.initLen) < 0:
            raise ValueError("trainLen/testLen/initLen must be non-negative")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if not 0 < self.a <= 1:
            raise ValueError("Leaky rate 'a' must be in (0, 1]")
        if not 0 <= self.sparsity < 1:
            raise ValueError("sparsity must be in [0, 1)")
        if self.data_path and not os.path.exists(self.data_path):
            raise ValueError(f"Data file {self.data_path} not found")

    def _preprocess_data(self, data: torch.Tensor) -> torch.Tensor:
        if torch.isnan(data).any():
            self.logger.warning("NaNs detected; replacing with 0.")
            data = torch.nan_to_num(data, nan=0.0)

        # Store min/max once so predictions can be mapped back later.
        self.data_min = float(data.min())
        self.data_max = float(data.max())

        # ESNs are easier to train when the driven signal stays in a stable range.
        if self.data_max > self.data_min:
            data = (data - self.data_min) / (self.data_max - self.data_min) * 2 - 1
        else:
            self.logger.warning("Data has zero range; skipping normalization.")

        return data.to(device=self.device, dtype=self.dtype)

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        # Undo the [-1, 1] scaling used during training and testing.
        if self.data_max is None or self.data_min is None or self.data_max == self.data_min:
            return x
        return (x + 1.0) * 0.5 * (self.data_max - self.data_min) + self.data_min

    def load_data(self) -> torch.Tensor:
        if self.data is not None:
            return self.data

        if not self.data_path or not self.data_path.endswith(".csv"):
            raise ValueError("Provide a .csv file path in data_path.")

        # Cache the series so repeated train/test calls do not re-read the CSV.
        self.logger.info(f"Loading data from {self.data_path}")
        raw = pd.read_csv(self.data_path).values.flatten()
        self.data = self._preprocess_data(torch.tensor(raw, dtype=self.dtype))

        self.logger.info(
            f"Data loaded: N={len(self.data)}, mean={self.data.mean():.4f}, std={self.data.std():.4f}"
        )
        return self.data

    def generate_reservoir(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self.logger.info("Generating reservoir...")

        # Random recurrent matrix before sparsification and scaling.
        W = torch.rand(self.resSize, self.resSize, device=self.device, dtype=self.dtype) - 0.5
        W[torch.rand(self.resSize, self.resSize, device=self.device) < self.sparsity] = 0.0

        # We rescale once here so later experiments start from a comparable dynamical regime.
        rho = spectral_radius_power(W, n_iter=30) + 1e-12
        self.W = W * (self.spectral_radius / rho)

        self.Win = (
            torch.rand(self.resSize, 1 + self.inSize, device=self.device, dtype=self.dtype) - 0.5
        )
        self.Win[:, :1] *= self.bias_scaling
        self.Win[:, 1:] *= self.input_scaling

        self.Wfb = (
            (torch.rand(self.resSize, self.outSize, device=self.device, dtype=self.dtype) - 0.5)
            * self.feedback_scaling
            if self.use_feedback
            else None
        )

        self.logger.info(
            f"Spectral radius (estimated post-scale): {spectral_radius_power(self.W, 30):.6f}"
        )
        return self.W, self.Win, self.Wfb

    def _feedback_term(self, y_prev: torch.Tensor) -> torch.Tensor:
        # Feedback is optional; when disabled this contributes zero.
        if self.use_feedback and self.Wfb is not None:
            return self.Wfb @ y_prev
        return self.zero_scalar

    def _update_state(self, u: torch.Tensor, x: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
        # Standard leaky-ESN state update.
        pre = self.Win @ torch.vstack((self.one, u)) + self.W @ x + self._feedback_term(y_prev)
        x_new = (1 - self.a) * x + self.a * self.activation(pre)

        if self.normalize_states:
            # Optional normalization can help keep large states bounded.
            x_new /= (torch.norm(x_new) + 1e-8)

        return x_new

    def _feature_vector(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # The readout sees a bias, the current input, and the current reservoir state.
        return torch.vstack((self.one, u, x))

    def _build_targets(self, data: torch.Tensor, start: int, length: int) -> torch.Tensor:
        # Each output block corresponds to one forecast horizon.
        Yt = torch.zeros(
            (self.outSize * self.multi_step, length),
            device=self.device,
            dtype=self.dtype,
        )

        for step in range(self.multi_step):
            s = start + self.tau * (step + 1)
            e = s + length

            if e > len(data):
                raise ValueError(f"Target slice [{s}:{e}] exceeds data length {len(data)}")

            Yt[
                step * self.outSize:(step + 1) * self.outSize,
                :,
            ] = data[s:e].reshape(self.outSize, -1)

        return Yt

    def _teacher_feedback(self, data: torch.Tensor, t: int) -> Optional[torch.Tensor]:
        # During training, feedback uses the known future target at the chosen delay.
        if not self.use_feedback or (t + self.tau) >= len(data):
            return None
        return data[t + self.tau].reshape(self.outSize, 1)

    def _warm_state(self, data: torch.Tensor, end_t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Replay from the start so the test state matches the actual driven trajectory.
        x = torch.zeros((self.resSize, 1), device=self.device, dtype=self.dtype)
        y_prev = torch.zeros((self.outSize, 1), device=self.device, dtype=self.dtype)

        for t in range(end_t):
            u = data[t].reshape(self.inSize, 1)
            x = self._update_state(u, x, y_prev)

            y_next = self._teacher_feedback(data, t)
            if y_next is not None:
                y_prev = y_next

        return x, y_prev

    def train_esn(self, data: torch.Tensor) -> torch.Tensor:
        self.logger.info(f"Training ESN (multi_step={self.multi_step})...")

        # Trim the requested train window so every horizon still fits inside the series.
        max_len = len(data) - self.initLen - self.tau * self.multi_step
        effective_trainLen = min(self.trainLen, max_len)

        if effective_trainLen <= 0:
            raise ValueError("Training length too short for given initLen/tau/multi_step.")

        self.trainLen = effective_trainLen
        self.logger.info(f"effective_trainLen={effective_trainLen}")

        X = torch.zeros(
            (1 + self.inSize + self.resSize, effective_trainLen),
            device=self.device,
            dtype=self.dtype,
        )

        if self.store_states:
            # These states are reused later by the pruning code.
            self.reservoir_states = torch.zeros(
                (effective_trainLen, self.resSize),
                device=self.device,
                dtype=self.dtype,
            )

        # Each column contains all requested forecast horizons for the same input time.
        Yt = self._build_targets(data, start=self.initLen, length=effective_trainLen)

        # Washout the reservoir before collecting training features.
        x, y_prev = self._warm_state(data, self.initLen)

        for t in range(effective_trainLen):
            idx = self.initLen + t
            u = data[idx].reshape(self.inSize, 1)

            x = self._update_state(u, x, y_prev)

            # One training column = bias + current input + current state.
            X[:, t] = self._feature_vector(u, x)[:, 0]

            if self.store_states:
                self.reservoir_states[t, :] = x.ravel()

            y_next = self._teacher_feedback(data, idx)
            if y_next is not None:
                y_prev = y_next

        # Closed-form ridge keeps the readout simple and fast to retrain during pruning.
        reg = self.ridge_alpha if self.use_ridge else 0.0
        gram = X @ X.T
        eye = torch.eye(X.shape[0], device=self.device, dtype=self.dtype)

        trace_scale = float(torch.trace(gram).abs().item()) / max(int(gram.shape[0]), 1)
        trace_scale = max(trace_scale, 1.0)

        rhs = (Yt @ X.T).T

        # Numerical guard: fall back to stronger jitter when solve is singular.
        self.Wout = None
        for mult in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
            add_reg = float(reg) + mult * trace_scale
            try:
                self.Wout = torch.linalg.solve((gram + add_reg * eye).T, rhs).T
                break
            except RuntimeError:
                self.Wout = None

        if self.Wout is None:
            # Last fallback on CPU with stronger diagonal jitter to avoid CUDA SVD failures.
            A_cpu = gram.detach().to(device="cpu", dtype=torch.float64)
            B_cpu = rhs.detach().to(device="cpu", dtype=torch.float64)
            eye_cpu = torch.eye(A_cpu.shape[0], device="cpu", dtype=torch.float64)

            solved = False
            for mult in (1e-2, 1e-1, 1.0):
                add_reg = float(reg) + mult * trace_scale
                try:
                    wout_cpu = torch.linalg.solve((A_cpu + add_reg * eye_cpu).T, B_cpu).T
                    self.Wout = wout_cpu.to(device=self.device, dtype=self.dtype)
                    solved = True
                    break
                except RuntimeError:
                    continue

            if not solved:
                add_reg = float(reg) + 1.0 * trace_scale
                wout_cpu = torch.linalg.lstsq((A_cpu + add_reg * eye_cpu).T, B_cpu).solution.T
                self.Wout = wout_cpu.to(device=self.device, dtype=self.dtype)

        # Bias is already part of X, so there is no separate learned intercept.
        self.output_bias = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.logger.info(f"Training done: Wout shape={self.Wout.shape}")

        return self.Wout

    def _predict_readout(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Readout uses the same feature layout as training.
        z = self.Wout @ self._feature_vector(u, x) + self.output_bias
        return self.output_activation(z)

    def test_esn(
        self,
        data: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        self.logger.info("Testing ESN...")

        test_start = self.initLen + self.trainLen
        effective_testLen = min(self.testLen, len(data) - test_start - self.tau * self.multi_step)

        if effective_testLen <= 0:
            raise ValueError("Test length too short for given initLen/trainLen/tau/multi_step.")

        # Build targets in the same normalized space used during training.
        target = self._build_targets(data, start=test_start, length=effective_testLen)

        if initial_state is None:
            # Default path: reconstruct the test-start state from the driven history.
            x, y_prev = self._warm_state(data, test_start)
        else:
            # Optional path: caller can inject a custom starting state.
            x = initial_state.clone()
            y_prev = torch.zeros((self.outSize, 1), device=self.device, dtype=self.dtype)

        # Test rollout is autoregressive: feedback comes from the model's own prediction.
        Y = torch.zeros(
            (self.outSize * self.multi_step, effective_testLen),
            device=self.device,
            dtype=self.dtype,
        )

        for t in range(effective_testLen):
            u = data[test_start + t].reshape(self.inSize, 1)

            x = self._update_state(u, x, y_prev)
            y = self._predict_readout(u, x)

            Y[:, t] = y.ravel()

            if self.use_feedback:
                y_prev = y[:self.outSize].reshape(self.outSize, 1)

        test_mse = float(torch.mean((target - Y) ** 2))
        self.logger.info(f"Test done: MSE={test_mse:.8f}")

        return Y, test_mse

    def err(
        self,
        data: torch.Tensor,
        print_mse: bool = True,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        if self.W is None or self.Win is None or self.Wout is None:
            raise ValueError("Call generate_reservoir() and train_esn() first.")

        # Rebuild the training targets so train and test metrics are reported consistently.
        target_train = self._build_targets(data, start=self.initLen, length=self.trainLen)

        X_train = torch.zeros(
            (1 + self.inSize + self.resSize, self.trainLen),
            device=self.device,
            dtype=self.dtype,
        )

        # Rebuilding features here makes the reported train error match the current reservoir exactly.
        x, y_prev = self._warm_state(data, self.initLen)

        for t in range(self.trainLen):
            idx = self.initLen + t
            u = data[idx].reshape(self.inSize, 1)

            x = self._update_state(u, x, y_prev)
            X_train[:, t] = self._feature_vector(u, x)[:, 0]

            y_next = self._teacher_feedback(data, idx)
            if y_next is not None:
                y_prev = y_next

        Y_train = self.output_activation(self.Wout @ X_train + self.output_bias)
        train_mse = float(torch.mean((target_train - Y_train) ** 2))

        self.logger.info(f"Train MSE (normalized): {train_mse:.8f}")

        # Test metrics come from the same public evaluation path used elsewhere.
        Y_test, test_mse = self.test_esn(data, initial_state)

        if print_mse:
            print(f"Train MSE: {train_mse:.8f} | Test MSE: {test_mse:.8f}")

        return Y_test, train_mse, test_mse


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cpu_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss / (1024 * 1024))


def _gpu_peak_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return 0.0


def _rmse_from_mse(mse: float) -> float:
    return float(np.sqrt(mse))


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


def _prediction_arrays(esn: EchoStateNetwork, y_test: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    target_test = compute_test_target(esn, esn.data)

    y_true = esn.denorm(target_test).detach().cpu().numpy()
    y_pred = esn.denorm(y_test).detach().cpu().numpy()

    y_true = y_true[0, :] if y_true.ndim == 2 else y_true.reshape(-1)
    y_pred = y_pred[0, :] if y_pred.ndim == 2 else y_pred.reshape(-1)

    return y_true, y_pred


@dataclass
class BaseESNRunConfig:
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
    device: str


class BaseESNRunner:
    """Simple train/eval runner for base ESN with optional CSV/PNG saving."""

    def __init__(
        self,
        res_size: int,
        seed: int,
        init_len: int,
        train_len: int,
        test_len: int,
        config: BaseESNRunConfig,
        data_path: Optional[str],
    ):
        self.res_size = int(res_size)
        self.seed = int(seed)
        self.init_len = int(init_len)
        self.train_len = int(train_len)
        self.test_len = int(test_len)
        self.cfg = config
        self.data_path = data_path

        # Saved for optional inspection by experiment scripts.
        self.last_model = None
        self.last_y_test = None

    def _build_model(self) -> EchoStateNetwork:
        return EchoStateNetwork(
            trainLen=self.train_len,
            testLen=self.test_len,
            initLen=self.init_len,
            tau=self.cfg.tau,
            resSize=self.res_size,
            inSize=1,
            outSize=1,
            a=self.cfg.leaky_rate,
            spectral_radius=self.cfg.spectral_radius,
            data_path=self.data_path,
            use_ridge=True,
            ridge_alpha=self.cfg.ridge_alpha,
            sparsity=self.cfg.sparsity,
            activation=torch.tanh,
            output_activation=identity,
            normalize_states=self.cfg.normalize_states,
            use_feedback=self.cfg.use_feedback,
            store_states=False,
            input_scaling=self.cfg.input_scaling,
            bias_scaling=self.cfg.bias_scaling,
            feedback_scaling=self.cfg.feedback_scaling,
            multi_step=self.cfg.multi_step,
            seed=self.seed,
            device=self.cfg.device,
        )

    def run(self, series_raw_cpu: Optional[torch.Tensor], out_dir: Optional[str], dataset_name: str) -> Dict:
        if torch.cuda.is_available() and str(self.cfg.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        cpu0 = _cpu_rss_mb()

        model = self._build_model()

        if series_raw_cpu is None:
            model.load_data()
        else:
            hydrate_esn_series(model, series_raw_cpu)

        model.generate_reservoir()
        model.train_esn(model.data)

        y_test, train_mse, test_mse = model.err(model.data, print_mse=False)

        runtime_ms = (time.time() - t0) * 1000.0
        cpu1 = _cpu_rss_mb()
        gmem = _gpu_peak_mb()

        row = {
            "dataset": dataset_name,
            "reservoir": int(self.res_size),
            "seed": int(self.seed),
            "train_mse": float(train_mse),
            "train_rmse": _rmse_from_mse(float(train_mse)),
            "test_mse": float(test_mse),
            "test_rmse": _rmse_from_mse(float(test_mse)),
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
            "device": str(self.cfg.device),
        }

        self.last_model = model
        self.last_y_test = y_test

        if out_dir:
            _ensure_dir(out_dir)

            y_true, y_pred = _prediction_arrays(model, y_test)

            pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
                os.path.join(out_dir, "test_pred_1step.csv"),
                index=False,
                float_format="%.8f",
            )

            plot_real_vs_prediction(
                y_true,
                y_pred,
                os.path.join(out_dir, "test_true_vs_pred.png"),
                title=f"Base ESN | RES={self.res_size} | seed={self.seed}",
            )

            plot_prediction_error_heatmap(
                y_true,
                y_pred,
                os.path.join(out_dir, "test_error_heatmap.png"),
                title=f"Base ESN Error | RES={self.res_size} | seed={self.seed}",
            )

            pd.DataFrame([row]).to_csv(
                os.path.join(out_dir, "metrics.csv"),
                index=False,
                float_format="%.8f",
            )

        return row


def plot_real_vs_prediction(y_true, y_pred, save_path: str, title: str, max_points: int = 500) -> None:
    y_true = _clean_series(y_true)
    y_pred = _clean_series(y_pred)

    n = int(min(len(y_true), len(y_pred), max_points))
    if n <= 1:
        return

    x = np.arange(n)
    abs_error = np.abs(y_true[:n] - y_pred[:n])

    mse = float(np.mean(abs_error**2))
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


# Backward-compatible aliases for older experiment scripts.
OldESNRunConfig = BaseESNRunConfig
OldESNRunner = BaseESNRunner


__all__ = [
    "identity",
    "spectral_radius_power",
    "EchoStateNetwork",
    "BaseESNRunConfig",
    "BaseESNRunner",
    "OldESNRunConfig",
    "OldESNRunner",
    "compute_test_target",
    "plot_real_vs_prediction",
    "plot_prediction_error_heatmap",
]