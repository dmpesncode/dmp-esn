"""
Dynamical Mode Pruning (DMP) for Echo State Networks.

This module implements a reviewer-facing / anonymous-release version of the
DMP pruning pipeline.

Core pipeline:
    1. Train a base ESN.
    2. Collect a teacher-forced reservoir trajectory.
    3. Build the trajectory-averaged Jacobian Gram matrix.
    4. Score neurons by participation in dominant dynamical modes.
    5. Prune low-score neurons.
    6. Optionally stabilize the pruned reservoir.
    7. Retrain only the readout.

Default setting:
    score_mode="dynamic_only"

This keeps the released method aligned with the paper description. Optional
task-aware variants are included but clearly separated.
"""

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import psutil
import torch

from src.sota_models.base_esn import EchoStateNetwork, identity
from src.utils.data_cache import hydrate_esn_series


# ==============================================================
# Logging
# ==============================================================

Path("logs").mkdir(exist_ok=True)

logger = logging.getLogger("DMP")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fh = logging.FileHandler("logs/dmp.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)


# ==============================================================
# Small utilities
# ==============================================================

def cpu_rss_mb() -> float:
    """Return current process resident memory in MB."""
    return float(psutil.Process().memory_info().rss / (1024 * 1024))


def gpu_peak_mb() -> float:
    """Return peak allocated CUDA memory in MB."""
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return 0.0


def rmse_from_mse(mse: float) -> float:
    """Convert MSE to RMSE."""
    return float(math.sqrt(mse))


def spectral_radius(W: torch.Tensor) -> float:
    """
    Compute the exact spectral radius of a square matrix.

    This is used only for stabilization / safety checks. For large reservoirs,
    this can be expensive because it performs an eigendecomposition.
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square.")

    if W.shape[0] == 0:
        return 0.0

    evs = torch.linalg.eigvals(W.to(torch.complex128))
    return float(torch.max(torch.abs(evs)).real.item())


def rescale_reservoir_to_target_rho(
    W: torch.Tensor,
    target_rho: float = 0.95,
) -> torch.Tensor:
    """
    Rescale a reservoir matrix to a target spectral radius.

    This is a fallback stabilization step. DMP primarily uses Jacobian-based
    statistics, but spectral-radius capping is useful to avoid unstable
    post-pruning dynamics.
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square.")

    if W.shape[0] == 0:
        return W.clone()

    current_rho = spectral_radius(W)

    if current_rho <= 1e-12:
        return W.clone()

    return W * (float(target_rho) / current_rho)


# ==============================================================
# Trajectory collection
# ==============================================================

@dataclass
class TrajectoryBundle:
    """
    Reservoir signals collected from a teacher-forced ESN rollout.

    Attributes
    ----------
    preacts:
        Pre-activation vectors z_t before tanh.

    states:
        Reservoir states x_t after leaky update.

    state_drives:
        tanh derivatives, f'(z_t) = 1 - tanh(z_t)^2.
        These are used to construct the analytic Jacobian.

    design_states:
        Optional stored design matrix states from the ESN training routine.
    """

    preacts: torch.Tensor
    states: torch.Tensor
    state_drives: torch.Tensor
    design_states: Optional[torch.Tensor] = None


@torch.no_grad()
def collect_teacher_forced_trajectory(
    esn: EchoStateNetwork,
    data: torch.Tensor,
) -> TrajectoryBundle:
    """
    Collect the teacher-forced reservoir trajectory used for training.

    Important assumptions
    ---------------------
    1. The reservoir activation is tanh.
    2. normalize_states=False.

    If state normalization is enabled, the closed-form Jacobian used by DMP
    no longer matches the implemented reservoir update. Therefore this
    implementation explicitly requires normalize_states=False.
    """
    if bool(esn.normalize_states):
        raise ValueError(
            "DMP requires normalize_states=False so the analytic Jacobian "
            "matches the implemented reservoir dynamics."
        )

    # This code assumes tanh dynamics because the derivative is hard-coded as
    # 1 - tanh(z)^2. If another activation is used, the Jacobian must change.
    if esn.activation is not torch.tanh:
        logger.warning(
            "DMP currently assumes tanh activation. Make sure the derivative "
            "matches the activation used in the ESN."
        )

    max_len = len(data) - esn.initLen - esn.tau * esn.multi_step
    effective_train_len = min(esn.trainLen, max_len)

    if effective_train_len <= 0:
        raise ValueError("Training length is too short for trajectory collection.")

    preacts = torch.zeros(
        (effective_train_len, esn.resSize),
        device=esn.device,
        dtype=esn.dtype,
    )
    states = torch.zeros_like(preacts)
    derivatives = torch.zeros_like(preacts)

    # Warm up the reservoir exactly as in the ESN implementation.
    x, y_prev = esn._warm_state(data, esn.initLen)

    for t in range(effective_train_len):
        idx = esn.initLen + t
        u = data[idx].reshape(esn.inSize, 1)

        pre = (
            esn.Win @ torch.vstack((esn.one, u))
            + esn.W @ x
            + esn._feedback_term(y_prev)
        )

        tanh_pre = esn.activation(pre)
        x = (1.0 - esn.a) * x + esn.a * tanh_pre

        preacts[t] = pre.ravel()
        states[t] = x.ravel()
        derivatives[t] = (1.0 - tanh_pre.pow(2)).ravel()

        # Teacher forcing: use the true next output when feedback is enabled.
        y_next = esn._teacher_feedback(data, idx)
        if y_next is not None:
            y_prev = y_next

    design_states = None
    if hasattr(esn, "X") and esn.X is not None:
        design_states = esn.X.detach().clone()

    return TrajectoryBundle(
        preacts=preacts,
        states=states,
        state_drives=derivatives,
        design_states=design_states,
    )


# ==============================================================
# Optional task-aware relevance
# ==============================================================

def extract_readout_relevance(esn: EchoStateNetwork) -> Optional[torch.Tensor]:
    """
    Extract optional readout-based neuron relevance.

    This is NOT used in the default DMP setting. It is only used when
    score_mode="hybrid".

    The signal is the magnitude of the trained output weights attached to
    reservoir neurons. Most ESN implementations store reservoir states in the
    last resSize columns of Wout, so we defensively take that slice.
    """
    Wout = getattr(esn, "Wout", None)

    if Wout is None:
        logger.warning("ESN does not expose Wout. Readout relevance is skipped.")
        return None

    Wout = Wout.detach()

    if Wout.ndim == 1:
        Wout = Wout.unsqueeze(0)

    res_size = int(esn.resSize)

    if Wout.shape[1] < res_size:
        logger.warning("Wout has fewer columns than reservoir size. Skipping readout relevance.")
        return None

    state_weights = Wout[:, -res_size:]
    relevance = torch.norm(state_weights, dim=0)

    return relevance


# ==============================================================
# DMP scorer
# ==============================================================

class DynamicalModePruner:
    """
    Score and select ESN reservoir neurons using Dynamical Mode Pruning.

    The main DMP score is based on a trajectory-averaged Jacobian Gram matrix:

        G = (1/T) sum_t J_t^T J_t

    where

        J_t = (1-a) I + a diag(f'(z_t)) W

    For tanh reservoirs:

        f'(z_t) = 1 - tanh(z_t)^2

    A neuron is important if it participates strongly in the dominant
    eigenmodes of G.

    Default score
    -------------
    score_mode="dynamic_only"

        score_i = sum_k lambda_k * V[i, k]^2

    Optional variants
    -----------------
    score_mode="dynamic_occupancy"

        Adds average nonlinear sensitivity.

    score_mode="hybrid"

        Adds both nonlinear sensitivity and readout relevance.

    For paper-aligned anonymous release, use dynamic_only by default.
    """

    def __init__(
        self,
        W: torch.Tensor,
        leaky_rate: float = 0.1,
        energy_tau: float = 0.90,
        gramian_chunk_size: int = 128,
        score_mode: str = "dynamic_only",
        alpha_dynamic: float = 1.0,
        beta_readout: float = 0.0,
        gamma_occupancy: float = 0.0,
        eps: float = 1e-12,
    ):
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be a square matrix.")

        if not 0.0 < float(leaky_rate) <= 1.0:
            raise ValueError("leaky_rate must be in (0, 1].")

        if not 0.0 < float(energy_tau) <= 1.0:
            raise ValueError("energy_tau must be in (0, 1].")

        if int(gramian_chunk_size) <= 0:
            raise ValueError("gramian_chunk_size must be positive.")

        if score_mode not in {"dynamic_only", "dynamic_occupancy", "hybrid"}:
            raise ValueError(
                "score_mode must be one of: dynamic_only, dynamic_occupancy, hybrid"
            )

        self.device = W.device
        self.dtype = torch.float64

        self.W = W.to(device=self.device, dtype=self.dtype)
        self.a = float(leaky_rate)
        self.energy_tau = float(energy_tau)
        self.gramian_chunk_size = int(gramian_chunk_size)
        self.score_mode = str(score_mode)

        self.alpha_dynamic = float(alpha_dynamic)
        self.beta_readout = float(beta_readout)
        self.gamma_occupancy = float(gamma_occupancy)
        self.eps = float(eps)
        self.N = int(W.shape[0])

        # Diagnostics saved for later inspection / paper plots.
        self.last_gramian: Optional[torch.Tensor] = None
        self.last_eigenvalues: Optional[torch.Tensor] = None
        self.last_normalized_eigenvalues: Optional[torch.Tensor] = None
        self.last_cumulative_energy: Optional[torch.Tensor] = None
        self.last_rank: Optional[int] = None
        self.last_dynamic_scores: Optional[torch.Tensor] = None
        self.last_occupancy_scores: Optional[torch.Tensor] = None
        self.last_readout_scores: Optional[torch.Tensor] = None
        self.last_final_scores: Optional[torch.Tensor] = None
        self.last_trace: Optional[float] = None
        self.last_total_energy: Optional[float] = None
        self.last_timesteps: Optional[int] = None

    def _normalize_vector(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Normalize a vector by its maximum value."""
        if x is None:
            return None

        x = x.to(device=self.device, dtype=self.dtype)
        denom = torch.clamp(torch.max(x), min=self.eps)
        return x / denom

    def _compute_expected_gramian(
        self,
        derivatives: torch.Tensor,
        W_current: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the trajectory-averaged Jacobian Gram matrix.

        G = (1/T) sum_t J_t^T J_t

        The computation is chunked over time to reduce peak memory usage.
        """
        D = derivatives.to(device=self.device, dtype=self.dtype)
        T, N = D.shape

        W_use = self.W if W_current is None else W_current.to(
            device=self.device,
            dtype=self.dtype,
        )

        if W_use.shape != (N, N):
            raise ValueError("W_current shape does not match derivative width.")

        I = torch.eye(N, device=self.device, dtype=self.dtype)
        G = torch.zeros((N, N), device=self.device, dtype=self.dtype)

        for start in range(0, T, self.gramian_chunk_size):
            end = min(start + self.gramian_chunk_size, T)
            D_chunk = D[start:end]

            # diag(d_t) @ W is equivalent to row-wise scaling of W.
            DW = D_chunk.unsqueeze(-1) * W_use.unsqueeze(0)

            # J_t = (1-a) I + a diag(f'(z_t)) W
            J = (1.0 - self.a) * I.unsqueeze(0) + self.a * DW

            G += torch.sum(torch.matmul(J.transpose(1, 2), J), dim=0)

        return G / float(T)

    def _select_rank(self, normalized_eigenvalues: torch.Tensor) -> int:
        """
        Select the number of dominant modes needed to preserve energy_tau energy.
        """
        if normalized_eigenvalues.ndim != 1:
            raise ValueError("normalized_eigenvalues must be a vector.")

        if normalized_eigenvalues.numel() == 0:
            return 0

        cumulative = torch.cumsum(normalized_eigenvalues, dim=0)
        idx = torch.where(cumulative >= self.energy_tau)[0]

        if len(idx) == 0:
            return int(normalized_eigenvalues.numel())

        return int(idx[0].item() + 1)

    @torch.no_grad()
    def compute_scores(
        self,
        derivatives: torch.Tensor,
        readout_relevance: Optional[torch.Tensor] = None,
        occupancy_signal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute neuron importance scores.

        Default dynamic-only score:
            score_i = sum_k lambda_k * V[i, k]^2

        Here:
            lambda_k are normalized eigenvalues of G
            V[:, k] are eigenvectors of G

        Squared eigenvector participation is used because it is stable,
        nonnegative, and directly measures modal energy contribution.
        """
        derivatives = derivatives.to(device=self.device, dtype=self.dtype)

        if derivatives.ndim != 2 or derivatives.shape[1] != self.N:
            raise ValueError(f"derivatives must have shape (T, {self.N}).")

        T = int(derivatives.shape[0])

        logger.info(
            "Computing DMP scores over %d timesteps on %s with chunk size %d.",
            T,
            self.device,
            self.gramian_chunk_size,
        )

        G = self._compute_expected_gramian(derivatives=derivatives)

        # G is symmetric positive semidefinite, so eigh is appropriate.
        evals, evecs = torch.linalg.eigh(G)
        evals = torch.clamp(evals, min=0.0)

        # Sort from largest to smallest eigenvalue.
        evals = torch.flip(evals, dims=[0])
        evecs = torch.flip(evecs, dims=[1])

        total_energy = torch.sum(evals)

        if total_energy <= self.eps:
            logger.warning("Near-zero Gramian energy. Returning zero scores.")

            zero = torch.zeros(self.N, device=self.device, dtype=self.dtype)

            self.last_gramian = G.detach().clone()
            self.last_eigenvalues = evals.detach().clone()
            self.last_normalized_eigenvalues = zero.clone()
            self.last_cumulative_energy = zero.clone()
            self.last_rank = 0
            self.last_dynamic_scores = zero.clone()
            self.last_occupancy_scores = None
            self.last_readout_scores = None
            self.last_final_scores = zero.clone()
            self.last_trace = 0.0
            self.last_total_energy = 0.0
            self.last_timesteps = T

            return zero

        normalized_evals = evals / total_energy
        r = self._select_rank(normalized_evals)

        logger.info(
            "Using top %d dynamical modes to preserve %.2f%% Gramian energy.",
            r,
            100.0 * self.energy_tau,
        )

        V_r = evecs[:, :r]
        L_r = normalized_evals[:r]

        # Main DMP score: participation in dominant Jacobian modes.
        dynamic_score = torch.sum(L_r.unsqueeze(0) * (V_r ** 2), dim=1)
        dynamic_score = self._normalize_vector(dynamic_score)

        # Start with the dynamic-only score.
        final_score = torch.clamp(dynamic_score, min=self.eps) ** self.alpha_dynamic

        occupancy_score = None
        readout_score = None

        # Optional: average local nonlinear sensitivity.
        if self.score_mode in {"dynamic_occupancy", "hybrid"}:
            if occupancy_signal is None:
                occupancy_signal = torch.mean(torch.abs(derivatives), dim=0)

            occupancy_score = self._normalize_vector(occupancy_signal)
            final_score = final_score * (
                torch.clamp(occupancy_score, min=self.eps) ** self.gamma_occupancy
            )

        # Optional: task-aware readout relevance.
        if self.score_mode == "hybrid" and readout_relevance is not None:
            readout_score = self._normalize_vector(readout_relevance)
            final_score = final_score * (
                torch.clamp(readout_score, min=self.eps) ** self.beta_readout
            )

        final_score = self._normalize_vector(final_score)

        # Save diagnostics.
        self.last_gramian = G.detach().clone()
        self.last_eigenvalues = evals.detach().clone()
        self.last_normalized_eigenvalues = normalized_evals.detach().clone()
        self.last_cumulative_energy = torch.cumsum(normalized_evals, dim=0).detach().clone()
        self.last_rank = int(r)
        self.last_dynamic_scores = dynamic_score.detach().clone()
        self.last_occupancy_scores = (
            occupancy_score.detach().clone() if occupancy_score is not None else None
        )
        self.last_readout_scores = (
            readout_score.detach().clone() if readout_score is not None else None
        )
        self.last_final_scores = final_score.detach().clone()
        self.last_trace = float(torch.trace(G).item())
        self.last_total_energy = float(total_energy.item())
        self.last_timesteps = T

        return final_score

    @torch.no_grad()
    def select_topk(self, scores: torch.Tensor, target_size: int) -> torch.Tensor:
        """Return sorted indices of the top-k highest-scoring neurons."""
        if scores.ndim != 1 or scores.shape[0] != self.N:
            raise ValueError(f"scores must have shape ({self.N},).")

        if not 1 <= int(target_size) <= self.N:
            raise ValueError(f"target_size must be in [1, {self.N}].")

        _, idx = torch.topk(scores, k=int(target_size), largest=True, sorted=False)
        return torch.sort(idx).values

    @torch.no_grad()
    def match_jacobian_energy_density(
        self,
        W_pruned: torch.Tensor,
        derivatives_pruned: torch.Tensor,
        target_trace_per_neuron: float,
        target_rho: Optional[float] = None,
        max_iter: int = 12,
    ) -> torch.Tensor:
        """
        Rescale the pruned reservoir to match Jacobian energy density.

        Instead of matching the total trace of the original full reservoir,
        this matches the average trace per neuron:

            trace(G_pruned) / N_pruned ≈ trace(G_original) / N_original

        This is safer than matching the full original trace after dimensionality
        reduction, because a smaller reservoir should naturally have smaller
        total Jacobian energy.

        Optionally, the final spectral radius is capped by target_rho.
        """
        if W_pruned.ndim != 2 or W_pruned.shape[0] != W_pruned.shape[1]:
            raise ValueError("W_pruned must be square.")

        if derivatives_pruned.ndim != 2:
            raise ValueError("derivatives_pruned must be a matrix.")

        if derivatives_pruned.shape[1] != W_pruned.shape[0]:
            raise ValueError("derivatives_pruned width must match W_pruned size.")

        if target_trace_per_neuron <= 0.0:
            return W_pruned.clone()

        W_scaled = W_pruned.to(device=self.device, dtype=self.dtype).clone()
        derivatives_pruned = derivatives_pruned.to(device=self.device, dtype=self.dtype)

        n_pruned = int(W_pruned.shape[0])
        target_trace = float(target_trace_per_neuron) * float(n_pruned)

        for _ in range(int(max_iter)):
            Gp = self._compute_expected_gramian(
                derivatives=derivatives_pruned,
                W_current=W_scaled,
            )

            current_trace = float(torch.trace(Gp).item())

            if current_trace <= self.eps:
                break

            ratio = target_trace / current_trace

            # Stop when energy density is already close.
            if abs(1.0 - ratio) < 0.03:
                break

            step = math.sqrt(max(ratio, self.eps))
            W_scaled = W_scaled * step

        # Safety cap: do not let rho(W) exceed the base ESN target.
        if target_rho is not None and target_rho > 0.0:
            current_rho = spectral_radius(W_scaled)

            if current_rho > 1e-12:
                W_scaled = W_scaled * min(1.0, float(target_rho) / current_rho)

        return W_scaled


# ==============================================================
# ESN cloning after pruning
# ==============================================================

def clone_pruned_esn(
    base_esn: EchoStateNetwork,
    keep_idx: torch.Tensor,
    w_override: Optional[torch.Tensor] = None,
) -> EchoStateNetwork:
    """
    Build a pruned ESN by copying only the retained reservoir neurons.

    Only the readout is retrained afterward. Recurrent weights are not
    re-optimized, which keeps the comparison simple and fair.
    """
    pruned_size = int(keep_idx.numel())
    keep_idx = keep_idx.to(device=base_esn.device)

    pruned_esn = EchoStateNetwork(
        trainLen=base_esn.trainLen,
        testLen=base_esn.testLen,
        initLen=base_esn.initLen,
        tau=base_esn.tau,
        resSize=pruned_size,
        inSize=base_esn.inSize,
        outSize=base_esn.outSize,
        a=base_esn.a,
        spectral_radius=base_esn.spectral_radius,
        data_path=base_esn.data_path,
        use_ridge=base_esn.use_ridge,
        ridge_alpha=base_esn.ridge_alpha,
        sparsity=base_esn.sparsity,
        activation=base_esn.activation,
        output_activation=base_esn.output_activation,
        normalize_states=False,
        use_feedback=base_esn.use_feedback,
        store_states=True,
        input_scaling=base_esn.input_scaling,
        bias_scaling=base_esn.bias_scaling,
        feedback_scaling=base_esn.feedback_scaling,
        multi_step=base_esn.multi_step,
        seed=base_esn.seed,
        device=str(base_esn.device),
    )

    pruned_esn.data = base_esn.data
    pruned_esn.data_min = base_esn.data_min
    pruned_esn.data_max = base_esn.data_max

    # Copy recurrent, input, and feedback blocks for kept neurons.
    w_pruned = base_esn.W.index_select(0, keep_idx).index_select(1, keep_idx)

    if w_override is not None:
        w_pruned = w_override.to(device=base_esn.device, dtype=base_esn.dtype)

    pruned_esn.W = w_pruned.to(dtype=base_esn.dtype)
    pruned_esn.Win = base_esn.Win.index_select(0, keep_idx).contiguous()

    if base_esn.Wfb is not None:
        pruned_esn.Wfb = base_esn.Wfb.index_select(0, keep_idx).contiguous()
    else:
        pruned_esn.Wfb = None

    pruned_esn.output_bias = torch.tensor(
        0.0,
        device=base_esn.device,
        dtype=base_esn.dtype,
    )

    return pruned_esn


# ==============================================================
# Configuration
# ==============================================================

@dataclass
class DMPConfig:
    """
    Configuration for Dynamical Mode Pruning.

    The defaults are chosen to keep the released implementation aligned with
    the paper method:
        score_mode="dynamic_only"
        alpha_dynamic=1.0
        beta_readout=0.0
        gamma_occupancy=0.0

    Optional hybrid variants can be enabled for ablation experiments.
    """

    # ESN settings
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

    # DMP settings
    prune_ratio: float
    energy_tau: float = 0.90
    gramian_chunk_size: int = 128

    # Main release setting: dynamic_only
    score_mode: str = "dynamic_only"

    # Exponents for optional score variants
    alpha_dynamic: float = 1.0
    beta_readout: float = 0.0
    gamma_occupancy: float = 0.0

    # Pruning schedule
    progressive: bool = True
    progressive_step_ratio: float = 0.05

    # Stabilization
    match_jacobian_energy_density: bool = True
    fallback_match_spectral_radius: bool = True

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

        if not 0.0 < self.energy_tau <= 1.0:
            raise ValueError("energy_tau must be in (0, 1].")

        if self.gramian_chunk_size <= 0:
            raise ValueError("gramian_chunk_size must be positive.")

        if self.score_mode not in {"dynamic_only", "dynamic_occupancy", "hybrid"}:
            raise ValueError("Invalid score_mode.")

        if not 0.0 < self.progressive_step_ratio < 1.0:
            raise ValueError("progressive_step_ratio must be in (0, 1).")

        if bool(self.normalize_states):
            raise ValueError(
                "DMP requires normalize_states=False so the analytic Jacobian "
                "matches the implemented ESN dynamics."
            )

        # Keep defaults consistent with dynamic_only mode.
        if self.score_mode == "dynamic_only":
            self.alpha_dynamic = 1.0
            self.beta_readout = 0.0
            self.gamma_occupancy = 0.0


# ==============================================================
# Main DMP pipeline
# ==============================================================

class DMPESN:
    """
    End-to-end DMP pipeline for ESN pruning.

    Steps
    -----
    1. Build and train the base ESN.
    2. Collect the teacher-forced trajectory.
    3. Compute DMP scores from the Jacobian Gram matrix.
    4. Prune neurons progressively or in one shot.
    5. Stabilize the pruned reservoir.
    6. Retrain only the output readout.
    """

    def __init__(
        self,
        res_size: int,
        seed: int,
        init_len: int,
        train_len: int,
        test_len: int,
        config: DMPConfig,
        data_path: Optional[str],
    ):
        self.res_size = int(res_size)
        self.seed = int(seed)
        self.init_len = int(init_len)
        self.train_len = int(train_len)
        self.test_len = int(test_len)
        self.cfg = config
        self.data_path = data_path

        self.original_reservoir_size = int(res_size)
        self.final_reservoir_size = int(res_size)

        # Saved state for diagnostics and plots.
        self.last_base_esn: Optional[EchoStateNetwork] = None
        self.last_pruned_esn: Optional[EchoStateNetwork] = None
        self.last_pruner: Optional[DynamicalModePruner] = None
        self.last_scores: Optional[torch.Tensor] = None
        self.last_keep_idx: Optional[torch.Tensor] = None
        self.keep_idx: Optional[torch.Tensor] = None

        self.last_base_y_test: Optional[torch.Tensor] = None
        self.last_pruned_y_test: Optional[torch.Tensor] = None

        self.last_base_train_mse: Optional[float] = None
        self.last_base_test_mse: Optional[float] = None
        self.last_pruned_train_mse: Optional[float] = None
        self.last_pruned_test_mse: Optional[float] = None

        self.last_progressive_sizes = []

    def _build_base_esn(self) -> EchoStateNetwork:
        """Construct the unpruned base ESN."""
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
            normalize_states=False,
            use_feedback=self.cfg.use_feedback,
            store_states=True,
            input_scaling=self.cfg.input_scaling,
            bias_scaling=self.cfg.bias_scaling,
            feedback_scaling=self.cfg.feedback_scaling,
            multi_step=self.cfg.multi_step,
            seed=self.seed,
            device=self.cfg.device,
        )

    def _compute_pruning_schedule(
        self,
        total_size: int,
        final_size: int,
    ) -> list:
        """
        Compute reservoir sizes used during progressive pruning.

        Example:
            total=1000, final=800, step=5%
            schedule = [950, 900, 850, 800]
        """
        if final_size >= total_size:
            return [total_size]

        if not self.cfg.progressive:
            return [final_size]

        step = max(1, int(round(total_size * self.cfg.progressive_step_ratio)))

        sizes = []
        current = total_size

        while current > final_size:
            current = max(final_size, current - step)
            sizes.append(int(current))

        return sizes

    @torch.no_grad()
    def _progressive_prune(
        self,
        base_esn: EchoStateNetwork,
        trajectory: TrajectoryBundle,
        pruner: DynamicalModePruner,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prune the reservoir progressively.

        At each step:
            1. Compute DMP scores on the current subnetwork.
            2. Keep the highest-scoring neurons.
            3. Restrict W and derivatives to the kept neurons.
            4. Optionally stabilize the pruned matrix.

        Returns
        -------
        keep_idx_global:
            Indices of retained neurons in the original reservoir indexing.

        final_W_pruned:
            Final stabilized recurrent matrix.
        """
        N = self.res_size
        target_final_size = max(1, int(round((1.0 - self.cfg.prune_ratio) * N)))
        schedule = self._compute_pruning_schedule(
            total_size=N,
            final_size=target_final_size,
        )

        self.last_progressive_sizes = list(schedule)

        keep_idx_global = torch.arange(N, device=base_esn.device)
        current_W = base_esn.W.detach().clone()
        current_derivatives = trajectory.state_drives.detach().clone()

        # Readout relevance is optional and only used in hybrid mode.
        current_readout = None
        if self.cfg.score_mode == "hybrid":
            current_readout = extract_readout_relevance(base_esn)

        # Compute full-resolution DMP scores once for diagnostics.
        baseline_scores_full = pruner.compute_scores(
            derivatives=trajectory.state_drives.detach().clone(),
            readout_relevance=current_readout,
            occupancy_signal=torch.mean(torch.abs(trajectory.state_drives), dim=0),
        )

        # Compute original Jacobian energy density, not total trace.
        G0 = pruner._compute_expected_gramian(current_derivatives.to(dtype=pruner.dtype))
        target_trace_per_neuron = float(torch.trace(G0).item()) / float(N)

        logger.info("DMP pruning schedule: %s", schedule)
        logger.info("Target Jacobian trace per neuron: %.8f", target_trace_per_neuron)

        for step_id, size_k in enumerate(schedule, start=1):
            sub_pruner = DynamicalModePruner(
                W=current_W,
                leaky_rate=base_esn.a,
                energy_tau=self.cfg.energy_tau,
                gramian_chunk_size=self.cfg.gramian_chunk_size,
                score_mode=self.cfg.score_mode,
                alpha_dynamic=self.cfg.alpha_dynamic,
                beta_readout=self.cfg.beta_readout,
                gamma_occupancy=self.cfg.gamma_occupancy,
            )

            scores = sub_pruner.compute_scores(
                derivatives=current_derivatives,
                readout_relevance=current_readout,
                occupancy_signal=torch.mean(torch.abs(current_derivatives), dim=0),
            )

            keep_local = sub_pruner.select_topk(
                scores=scores,
                target_size=size_k,
            )

            # Restrict current subnetwork to selected neurons.
            keep_idx_global = keep_idx_global.index_select(0, keep_local)
            current_W = current_W.index_select(0, keep_local).index_select(1, keep_local)
            current_derivatives = current_derivatives.index_select(1, keep_local)

            if current_readout is not None:
                current_readout = current_readout.index_select(0, keep_local)

            # Stabilize the pruned recurrent matrix.
            if self.cfg.match_jacobian_energy_density:
                current_W = sub_pruner.match_jacobian_energy_density(
                    W_pruned=current_W,
                    derivatives_pruned=current_derivatives,
                    target_trace_per_neuron=target_trace_per_neuron,
                    target_rho=(
                        base_esn.spectral_radius
                        if self.cfg.fallback_match_spectral_radius
                        else None
                    ),
                )
            elif self.cfg.fallback_match_spectral_radius:
                current_W = rescale_reservoir_to_target_rho(
                    current_W,
                    target_rho=base_esn.spectral_radius,
                )

            logger.info(
                "DMP step %d/%d complete: kept=%d",
                step_id,
                len(schedule),
                int(keep_idx_global.numel()),
            )

        self.last_pruner = pruner
        self.last_scores = baseline_scores_full.detach().clone()
        self.last_keep_idx = keep_idx_global.detach().clone()
        self.keep_idx = self.last_keep_idx

        return keep_idx_global, current_W

    def run(
        self,
        series_raw_cpu: Optional[torch.Tensor],
        dataset_name: str,
    ) -> Dict:
        """
        Run DMP pruning and return evaluation metrics.

        Parameters
        ----------
        series_raw_cpu:
            Optional raw CPU tensor. If None, the ESN loads data from data_path.

        dataset_name:
            Name used only for logging/result records.
        """
        if torch.cuda.is_available() and str(self.cfg.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        cpu0 = cpu_rss_mb()

        # ------------------------------------------------------
        # 1. Build and train base ESN
        # ------------------------------------------------------
        base_esn = self._build_base_esn()

        if series_raw_cpu is None:
            base_esn.load_data()
        else:
            hydrate_esn_series(base_esn, series_raw_cpu)

        base_esn.generate_reservoir()
        base_esn.train_esn(base_esn.data)

        y_base, train_mse_base, test_mse_base = base_esn.err(
            base_esn.data,
            print_mse=False,
        )

        # ------------------------------------------------------
        # 2. Collect teacher-forced trajectory
        # ------------------------------------------------------
        trajectory = collect_teacher_forced_trajectory(base_esn, base_esn.data)

        # ------------------------------------------------------
        # 3. Create DMP scorer and prune
        # ------------------------------------------------------
        pruner = DynamicalModePruner(
            W=base_esn.W,
            leaky_rate=base_esn.a,
            energy_tau=self.cfg.energy_tau,
            gramian_chunk_size=self.cfg.gramian_chunk_size,
            score_mode=self.cfg.score_mode,
            alpha_dynamic=self.cfg.alpha_dynamic,
            beta_readout=self.cfg.beta_readout,
            gamma_occupancy=self.cfg.gamma_occupancy,
        )

        keep_idx, final_W_pruned = self._progressive_prune(
            base_esn=base_esn,
            trajectory=trajectory,
            pruner=pruner,
        )

        # ------------------------------------------------------
        # 4. Build pruned ESN and retrain only readout
        # ------------------------------------------------------
        pruned_esn = clone_pruned_esn(
            base_esn,
            keep_idx=keep_idx,
            w_override=final_W_pruned,
        )

        pruned_esn.train_esn(pruned_esn.data)

        y_pruned, train_mse_pruned, test_mse_pruned = pruned_esn.err(
            pruned_esn.data,
            print_mse=False,
        )

        # ------------------------------------------------------
        # 5. Save state for inspection
        # ------------------------------------------------------
        self.last_base_esn = base_esn
        self.last_pruned_esn = pruned_esn
        self.last_base_y_test = y_base
        self.last_pruned_y_test = y_pruned

        self.keep_idx = keep_idx.detach().clone()
        self.final_reservoir_size = int(keep_idx.numel())

        self.last_base_train_mse = float(train_mse_base)
        self.last_base_test_mse = float(test_mse_base)
        self.last_pruned_train_mse = float(train_mse_pruned)
        self.last_pruned_test_mse = float(test_mse_pruned)

        runtime_ms = (time.time() - t0) * 1000.0
        cpu1 = cpu_rss_mb()
        gmem = gpu_peak_mb()

        # Positive gain means pruning improved the metric.
        train_gain = float(train_mse_base - train_mse_pruned)
        test_gain = float(test_mse_base - test_mse_pruned)

        return {
            "dataset": dataset_name,
            "seed": int(self.seed),
            "reservoir": int(self.original_reservoir_size),
            "final_reservoir_size": int(self.final_reservoir_size),
            "pruned_count": int(
                self.original_reservoir_size - self.final_reservoir_size
            ),

            # Pruned ESN metrics
            "train_mse": float(train_mse_pruned),
            "train_rmse": rmse_from_mse(float(train_mse_pruned)),
            "test_mse": float(test_mse_pruned),
            "test_rmse": rmse_from_mse(float(test_mse_pruned)),

            # Base ESN metrics
            "base_train_mse": float(train_mse_base),
            "base_test_mse": float(test_mse_base),

            # Positive gain means lower MSE after pruning.
            "train_mse_gain_vs_base": train_gain,
            "test_mse_gain_vs_base": test_gain,

            # Runtime and memory
            "total_time_ms": float(runtime_ms),
            "total_time_s": float(runtime_ms / 1000.0),
            "cpu_rss_mb": float(cpu1),
            "cpu_rss_delta_mb": float(cpu1 - cpu0),
            "gpu_peak_mem_mb": float(gmem),

            # ESN settings
            "horizon": int(self.cfg.multi_step),
            "tau": int(self.cfg.tau),
            "leaky_rate": float(self.cfg.leaky_rate),
            "spectral_radius": float(self.cfg.spectral_radius),
            "sparsity": float(self.cfg.sparsity),
            "ridge_alpha": float(self.cfg.ridge_alpha),
            "input_scaling": float(self.cfg.input_scaling),
            "bias_scaling": float(self.cfg.bias_scaling),
            "feedback_scaling": float(self.cfg.feedback_scaling),
            "normalize_states": False,
            "use_feedback": bool(self.cfg.use_feedback),

            # DMP settings
            "prune_ratio": float(self.cfg.prune_ratio),
            "energy_tau": float(self.cfg.energy_tau),
            "gramian_energy_tau": float(self.cfg.energy_tau),
            "gramian_chunk_size": int(self.cfg.gramian_chunk_size),
            "score_mode": str(self.cfg.score_mode),
            "alpha_dynamic": float(self.cfg.alpha_dynamic),
            "beta_readout": float(self.cfg.beta_readout),
            "gamma_occupancy": float(self.cfg.gamma_occupancy),
            "progressive": bool(self.cfg.progressive),
            "progressive_step_ratio": float(self.cfg.progressive_step_ratio),
            "match_jacobian_energy_density": bool(
                self.cfg.match_jacobian_energy_density
            ),
            "fallback_match_spectral_radius": bool(
                self.cfg.fallback_match_spectral_radius
            ),
            "device": str(self.cfg.device),
        }


# ==============================================================
# Example usage
# ==============================================================

# cfg = DMPConfig(
#     tau=1,
#     multi_step=20,
#     leaky_rate=0.1,
#     spectral_radius=0.95,
#     ridge_alpha=1e-6,
#     sparsity=0.8,
#     input_scaling=1.0,
#     bias_scaling=0.2,
#     feedback_scaling=0.0,
#     normalize_states=False,
#     use_feedback=False,
#     device="cuda:0" if torch.cuda.is_available() else "cpu",
#     prune_ratio=0.20,
#     energy_tau=0.90,
#     gramian_chunk_size=128,
#
#     # Paper-aligned default:
#     score_mode="dynamic_only",
#
#     # Optional variants:
#     # score_mode="dynamic_occupancy",
#     # gamma_occupancy=0.15,
#     #
#     # score_mode="hybrid",
#     # alpha_dynamic=0.70,
#     # beta_readout=0.30,
#     # gamma_occupancy=0.15,
#
#     progressive=True,
#     progressive_step_ratio=0.05,
#     match_jacobian_energy_density=True,
#     fallback_match_spectral_radius=True,
# )
#
# model = DMPESN(
#     res_size=1000,
#     seed=0,
#     init_len=100,
#     train_len=2000,
#     test_len=500,
#     config=cfg,
#     data_path="path/to/data.csv",
# )
#
# result = model.run(series_raw_cpu=None, dataset_name="mackey_glass")
# print(result)
