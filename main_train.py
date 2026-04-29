from pathlib import Path
import time
from typing import Dict, Optional

import pandas as pd
import psutil
import torch
import yaml

from src.dmp_model import (
    DMPConfig,
    DMPESN,
    DMPResultSaveConfig,
    DMPResultsSaver,
)

from src.sota_models import (
    base_esn as base_model,
    betweenness_esn as betweenness_model,
    closeness_esn as closeness_model,
    deep_esn as deep_model,
    leaky_esn as leaky_model,
)

from src.utils.overleaf_table_formatter import save_all_ours_latex_tables


# ==================== Config Loading ==================== #
CONFIG_PATH = Path("static/config.yml")
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    CONFIG = yaml.safe_load(handle) or {}


# ==================== Small Config Helpers ==================== #
def cfg_section(*names: str) -> dict:
    for name in names:
        if name in CONFIG and isinstance(CONFIG[name], dict):
            return CONFIG[name]
    raise KeyError(f"Missing config section. Tried: {names}")


def get_bool(section: dict, key: str, default: bool = False) -> bool:
    value = section.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def get_float(section: dict, key: str, default: float) -> float:
    return float(section.get(key, default))


def get_int(section: dict, key: str, default: int) -> int:
    return int(section.get(key, default))


def get_list_int(section: dict, key: str, default) -> list[int]:
    return [int(x) for x in section.get(key, default)]


def get_list_float(section: dict, key: str, default) -> list[float]:
    return [float(x) for x in section.get(key, default)]


# ==================== Data Section ==================== #
CFG_DATA = cfg_section("data")
DATA_ROOT = Path(str(CFG_DATA["root"]))

DATASETS = [
    {
        "name": str(ds["name"]),
        "file": str(ds["file"]),
        "path": str(DATA_ROOT / str(ds["file"])),
    }
    for ds in CFG_DATA["datasets"]
]


# ==================== GPU Section ==================== #
CFG_GPU = cfg_section("gpus_setting")

GPU_DEVICE = str(CFG_GPU.get("device", "auto"))
GPU_IDS = [int(x) for x in CFG_GPU.get("gpus", [0])]

if GPU_DEVICE == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(GPU_DEVICE)

DTYPE = torch.float32 if DEVICE.type == "cuda" else torch.float64


# ==================== Save Section ==================== #
CFG_SAVE = cfg_section("save_dir")
RESULTS_ROOT = str(CFG_SAVE.get("results_root", "results"))


# ==================== DMP / Ours Section ==================== #
# Public name is DMP. Older configs may still use "ours".
CFG_DMP = cfg_section("dmp", "ours")

DMP_CONFIG = {
    "tau": int(CFG_DMP["tau"]),
    "horizons": get_list_int(CFG_DMP, "horizons", [20]),
    "prune_ratios": get_list_float(CFG_DMP, "prune_ratios", [0.20]),
    "seeds": get_list_int(CFG_DMP, "seeds", [0]),
    "skip_existing": get_bool(CFG_DMP, "skip_existing", True),

    # ESN hyperparameters
    "leaky_rate": float(CFG_DMP["leaky_rate"]),
    "spectral_radius": float(CFG_DMP["spectral_radius"]),
    "input_scaling": get_float(CFG_DMP, "input_scaling", 0.5),
    "bias_scaling": get_float(CFG_DMP, "bias_scaling", 0.2),
    "feedback_scaling": get_float(CFG_DMP, "feedback_scaling", 0.0),
    "sparsity": get_float(CFG_DMP, "sparsity", 0.1),
    "ridge_alpha": get_float(CFG_DMP, "ridge_alpha", 1e-4),
    "normalize_states": get_bool(CFG_DMP, "normalize_states", False),
    "use_feedback": get_bool(CFG_DMP, "use_feedback", False),

    # DMP scoring
    "energy_tau": float(CFG_DMP.get("energy_tau", CFG_DMP.get("gramian_energy_tau", 0.90))),
    "gramian_chunk_size": get_int(CFG_DMP, "gramian_chunk_size", 128),
    "score_mode": str(CFG_DMP.get("score_mode", "dynamic_only")),
    "alpha_dynamic": get_float(CFG_DMP, "alpha_dynamic", 1.0),
    "beta_readout": get_float(CFG_DMP, "beta_readout", 0.0),
    "gamma_occupancy": get_float(CFG_DMP, "gamma_occupancy", 0.0),

    # Pruning schedule / stabilization
    "progressive": get_bool(CFG_DMP, "progressive", True),
    "progressive_step_ratio": get_float(CFG_DMP, "progressive_step_ratio", 0.05),
    "match_jacobian_energy_density": get_bool(
        CFG_DMP,
        "match_jacobian_energy_density",
        get_bool(CFG_DMP, "match_jacobian_energy", True),
    ),
    "fallback_match_spectral_radius": get_bool(CFG_DMP, "fallback_match_spectral_radius", True),

    "reservoir_sizes": get_list_int(CFG_DMP, "reservoir_sizes", [1000]),
}


# ==================== SOTA Base ESN Section ==================== #
CFG_BASE_ESN = cfg_section("base_esn")

BASE_ESN_CONFIG = {
    "reservoir_sizes": get_list_int(CFG_BASE_ESN, "reservoir_sizes", [1000]),
    "seeds": get_list_int(CFG_BASE_ESN, "seeds", [0]),
    "leaky_rate": float(CFG_BASE_ESN["leaky_rate"]),
    "spectral_radius": float(CFG_BASE_ESN["spectral_radius"]),
    "input_scaling": float(CFG_BASE_ESN["input_scaling"]),
    "bias_scaling": float(CFG_BASE_ESN["bias_scaling"]),
    "feedback_scaling": get_float(CFG_BASE_ESN, "feedback_scaling", 0.0),
    "sparsity": float(CFG_BASE_ESN["sparsity"]),
    "ridge_alpha": float(CFG_BASE_ESN["ridge_alpha"]),
    "normalize_states": get_bool(CFG_BASE_ESN, "normalize_states", False),
    "use_feedback": get_bool(CFG_BASE_ESN, "use_feedback", False),
}


# ==================== SOTA Betweenness Section ==================== #
CFG_BETWEENNESS = cfg_section("betweenness")

BETWEENNESS_CONFIG = {
    "reservoir_sizes": get_list_int(CFG_BETWEENNESS, "reservoir_sizes", [1000]),
    "seeds": get_list_int(CFG_BETWEENNESS, "seeds", [0]),
    "prune_ratio": float(CFG_BETWEENNESS["prune_ratio"]),
    "prune_sr_iters": get_int(CFG_BETWEENNESS, "prune_sr_iters", 30),
    "leaky_rate": float(CFG_BETWEENNESS["leaky_rate"]),
    "spectral_radius": float(CFG_BETWEENNESS["spectral_radius"]),
    "input_scaling": float(CFG_BETWEENNESS["input_scaling"]),
    "bias_scaling": float(CFG_BETWEENNESS["bias_scaling"]),
    "feedback_scaling": get_float(CFG_BETWEENNESS, "feedback_scaling", 0.0),
    "sparsity": float(CFG_BETWEENNESS["sparsity"]),
    "ridge_alpha": float(CFG_BETWEENNESS["ridge_alpha"]),
    "normalize_states": get_bool(CFG_BETWEENNESS, "normalize_states", False),
    "use_feedback": get_bool(CFG_BETWEENNESS, "use_feedback", False),
}


# ==================== SOTA Closeness Section ==================== #
CFG_CLOSENESS = cfg_section("closeness")

CLOSENESS_CONFIG = {
    "reservoir_sizes": get_list_int(CFG_CLOSENESS, "reservoir_sizes", [1000]),
    "seeds": get_list_int(CFG_CLOSENESS, "seeds", [0]),
    "prune_ratio": float(CFG_CLOSENESS["prune_ratio"]),
    "prune_sr_iters": get_int(CFG_CLOSENESS, "prune_sr_iters", 30),
    "leaky_rate": float(CFG_CLOSENESS["leaky_rate"]),
    "spectral_radius": float(CFG_CLOSENESS["spectral_radius"]),
    "input_scaling": float(CFG_CLOSENESS["input_scaling"]),
    "bias_scaling": float(CFG_CLOSENESS["bias_scaling"]),
    "feedback_scaling": get_float(CFG_CLOSENESS, "feedback_scaling", 0.0),
    "sparsity": float(CFG_CLOSENESS["sparsity"]),
    "ridge_alpha": float(CFG_CLOSENESS["ridge_alpha"]),
    "normalize_states": get_bool(CFG_CLOSENESS, "normalize_states", False),
    "use_feedback": get_bool(CFG_CLOSENESS, "use_feedback", False),
}


# ==================== SOTA Leaky ESN Section ==================== #
CFG_LEAKY_ESN = cfg_section("leaky_esn")

LEAKY_ESN_CONFIG = {
    "reservoir_sizes": get_list_int(CFG_LEAKY_ESN, "reservoir_sizes", [1000]),
    "seeds": get_list_int(CFG_LEAKY_ESN, "seeds", [0]),
    "leaky_rate": float(CFG_LEAKY_ESN["leaky_rate"]),
    "spectral_radius": float(CFG_LEAKY_ESN["spectral_radius"]),
    "input_scaling": float(CFG_LEAKY_ESN["input_scaling"]),
    "bias_scaling": float(CFG_LEAKY_ESN["bias_scaling"]),
    "sparsity_zero_frac": float(CFG_LEAKY_ESN.get("sparsity_zero_frac", CFG_LEAKY_ESN.get("sparsity", 0.1))),
    "ridge_alpha": float(CFG_LEAKY_ESN["ridge_alpha"]),
    "plot_first_n": get_int(CFG_LEAKY_ESN, "plot_first_n", CONFIG.get("general_config", {}).get("plot_first_n", 500)),
    "plot_dpi": get_int(CFG_LEAKY_ESN, "plot_dpi", CONFIG.get("general_config", {}).get("plot_dpi", 300)),
}


# ==================== SOTA Deep ESN Section ==================== #
CFG_DEEP_ESN = cfg_section("deep_esn")

DEEP_ESN_CONFIG = {
    "reservoir_sizes": get_list_int(CFG_DEEP_ESN, "reservoir_sizes", [1000]),
    "seeds": get_list_int(CFG_DEEP_ESN, "seeds", [0]),
    "n_layers": int(CFG_DEEP_ESN["n_layers"]),
    "leaky_rate": float(CFG_DEEP_ESN["leaky_rate"]),
    "spectral_radius": float(CFG_DEEP_ESN["spectral_radius"]),
    "input_scaling": float(CFG_DEEP_ESN["input_scaling"]),
    "inter_layer_scaling": float(CFG_DEEP_ESN["inter_layer_scaling"]),
    "bias_scaling": float(CFG_DEEP_ESN["bias_scaling"]),
    "sparsity_zero_frac": float(CFG_DEEP_ESN.get("sparsity_zero_frac", CFG_DEEP_ESN.get("sparsity", 0.1))),
    "ridge_alpha": float(CFG_DEEP_ESN["ridge_alpha"]),
    "plot_first_n": get_int(CFG_DEEP_ESN, "plot_first_n", CONFIG.get("general_config", {}).get("plot_first_n", 500)),
    "plot_dpi": get_int(CFG_DEEP_ESN, "plot_dpi", CONFIG.get("general_config", {}).get("plot_dpi", 300)),
}


# ==================== General Section ==================== #
CFG_GENERAL = cfg_section("general_config")

GENERAL_CONFIG = {
    "train_ratio": float(CFG_GENERAL["train_ratio"]),
    "test_ratio": float(CFG_GENERAL["test_ratio"]),
    "initial_ratio": float(CFG_GENERAL["initial_ratio"]),
    "results_root_template": str(CFG_GENERAL["results_root_template"]),
    "plot_first_n": get_int(CFG_GENERAL, "plot_first_n", 500),
    "plot_dpi": get_int(CFG_GENERAL, "plot_dpi", 300),
}


# ==================== Run Control ==================== #
# Turn models on/off here without commenting code blocks.
RUN_MODELS = {
    "base_esn": True,
    "leaky_esn": True,
    "deep_esn": True,
    "closeness_esn": True,
    "betweenness_esn": True,
    "dmp": True,
}

TAU = int(DMP_CONFIG["tau"])
DEFAULT_PRUNE_RATIO = float(DMP_CONFIG["prune_ratios"][0])


# ==================== Runtime Helpers ==================== #
def start_profile() -> tuple[float, float]:
    if torch.cuda.is_available() and str(DEVICE).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    cpu0 = float(psutil.Process().memory_info().rss / (1024 * 1024))

    return t0, cpu0


def end_profile(t0: float, cpu0: float) -> dict:
    cpu1 = float(psutil.Process().memory_info().rss / (1024 * 1024))

    gpu_peak = 0.0
    if torch.cuda.is_available() and str(DEVICE).startswith("cuda"):
        gpu_peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    dt_s = float(time.time() - t0)

    return {
        "total_time_ms": float(dt_s * 1000.0),
        "total_time_s": dt_s,
        "cpu_rss_mb": cpu1,
        "cpu_rss_delta_mb": float(cpu1 - cpu0),
        "gpu_peak_mem_mb": gpu_peak,
    }


def update_model_totals(model_totals: dict, row: dict) -> None:
    model_name = str(row.get("model", "unknown"))

    if model_name not in model_totals:
        model_totals[model_name] = {
            "runs": 0,
            "total_time_s": 0.0,
            "max_cpu_rss_mb": 0.0,
            "max_gpu_peak_mem_mb": 0.0,
        }

    model_totals[model_name]["runs"] += 1
    model_totals[model_name]["total_time_s"] += float(row.get("total_time_s", 0.0))
    model_totals[model_name]["max_cpu_rss_mb"] = max(
        float(model_totals[model_name]["max_cpu_rss_mb"]),
        float(row.get("cpu_rss_mb", 0.0)),
    )
    model_totals[model_name]["max_gpu_peak_mem_mb"] = max(
        float(model_totals[model_name]["max_gpu_peak_mem_mb"]),
        float(row.get("gpu_peak_mem_mb", 0.0)),
    )


def model_out_dir(
    dataset_name: str,
    horizon: int,
    model_name: str,
    res_size: int,
    seed: int,
    extra_tag: str = "",
) -> str:
    root = (
        Path(RESULTS_ROOT)
        / GENERAL_CONFIG["results_root_template"]
        / dataset_name
        / f"h_{int(horizon)}"
        / model_name
    )

    if extra_tag:
        root = root / extra_tag

    return str(root / str(int(res_size)) / f"seed_{int(seed)}")


def metrics_path_for(
    dataset_name: str,
    horizon: int,
    model_name: str,
    res_size: int,
    seed: int,
    extra_tag: str = "",
) -> Path:
    return Path(
        model_out_dir(
            dataset_name=dataset_name,
            horizon=horizon,
            model_name=model_name,
            res_size=res_size,
            seed=seed,
            extra_tag=extra_tag,
        )
    ) / "metrics.csv"


def should_skip(path: Path, skip_existing: bool) -> bool:
    return bool(skip_existing and path.exists())


def read_series(path: str) -> torch.Tensor:
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    series = torch.tensor(
        pd.read_csv(path, header=0).iloc[:, 0].values,
        dtype=torch.float64,
    )

    return torch.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)


def append_row(all_rows: list[dict], model_totals: dict, row: dict, model_name: str) -> None:
    row["model"] = model_name
    all_rows.append(row)
    update_model_totals(model_totals, row)


# ==================== Main Training ==================== #
def main() -> None:
    all_rows = []
    model_totals = {}

    dmp_saver = DMPResultsSaver(
        DMPResultSaveConfig(
            results_root=RESULTS_ROOT,
            results_root_template=GENERAL_CONFIG["results_root_template"],
            model_name="DMP",
        )
    )

    for ds in DATASETS:
        dataset_name = ds["name"]
        dataset_path = ds["path"]

        series_raw_cpu = read_series(dataset_path)
        n_total = len(series_raw_cpu)

        init_len = int(GENERAL_CONFIG["initial_ratio"] * n_total)
        test_len = int(GENERAL_CONFIG["test_ratio"] * n_total)
        train_len = n_total - init_len - test_len

        print(f"\n================ DATASET: {dataset_name} ================")
        print(f"N={n_total} | init={init_len} | train={train_len} | test={test_len}")

        for horizon in DMP_CONFIG["horizons"]:
            print(f"\n----- HORIZON: {horizon} -----")

            # --------------------------------------------------
            # 1) Leaky ESN
            # --------------------------------------------------
            if RUN_MODELS["leaky_esn"]:
                print(f"[START] Leaky ESN | dataset={dataset_name} | h={horizon}")

                leaky_cfg = leaky_model.LeakyESNConfig(
                    multi_step=int(horizon),
                    tau=TAU,
                    leaky_rate=float(LEAKY_ESN_CONFIG["leaky_rate"]),
                    spectral_radius=float(LEAKY_ESN_CONFIG["spectral_radius"]),
                    sparsity_zero_frac=float(LEAKY_ESN_CONFIG["sparsity_zero_frac"]),
                    ridge_alpha=float(LEAKY_ESN_CONFIG["ridge_alpha"]),
                    input_scaling=float(LEAKY_ESN_CONFIG["input_scaling"]),
                    bias_scaling=float(LEAKY_ESN_CONFIG["bias_scaling"]),
                    device=str(DEVICE),
                    dtype=DTYPE,
                    plot_first_n=int(LEAKY_ESN_CONFIG["plot_first_n"]),
                    plot_dpi=int(LEAKY_ESN_CONFIG["plot_dpi"]),
                )

                for res_size in LEAKY_ESN_CONFIG["reservoir_sizes"]:
                    for seed in LEAKY_ESN_CONFIG["seeds"]:
                        out_dir = model_out_dir(dataset_name, horizon, "leaky_esn", res_size, seed)
                        metrics_path = Path(out_dir) / "metrics.csv"

                        if should_skip(metrics_path, DMP_CONFIG["skip_existing"]):
                            print(f"[SKIP] leaky_esn | {dataset_name} | h={horizon} | res={res_size} | seed={seed}")
                            continue

                        t0, cpu0 = start_profile()

                        runner = leaky_model.LeakyESN(
                            res_size=int(res_size),
                            seed=int(seed),
                            config=leaky_cfg,
                        )

                        row = runner.run(
                            series_raw_cpu=series_raw_cpu,
                            init_len=init_len,
                            train_len=train_len,
                            test_len=test_len,
                            out_dir=out_dir,
                            dataset_name=dataset_name,
                        )

                        row.update(end_profile(t0, cpu0))
                        append_row(all_rows, model_totals, row, "leaky_esn")

                        print(
                            f"dataset={dataset_name} | h={horizon} | model=leaky_esn | "
                            f"res={res_size} seed={seed} test_mse={row['test_mse']:.8f}"
                        )

                print(f"[DONE] Leaky ESN | dataset={dataset_name} | h={horizon}")

            # --------------------------------------------------
            # 2) Deep ESN
            # --------------------------------------------------
            if RUN_MODELS["deep_esn"]:
                print(f"[START] Deep ESN | dataset={dataset_name} | h={horizon}")

                deep_cfg = deep_model.DeepESNConfig(
                    multi_step=int(horizon),
                    tau=TAU,
                    n_layers=int(DEEP_ESN_CONFIG["n_layers"]),
                    leaky_rate=float(DEEP_ESN_CONFIG["leaky_rate"]),
                    spectral_radius=float(DEEP_ESN_CONFIG["spectral_radius"]),
                    sparsity_zero_frac=float(DEEP_ESN_CONFIG["sparsity_zero_frac"]),
                    ridge_alpha=float(DEEP_ESN_CONFIG["ridge_alpha"]),
                    input_scaling=float(DEEP_ESN_CONFIG["input_scaling"]),
                    inter_layer_scaling=float(DEEP_ESN_CONFIG["inter_layer_scaling"]),
                    bias_scaling=float(DEEP_ESN_CONFIG["bias_scaling"]),
                    device=str(DEVICE),
                    dtype=DTYPE,
                    plot_first_n=int(DEEP_ESN_CONFIG["plot_first_n"]),
                    plot_dpi=int(DEEP_ESN_CONFIG["plot_dpi"]),
                )

                for res_size in DEEP_ESN_CONFIG["reservoir_sizes"]:
                    for seed in DEEP_ESN_CONFIG["seeds"]:
                        out_dir = model_out_dir(dataset_name, horizon, "deep_esn", res_size, seed)
                        metrics_path = Path(out_dir) / "metrics.csv"

                        if should_skip(metrics_path, DMP_CONFIG["skip_existing"]):
                            print(f"[SKIP] deep_esn | {dataset_name} | h={horizon} | res={res_size} | seed={seed}")
                            continue

                        t0, cpu0 = start_profile()

                        runner = deep_model.DeepESN(
                            res_size=int(res_size),
                            seed=int(seed),
                            config=deep_cfg,
                        )

                        row = runner.run(
                            series_raw_cpu=series_raw_cpu,
                            init_len=init_len,
                            train_len=train_len,
                            test_len=test_len,
                            out_dir=out_dir,
                            dataset_name=dataset_name,
                        )

                        row.update(end_profile(t0, cpu0))
                        append_row(all_rows, model_totals, row, "deep_esn")

                        print(
                            f"dataset={dataset_name} | h={horizon} | model=deep_esn | "
                            f"res={res_size} seed={seed} test_mse={row['test_mse']:.8f}"
                        )

                print(f"[DONE] Deep ESN | dataset={dataset_name} | h={horizon}")

            # --------------------------------------------------
            # 3) Closeness ESN
            # --------------------------------------------------
            if RUN_MODELS["closeness_esn"]:
                print(f"[START] Closeness ESN | dataset={dataset_name} | h={horizon}")

                closeness_cfg = closeness_model.ClosenessPruningConfig(
                    tau=TAU,
                    multi_step=int(horizon),
                    leaky_rate=float(CLOSENESS_CONFIG["leaky_rate"]),
                    spectral_radius=float(CLOSENESS_CONFIG["spectral_radius"]),
                    ridge_alpha=float(CLOSENESS_CONFIG["ridge_alpha"]),
                    sparsity=float(CLOSENESS_CONFIG["sparsity"]),
                    input_scaling=float(CLOSENESS_CONFIG["input_scaling"]),
                    bias_scaling=float(CLOSENESS_CONFIG["bias_scaling"]),
                    feedback_scaling=float(CLOSENESS_CONFIG["feedback_scaling"]),
                    normalize_states=bool(CLOSENESS_CONFIG["normalize_states"]),
                    use_feedback=bool(CLOSENESS_CONFIG["use_feedback"]),
                    prune_ratio=float(CLOSENESS_CONFIG["prune_ratio"]),
                    prune_sr_iters=int(CLOSENESS_CONFIG["prune_sr_iters"]),
                    device=str(DEVICE),
                )

                prune_tag = f"prune_{int(round(float(closeness_cfg.prune_ratio) * 100))}pct"

                for res_size in CLOSENESS_CONFIG["reservoir_sizes"]:
                    for seed in CLOSENESS_CONFIG["seeds"]:
                        out_dir = model_out_dir(
                            dataset_name,
                            horizon,
                            "closeness_esn",
                            res_size,
                            seed,
                            extra_tag=prune_tag,
                        )
                        metrics_path = Path(out_dir) / "metrics.csv"

                        if should_skip(metrics_path, DMP_CONFIG["skip_existing"]):
                            print(f"[SKIP] closeness_esn | {dataset_name} | h={horizon} | res={res_size} | seed={seed}")
                            continue

                        t0, cpu0 = start_profile()

                        runner = closeness_model.ClosenessPruningESN(
                            res_size=int(res_size),
                            seed=int(seed),
                            init_len=init_len,
                            train_len=train_len,
                            test_len=test_len,
                            config=closeness_cfg,
                            data_path=None,
                        )

                        row = runner.run(
                            series_raw_cpu=series_raw_cpu,
                            out_dir=out_dir,
                            dataset_name=dataset_name,
                        )

                        row.update(end_profile(t0, cpu0))
                        append_row(all_rows, model_totals, row, "closeness_esn")

                        print(
                            f"dataset={dataset_name} | h={horizon} | model=closeness_esn | "
                            f"res={res_size} seed={seed} test_mse={row['test_mse']:.8f}"
                        )

                print(f"[DONE] Closeness ESN | dataset={dataset_name} | h={horizon}")

            # --------------------------------------------------
            # 4) Betweenness ESN
            # --------------------------------------------------
            if RUN_MODELS["betweenness_esn"]:
                print(f"[START] Betweenness ESN | dataset={dataset_name} | h={horizon}")

                betweenness_cfg = betweenness_model.BetweennessPruningConfig(
                    tau=TAU,
                    multi_step=int(horizon),
                    leaky_rate=float(BETWEENNESS_CONFIG["leaky_rate"]),
                    spectral_radius=float(BETWEENNESS_CONFIG["spectral_radius"]),
                    ridge_alpha=float(BETWEENNESS_CONFIG["ridge_alpha"]),
                    sparsity=float(BETWEENNESS_CONFIG["sparsity"]),
                    input_scaling=float(BETWEENNESS_CONFIG["input_scaling"]),
                    bias_scaling=float(BETWEENNESS_CONFIG["bias_scaling"]),
                    feedback_scaling=float(BETWEENNESS_CONFIG["feedback_scaling"]),
                    normalize_states=bool(BETWEENNESS_CONFIG["normalize_states"]),
                    use_feedback=bool(BETWEENNESS_CONFIG["use_feedback"]),
                    prune_ratio=float(BETWEENNESS_CONFIG["prune_ratio"]),
                    prune_sr_iters=int(BETWEENNESS_CONFIG["prune_sr_iters"]),
                    device=str(DEVICE),
                )

                prune_tag = f"prune_{int(round(float(betweenness_cfg.prune_ratio) * 100))}pct"

                for res_size in BETWEENNESS_CONFIG["reservoir_sizes"]:
                    for seed in BETWEENNESS_CONFIG["seeds"]:
                        out_dir = model_out_dir(
                            dataset_name,
                            horizon,
                            "betweenness_esn",
                            res_size,
                            seed,
                            extra_tag=prune_tag,
                        )
                        metrics_path = Path(out_dir) / "metrics.csv"

                        if should_skip(metrics_path, DMP_CONFIG["skip_existing"]):
                            print(f"[SKIP] betweenness_esn | {dataset_name} | h={horizon} | res={res_size} | seed={seed}")
                            continue

                        t0, cpu0 = start_profile()

                        runner = betweenness_model.BetweennessPruningESN(
                            res_size=int(res_size),
                            seed=int(seed),
                            init_len=init_len,
                            train_len=train_len,
                            test_len=test_len,
                            config=betweenness_cfg,
                            data_path=None,
                        )

                        row = runner.run(
                            series_raw_cpu=series_raw_cpu,
                            out_dir=out_dir,
                            dataset_name=dataset_name,
                        )

                        row.update(end_profile(t0, cpu0))
                        append_row(all_rows, model_totals, row, "betweenness_esn")

                        print(
                            f"dataset={dataset_name} | h={horizon} | model=betweenness_esn | "
                            f"res={res_size} seed={seed} test_mse={row['test_mse']:.8f}"
                        )

                print(f"[DONE] Betweenness ESN | dataset={dataset_name} | h={horizon}")

            # --------------------------------------------------
            # 5) Base ESN
            # --------------------------------------------------
            if RUN_MODELS["base_esn"]:
                print(f"[START] Base ESN | dataset={dataset_name} | h={horizon}")

                base_cfg = base_model.BaseESNRunConfig(
                    tau=TAU,
                    multi_step=int(horizon),
                    leaky_rate=float(BASE_ESN_CONFIG["leaky_rate"]),
                    spectral_radius=float(BASE_ESN_CONFIG["spectral_radius"]),
                    ridge_alpha=float(BASE_ESN_CONFIG["ridge_alpha"]),
                    sparsity=float(BASE_ESN_CONFIG["sparsity"]),
                    input_scaling=float(BASE_ESN_CONFIG["input_scaling"]),
                    bias_scaling=float(BASE_ESN_CONFIG["bias_scaling"]),
                    feedback_scaling=float(BASE_ESN_CONFIG["feedback_scaling"]),
                    normalize_states=bool(BASE_ESN_CONFIG["normalize_states"]),
                    use_feedback=bool(BASE_ESN_CONFIG["use_feedback"]),
                    device=str(DEVICE),
                )

                for res_size in BASE_ESN_CONFIG["reservoir_sizes"]:
                    for seed in BASE_ESN_CONFIG["seeds"]:
                        out_dir = model_out_dir(dataset_name, horizon, "base_esn", res_size, seed)
                        metrics_path = Path(out_dir) / "metrics.csv"

                        if should_skip(metrics_path, DMP_CONFIG["skip_existing"]):
                            print(f"[SKIP] base_esn | {dataset_name} | h={horizon} | res={res_size} | seed={seed}")
                            continue

                        t0, cpu0 = start_profile()

                        runner = base_model.BaseESNRunner(
                            res_size=int(res_size),
                            seed=int(seed),
                            init_len=init_len,
                            train_len=train_len,
                            test_len=test_len,
                            config=base_cfg,
                            data_path=None,
                        )

                        row = runner.run(
                            series_raw_cpu=series_raw_cpu,
                            out_dir=out_dir,
                            dataset_name=dataset_name,
                        )

                        row.update(end_profile(t0, cpu0))
                        append_row(all_rows, model_totals, row, "base_esn")

                        print(
                            f"dataset={dataset_name} | h={horizon} | model=base_esn | "
                            f"res={res_size} seed={seed} test_mse={row['test_mse']:.8f}"
                        )

                print(f"[DONE] Base ESN | dataset={dataset_name} | h={horizon}")

            # --------------------------------------------------
            # 6) DMP ESN
            # --------------------------------------------------
            if RUN_MODELS["dmp"]:
                print(f"[START] DMP ESN | dataset={dataset_name} | h={horizon}")

                for prune_ratio in DMP_CONFIG["prune_ratios"]:
                    prune_tag = f"prune_{int(round(float(prune_ratio) * 100))}pct"

                    dmp_cfg = DMPConfig(
                        tau=TAU,
                        multi_step=int(horizon),
                        leaky_rate=float(DMP_CONFIG["leaky_rate"]),
                        spectral_radius=float(DMP_CONFIG["spectral_radius"]),
                        ridge_alpha=float(DMP_CONFIG["ridge_alpha"]),
                        sparsity=float(DMP_CONFIG["sparsity"]),
                        input_scaling=float(DMP_CONFIG["input_scaling"]),
                        bias_scaling=float(DMP_CONFIG["bias_scaling"]),
                        feedback_scaling=float(DMP_CONFIG["feedback_scaling"]),
                        normalize_states=False,
                        use_feedback=bool(DMP_CONFIG["use_feedback"]),
                        device=str(DEVICE),
                        prune_ratio=float(prune_ratio),
                        energy_tau=float(DMP_CONFIG["energy_tau"]),
                        gramian_chunk_size=int(DMP_CONFIG["gramian_chunk_size"]),
                        score_mode=str(DMP_CONFIG["score_mode"]),
                        alpha_dynamic=float(DMP_CONFIG["alpha_dynamic"]),
                        beta_readout=float(DMP_CONFIG["beta_readout"]),
                        gamma_occupancy=float(DMP_CONFIG["gamma_occupancy"]),
                        progressive=bool(DMP_CONFIG["progressive"]),
                        progressive_step_ratio=float(DMP_CONFIG["progressive_step_ratio"]),
                        match_jacobian_energy_density=bool(DMP_CONFIG["match_jacobian_energy_density"]),
                        fallback_match_spectral_radius=bool(DMP_CONFIG["fallback_match_spectral_radius"]),
                    )

                    for res_size in DMP_CONFIG["reservoir_sizes"]:
                        for seed in DMP_CONFIG["seeds"]:
                            out_dir = model_out_dir(
                                dataset_name,
                                horizon,
                                "DMP",
                                res_size,
                                seed,
                                extra_tag=prune_tag,
                            )
                            metrics_path = Path(out_dir) / "metrics.csv"

                            if should_skip(metrics_path, DMP_CONFIG["skip_existing"]):
                                print(
                                    f"[SKIP] DMP | {dataset_name} | h={horizon} | "
                                    f"prune={int(round(float(prune_ratio) * 100))}% | "
                                    f"res={res_size} | seed={seed}"
                                )
                                continue

                            t0, cpu0 = start_profile()

                            runner = DMPESN(
                                res_size=int(res_size),
                                seed=int(seed),
                                init_len=init_len,
                                train_len=train_len,
                                test_len=test_len,
                                config=dmp_cfg,
                                data_path=None,
                            )

                            row = runner.run(
                                series_raw_cpu=series_raw_cpu,
                                dataset_name=dataset_name,
                            )

                            row.update(end_profile(t0, cpu0))
                            row["model"] = "DMP"

                            # DMPResultsSaver builds the same output path internally.
                            dmp_saver.save(row=row, runner=runner)

                            all_rows.append(row)
                            update_model_totals(model_totals, row)

                            print(
                                f"dataset={dataset_name} | h={horizon} | model=DMP | "
                                f"prune={int(round(float(prune_ratio) * 100))}% | "
                                f"res={res_size} seed={seed} test_mse={row['test_mse']:.8f}"
                            )

                print(f"[DONE] DMP ESN | dataset={dataset_name} | h={horizon}")

    print(f"\nAll training completed. Total new runs: {len(all_rows)}")

    if model_totals:
        print("\nPer-model resource summary:")
        for model_name in sorted(model_totals.keys()):
            s = model_totals[model_name]
            print(
                f"{model_name}: runs={s['runs']} | total_time={s['total_time_s']:.2f}s | "
                f"max_cpu={s['max_cpu_rss_mb']:.1f}MB | max_gpu={s['max_gpu_peak_mem_mb']:.1f}MB"
            )

    latex_out_dir = Path(RESULTS_ROOT) / GENERAL_CONFIG["results_root_template"] / "latex_tables"

    save_all_ours_latex_tables(
        save_dir=latex_out_dir,
        results_root=RESULTS_ROOT,
        results_root_template=GENERAL_CONFIG["results_root_template"],
        model_name="DMP",
        benchmark_reservoirs=(300, 500, 700, 1000),
        scaling_sizes=tuple(int(x) for x in DMP_CONFIG["reservoir_sizes"]),
        horizons=tuple(int(x) for x in DMP_CONFIG["horizons"]),
        prune_ratios=tuple(float(x) for x in DMP_CONFIG["prune_ratios"]),
        default_horizon=20,
        default_prune_ratio=float(DEFAULT_PRUNE_RATIO),
        clear_existing_tex=True,
    )

    print(f"\nSaved LaTeX tables to: {latex_out_dir}")


if __name__ == "__main__":
    main()