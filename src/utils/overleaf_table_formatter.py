from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


NL = r"\\"

DATASET_LABELS = {
    "mackey_glass_1d": "Mackey--Glass",
    "electricity_consumption_1d": "Electricity",
    "temperature": "Temperature",
    "wind": "Wind",
    "solar_consumption_1d": "Solar",
}

DATASET_ORDER = [
    "mackey_glass_1d",
    "electricity_consumption_1d",
    "temperature",
    "wind",
    "solar_consumption_1d",
]


# ==============================================================
# Model naming compatibility
# ==============================================================

MODEL_ALIASES = {
    # Base ESN
    "old_esn": "base_esn",
    "base_esn": "base_esn",
    "BaseESN": "base_esn",
    "Base ESN": "base_esn",

    # Leaky ESN
    "leaky_esn": "leaky_esn",
    "LeakyESN": "leaky_esn",
    "Leaky ESN": "leaky_esn",

    # Deep ESN
    "deep_esn": "deep_esn",
    "DeepESN": "deep_esn",
    "Deep ESN": "deep_esn",

    # Betweenness
    "betweenness_esn": "betweenness_esn",
    "BetweennessPruningESN": "betweenness_esn",
    "Betweenness ESN": "betweenness_esn",
    "Betweenness-pruned ESN": "betweenness_esn",

    # Closeness
    "closeness_esn": "closeness_esn",
    "ClosenessPruningESN": "closeness_esn",
    "Closeness ESN": "closeness_esn",
    "Closeness-pruned ESN": "closeness_esn",

    # Ours / DMP
    "ours_jacobian": "dmp",
    "ImprovedJacobianPruningESN": "dmp",
    "DMP": "dmp",
    "dmp": "dmp",
    "dmp_esn": "dmp",
    "DMPESN": "dmp",
    "DMP ESN": "dmp",
}

MODEL_SEARCH_NAMES = {
    "base_esn": [
        "base_esn",
        "old_esn",
        "BaseESN",
        "Base ESN",
    ],
    "leaky_esn": [
        "leaky_esn",
        "LeakyESN",
        "Leaky ESN",
    ],
    "deep_esn": [
        "deep_esn",
        "DeepESN",
        "Deep ESN",
    ],
    "betweenness_esn": [
        "betweenness_esn",
        "BetweennessPruningESN",
        "Betweenness ESN",
        "Betweenness-pruned ESN",
    ],
    "closeness_esn": [
        "closeness_esn",
        "ClosenessPruningESN",
        "Closeness ESN",
        "Closeness-pruned ESN",
    ],
    "dmp": [
        "DMP",
        "dmp",
        "dmp_esn",
        "ours_jacobian",
        "DMPESN",
        "DMP ESN",
        "ImprovedJacobianPruningESN",
    ],
}

MODEL_ROWS = [
    ("base_esn", "Base ESN", None),
    ("leaky_esn", "Leaky ESN", None),
    ("deep_esn", "Deep ESN", None),
    ("betweenness_esn", "Betweenness", "__DEFAULT_PRUNE__"),
    ("closeness_esn", "Closeness", "__DEFAULT_PRUNE__"),
    ("dmp", "DMP", "__DEFAULT_PRUNE__"),
]

LOOKUP_COLUMNS = [
    "dataset",
    "model",
    "horizon",
    "reservoir",
    "seed",
    "prune_ratio",
    "test_mse",
    "total_time_s",
    "memory_mb",
]


def _canonical_model_name(name) -> Optional[str]:
    if name is None:
        return None
    text = str(name)
    return MODEL_ALIASES.get(text, text)


def _model_search_names(model_name: str | Iterable[str]) -> list[str]:
    if isinstance(model_name, str):
        canonical = _canonical_model_name(model_name) or model_name
        return MODEL_SEARCH_NAMES.get(canonical, [model_name])

    out = []
    for name in model_name:
        canonical = _canonical_model_name(name) or str(name)
        out.extend(MODEL_SEARCH_NAMES.get(canonical, [str(name)]))

    seen = set()
    unique = []
    for x in out:
        if x not in seen:
            unique.append(x)
            seen.add(x)

    return unique


# ==============================================================
# General helpers
# ==============================================================

def _dataset_label(name: str) -> str:
    return DATASET_LABELS.get(str(name), str(name).replace("_", " "))


def _write_text(save_path: Path, text: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        old_text = save_path.read_text(encoding="utf-8")
        # Do not overwrite an existing table with a strictly more incomplete one.
        if old_text.count("XXX") < text.count("XXX"):
            return

    save_path.write_text(text, encoding="utf-8")


def _to_int(value) -> Optional[int]:
    try:
        return int(float(value))
    except Exception:
        return None


def _to_float(value) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None

    return None if math.isnan(out) else out


def _normalize_ratio_key(value) -> Optional[float]:
    x = _to_float(value)

    if x is None:
        return None

    if x > 1.0:
        x = x / 100.0

    return round(float(x), 6)


def _fmt(value: Optional[float], digits: int = 6, missing: str = "XXX") -> str:
    if value is None:
        return missing

    return f"{float(value):.{digits}f}"


def _best_idx(values: list[Optional[float]]) -> Optional[int]:
    valid = [(i, v) for i, v in enumerate(values) if v is not None]

    if not valid:
        return None

    return min(valid, key=lambda x: x[1])[0]


def _mean_std_ci95(values: list[float]) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None, None

    n = len(values)
    mean = float(sum(values) / n)

    if n == 1:
        return mean, 0.0, mean, mean

    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    half = 1.96 * std / math.sqrt(n)

    return mean, std, mean - half, mean + half


def _stats_cell(values: list[float], bold: bool = False) -> str:
    mean, std, lo, hi = _mean_std_ci95(values)

    if mean is None:
        core = "XXX \\pm XXX"
        core = f"\\mathbf{{{core}}}" if bold else core
        return f"${core}$ \\; [XXX, XXX]"

    core = f"{mean:.6f} \\pm {std:.6f}"

    if bold:
        core = f"\\mathbf{{{core}}}"

    return f"${core}$ \\; [{lo:.6f}, {hi:.6f}]"


# ==============================================================
# Loading metrics
# ==============================================================

def _parse_path_info(metrics_path: Path, root: Path):
    rel = metrics_path.relative_to(root)
    parts = rel.parts

    # Supported structures:
    # dataset / h_20 / model / reservoir / seed_0 / metrics.csv
    # dataset / h_20 / model / bulk_20pct / reservoir / seed_0 / metrics.csv
    # dataset / h_20 / model / prune_20pct / reservoir / seed_0 / metrics.csv

    dataset = parts[0] if len(parts) > 0 else None
    horizon = _to_int(parts[1].replace("h_", "")) if len(parts) > 1 and parts[1].startswith("h_") else None
    model = parts[2] if len(parts) > 2 else None

    prune_ratio = None
    idx = 3

    if len(parts) > 3:
        tag = parts[3]
        if (tag.startswith("bulk_") or tag.startswith("prune_")) and tag.endswith("pct"):
            pct = _to_float(
                tag.replace("bulk_", "").replace("prune_", "").replace("pct", "")
            )
            prune_ratio = (pct / 100.0) if pct is not None else None
            idx = 4

    reservoir = _to_int(parts[idx]) if len(parts) > idx else None
    seed = (
        _to_int(parts[idx + 1].replace("seed_", ""))
        if len(parts) > idx + 1 and parts[idx + 1].startswith("seed_")
        else None
    )

    return dataset, horizon, model, reservoir, seed, prune_ratio


def _memory_from_row(row: dict) -> Optional[float]:
    gpu = _to_float(row.get("gpu_peak_mem_mb"))
    cpu = _to_float(row.get("cpu_rss_mb"))

    # Prefer GPU memory only when it is actually nonzero.
    if gpu is not None and gpu > 0.0:
        return gpu

    return cpu


def load_model_metrics(
    model_name: str | Iterable[str],
    results_root: str = "results",
    results_root_template: str = "results_unified_jacobian",
) -> pd.DataFrame:
    root = Path(results_root) / results_root_template
    search_names = _model_search_names(model_name)

    rows = []
    seen_paths = set()

    for search_name in search_names:
        paths = sorted(root.glob(f"**/{search_name}/**/metrics.csv"))

        for p in paths:
            if p in seen_paths:
                continue

            seen_paths.add(p)

            ds_p, h_p, model_p, res_p, seed_p, pr_p = _parse_path_info(p, root)

            with p.open("r", encoding="utf-8", newline="") as f:
                row = next(csv.DictReader(f), None)

            if not row:
                continue

            total_time_s = _to_float(row.get("total_time_s"))
            total_time_ms = _to_float(row.get("total_time_ms"))

            if total_time_s is None and total_time_ms is not None:
                total_time_s = total_time_ms / 1000.0

            model_raw = row.get("model") or model_p or search_name
            model_canonical = _canonical_model_name(model_raw)

            rec = {
                "dataset": row.get("dataset") or ds_p,
                "model": model_canonical,
                "horizon": _to_int(row.get("horizon")) if row.get("horizon") not in (None, "") else h_p,
                "reservoir": _to_int(row.get("reservoir")) if row.get("reservoir") not in (None, "") else res_p,
                "seed": _to_int(row.get("seed")) if row.get("seed") not in (None, "") else seed_p,
                "prune_ratio": (
                    _normalize_ratio_key(row.get("prune_ratio"))
                    if row.get("prune_ratio") not in (None, "")
                    else _normalize_ratio_key(pr_p)
                ),
                "test_mse": _to_float(row.get("test_mse")),
                "total_time_s": total_time_s,
                "gpu_peak_mem_mb": _to_float(row.get("gpu_peak_mem_mb")),
                "cpu_rss_mb": _to_float(row.get("cpu_rss_mb")),
                "memory_mb": _memory_from_row(row),
                "__metrics_path": str(p),
            }

            if None in (rec["dataset"], rec["model"], rec["horizon"], rec["reservoir"], rec["seed"], rec["test_mse"]):
                continue

            rows.append(rec)

    return pd.DataFrame(rows)


def load_ours_metrics(
    results_root: str = "results",
    results_root_template: str = "results_unified_jacobian",
    model_name: str = "DMP",
) -> pd.DataFrame:
    return load_model_metrics(
        model_name=model_name,
        results_root=results_root,
        results_root_template=results_root_template,
    )


def aggregate_ours_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "reservoir",
                "horizon",
                "prune_ratio",
                "test_mse",
                "total_time_s",
                "memory_mb",
            ]
        )

    df = metrics_df.copy()

    for col in ["reservoir", "horizon", "prune_ratio", "seed", "test_mse", "total_time_s", "memory_mb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.groupby(
            ["dataset", "reservoir", "horizon", "prune_ratio"],
            dropna=False,
        )[["test_mse", "total_time_s", "memory_mb"]]
        .mean()
        .reset_index()
    )


# ==============================================================
# Lookup index
# ==============================================================

def _build_best_index(frames: list[pd.DataFrame]) -> dict:
    best = {}

    for df in frames:
        if df.empty:
            continue

        for _, row in df.iterrows():
            model_name = _canonical_model_name(row.get("model"))

            key = (
                str(row.get("dataset")),
                _to_int(row.get("horizon")),
                _to_int(row.get("reservoir")),
                model_name,
                _normalize_ratio_key(row.get("prune_ratio")),
                _to_int(row.get("seed")),
            )

            if None in (key[0], key[1], key[2], key[3], key[5]):
                continue

            mse = _to_float(row.get("test_mse"))
            if mse is None:
                continue

            candidate = {
                "test_mse": mse,
                "total_time_s": _to_float(row.get("total_time_s")),
                "memory_mb": _to_float(row.get("memory_mb")),
            }

            existing = best.get(key)
            if existing is None or candidate["test_mse"] < existing["test_mse"]:
                best[key] = candidate

    return best


def _best_index_to_df(best_index: dict) -> pd.DataFrame:
    rows = []

    for (dataset, horizon, reservoir, model, prune_ratio, seed), vals in best_index.items():
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "horizon": horizon,
                "reservoir": reservoir,
                "seed": seed,
                "prune_ratio": prune_ratio,
                "test_mse": vals.get("test_mse"),
                "total_time_s": vals.get("total_time_s"),
                "memory_mb": vals.get("memory_mb"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=LOOKUP_COLUMNS)

    return pd.DataFrame(rows, columns=LOOKUP_COLUMNS)


def _load_lookup_df(lookup_path: Path) -> pd.DataFrame:
    lookup_path = Path(lookup_path)

    if not lookup_path.exists():
        return pd.DataFrame(columns=LOOKUP_COLUMNS)

    try:
        df = pd.read_csv(lookup_path)
    except Exception:
        return pd.DataFrame(columns=LOOKUP_COLUMNS)

    for c in LOOKUP_COLUMNS:
        if c not in df.columns:
            df[c] = None

    df = df[LOOKUP_COLUMNS].copy()
    df["model"] = df["model"].map(_canonical_model_name)

    return df


def _observed_reservoirs(best_index: dict) -> list[int]:
    vals = sorted({int(k[2]) for k in best_index.keys() if _to_int(k[2]) is not None})
    return vals


def _observed_horizons(best_index: dict) -> list[int]:
    vals = sorted({int(k[1]) for k in best_index.keys() if _to_int(k[1]) is not None})
    return vals


def _observed_prunes_for_model(best_index: dict, model: str) -> list[float]:
    canonical = _canonical_model_name(model)

    vals = sorted(
        {
            float(k[4])
            for k in best_index.keys()
            if str(k[3]) == str(canonical) and k[4] is not None and _to_float(k[4]) is not None
        }
    )

    return vals


def _lookup_entry(
    best_index: dict,
    dataset: str,
    model: str,
    horizon: int,
    reservoir: int,
    seed: int,
    prune_ratio: Optional[float],
) -> Optional[dict]:
    key = (
        str(dataset),
        int(horizon),
        int(reservoir),
        _canonical_model_name(model),
        _normalize_ratio_key(prune_ratio),
        int(seed),
    )

    return best_index.get(key)


def _resolve_model_prune(prune_flag, default_prune_ratio: float) -> Optional[float]:
    if prune_flag == "__DEFAULT_PRUNE__":
        return default_prune_ratio

    return prune_flag


def _get_dataset_order(frames: list[pd.DataFrame]) -> list[str]:
    present = set()

    for df in frames:
        if df.empty or "dataset" not in df.columns:
            continue

        present.update([str(x) for x in df["dataset"].dropna().tolist()])

    ordered = [d for d in DATASET_ORDER if d in present]
    ordered.extend(sorted([d for d in present if d not in set(ordered)]))

    return ordered


def _get_seed_order(frames: list[pd.DataFrame]) -> list[int]:
    seeds = set()

    for df in frames:
        if df.empty or "seed" not in df.columns:
            continue

        for s in df["seed"].dropna().tolist():
            si = _to_int(s)
            if si is not None:
                seeds.add(si)

    if not seeds:
        return [0, 1, 2, 3, 4]

    return sorted(seeds)


def _values_over_seeds(
    best_index: dict,
    seeds: Iterable[int],
    dataset: str,
    model: str,
    horizon: int,
    reservoir: int,
    prune_ratio: Optional[float],
) -> list[float]:
    out = []

    for seed in seeds:
        entry = _lookup_entry(
            best_index,
            dataset,
            model,
            horizon,
            reservoir,
            int(seed),
            prune_ratio,
        )

        if entry is not None and entry.get("test_mse") is not None:
            out.append(float(entry["test_mse"]))

    return out


# ==============================================================
# Table renderers
# ==============================================================

def _render_main_comparison_seedwise(
    save_dir: Path,
    best_index: dict,
    datasets: list[str],
    seeds: list[int],
    reservoirs: list[int],
    horizon: int,
    default_prune_ratio: float,
) -> None:
    for res in reservoirs:
        lines = [f"% Auto-generated: seed-wise model comparison | horizon={horizon}, reservoir={res}", ""]

        for seed in seeds:
            lines.extend(
                [
                    "\\begin{table*}[t]",
                    "\\centering",
                    f"\\caption{{Seed {seed}: model comparison at horizon $h={horizon}$, reservoir {res}. All values are per-seed MSE and runtime; lower is better.}}",
                    "\\small",
                    "\\resizebox{\\linewidth}{!}{",
                    "\\begin{tabular}{llcc}",
                    "\\toprule",
                    f"Dataset & Model & MSE $\\downarrow$ & Runtime (s) $\\downarrow$ {NL}",
                    "\\midrule",
                ]
            )

            for ds_idx, ds in enumerate(datasets):
                mse_values = []
                rows = []

                for model_key, model_label, prune_flag in MODEL_ROWS:
                    prune_ratio = _resolve_model_prune(prune_flag, default_prune_ratio)
                    entry = _lookup_entry(best_index, ds, model_key, horizon, res, seed, prune_ratio)

                    if entry is None:
                        continue

                    mse = entry.get("test_mse")
                    runtime = entry.get("total_time_s")

                    rows.append((model_label, mse, runtime))
                    mse_values.append(mse)

                if not rows:
                    lines.append(f"{_dataset_label(ds)} & --- & XXX & XXX {NL}")
                    if ds_idx < len(datasets) - 1:
                        lines.append("\\midrule")
                    continue

                best = _best_idx(mse_values)

                for row_idx, (model_label, mse, runtime) in enumerate(rows):
                    ds_label = _dataset_label(ds) if row_idx == 0 else ""
                    ds_cell = f"\\multirow{{{len(rows)}}}{{*}}{{{ds_label}}}" if row_idx == 0 else ""

                    mse_txt = _fmt(mse, digits=6)

                    if best is not None and row_idx == best and mse is not None:
                        mse_txt = f"\\textbf{{{mse_txt}}}"

                    lines.append(f"{ds_cell} & {model_label} & {mse_txt} & {_fmt(runtime, digits=3)} {NL}")

                if ds_idx < len(datasets) - 1:
                    lines.append("\\midrule")

            lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""])

        _write_text(save_dir / f"dmp_main_comparison_res{res}_h{horizon}.tex", "\n".join(lines))


def _render_old_vs_ours_scaling_seedwise(
    save_dir: Path,
    best_index: dict,
    datasets: list[str],
    seeds: list[int],
    reservoirs: list[int],
    horizon: int,
    default_prune_ratio: float,
) -> None:
    col_spec = "l" + ("c" * len(reservoirs))
    header = " & ".join([str(r) for r in reservoirs])

    lines = [
        f"% Auto-generated: seed-wise Base ESN vs DMP scaling | horizon={horizon}, prune={int(default_prune_ratio * 100)}%",
        "",
    ]

    for seed in seeds:
        lines.extend(
            [
                "\\begin{table*}[t]",
                "\\centering",
                f"\\caption{{Seed {seed}: Base ESN vs DMP across reservoir sizes (h={horizon}, prune={int(default_prune_ratio * 100)}\\% for DMP). MSE only.}}",
                "\\small",
                "\\resizebox{\\linewidth}{!}{",
                f"\\begin{{tabular}}{{{col_spec}}}",
                "\\toprule",
                f"Dataset / Model & {header} {NL}",
                "\\midrule",
            ]
        )

        for ds_idx, ds in enumerate(datasets):
            n_cols = len(reservoirs) + 1
            lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{_dataset_label(ds)}}}}} {NL}")

            base_vals = []
            dmp_vals = []

            for res in reservoirs:
                b = _lookup_entry(best_index, ds, "base_esn", horizon, res, seed, None)
                o = _lookup_entry(best_index, ds, "dmp", horizon, res, seed, default_prune_ratio)

                base_vals.append(b["test_mse"] if b else None)
                dmp_vals.append(o["test_mse"] if o else None)

            base_cells = []
            dmp_cells = []

            for b, o in zip(base_vals, dmp_vals):
                b_txt = _fmt(b, digits=6)
                o_txt = _fmt(o, digits=6)

                if b is not None and o is not None:
                    if b <= o:
                        b_txt = f"\\textbf{{{b_txt}}}"
                    else:
                        o_txt = f"\\textbf{{{o_txt}}}"

                base_cells.append(b_txt)
                dmp_cells.append(o_txt)

            lines.append("Base ESN & " + " & ".join(base_cells) + f" {NL}")
            lines.append("DMP & " + " & ".join(dmp_cells) + f" {NL}")

            if ds_idx < len(datasets) - 1:
                lines.append("\\midrule")

        lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""])

    _write_text(save_dir / f"dmp_base_vs_dmp_scaling_h{horizon}_seedwise.tex", "\n".join(lines))


def _render_ours_prune_ratio_seedwise(
    save_dir: Path,
    best_index: dict,
    datasets: list[str],
    seeds: list[int],
    reservoirs: list[int],
    horizon: int,
    prune_ratios: list[float],
) -> None:
    ratios = [_normalize_ratio_key(x) for x in prune_ratios if _normalize_ratio_key(x) is not None]
    ratios = sorted(set(ratios))

    for res in reservoirs:
        header = " & ".join([f"{int(r * 100)}\\%" for r in ratios])
        col_spec = "l" + ("c" * len(ratios))
        lines = [f"% Auto-generated: seed-wise DMP prune-ratio study | horizon={horizon}, reservoir={res}", ""]

        for seed in seeds:
            lines.extend(
                [
                    "\\begin{table}[!h]",
                    "\\centering",
                    f"\\caption{{Seed {seed}: DMP pruning-ratio sensitivity at horizon $h={horizon}$, reservoir {res}. MSE only.}}",
                    "\\small",
                    "\\resizebox{\\linewidth}{!}{",
                    f"\\begin{{tabular}}{{{col_spec}}}",
                    "\\toprule",
                    f"Dataset & {header} {NL}",
                    "\\midrule",
                ]
            )

            for ds in datasets:
                vals = []

                for ratio in ratios:
                    entry = _lookup_entry(best_index, ds, "dmp", horizon, res, seed, ratio)
                    vals.append(entry["test_mse"] if entry else None)

                best = _best_idx(vals)
                cells = []

                for i, v in enumerate(vals):
                    txt = _fmt(v, digits=6)

                    if best is not None and i == best and v is not None:
                        txt = f"\\textbf{{{txt}}}"

                    cells.append(txt)

                lines.append(f"{_dataset_label(ds)} & " + " & ".join(cells) + f" {NL}")

            lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])

        _write_text(save_dir / f"dmp_prune_ratio_res{res}_h{horizon}.tex", "\n".join(lines))


def _render_ours_horizon_seedwise(
    save_dir: Path,
    best_index: dict,
    datasets: list[str],
    seeds: list[int],
    reservoirs: list[int],
    horizons: list[int],
    default_prune_ratio: float,
) -> None:
    horizons = sorted([int(h) for h in horizons])

    for res in reservoirs:
        header = " & ".join([f"\\textbf{{$h={h}$}}" for h in horizons])
        col_spec = "l" + ("c" * len(horizons))
        lines = [f"% Auto-generated: seed-wise DMP horizon analysis | reservoir={res}, prune={int(default_prune_ratio * 100)}%", ""]

        for seed in seeds:
            lines.extend(
                [
                    "\\begin{table}[!h]",
                    "\\centering",
                    f"\\caption{{Seed {seed}: DMP horizon sensitivity at reservoir {res} (pruning {int(default_prune_ratio * 100)}\\%). Lower is better.}}",
                    "\\small",
                    f"\\begin{{tabular}}{{{col_spec}}}",
                    "\\toprule",
                    f"\\textbf{{Dataset}} & {header} {NL}",
                    "\\midrule",
                ]
            )

            for ds in datasets:
                vals = []

                for h in horizons:
                    entry = _lookup_entry(best_index, ds, "dmp", h, res, seed, default_prune_ratio)
                    vals.append(entry["test_mse"] if entry else None)

                lines.append(f"{_dataset_label(ds)} & " + " & ".join([_fmt(v, digits=6) for v in vals]) + f" {NL}")

            lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

        _write_text(save_dir / f"dmp_horizon_res{res}.tex", "\n".join(lines))


def _render_old_vs_ours_ci_with_seed_tables(
    save_dir: Path,
    best_index: dict,
    datasets: list[str],
    seeds: list[int],
    reservoirs: list[int],
    horizon: int,
    default_prune_ratio: float,
) -> None:
    for res in reservoirs:
        lines = [
            f"% Auto-generated: Base ESN vs DMP CI + seed tables | horizon={horizon}, reservoir={res}, prune={int(default_prune_ratio * 100)}%",
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Mean test MSE over available seeds. Values are mean $\\pm$ standard deviation, with 95\\% confidence intervals in brackets. Lower is better.}",
            f"\\label{{tab:ci_res{res}_h{horizon}}}",
            "\\setlength{\\tabcolsep}{6pt}",
            "\\renewcommand{\\arraystretch}{1.15}",
            "\\small",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            f"\\textbf{{Dataset}} & \\textbf{{Base ESN}} & \\textbf{{DMP}} {NL}",
            "\\midrule",
        ]

        for ds in datasets:
            base_vals = _values_over_seeds(best_index, seeds, ds, "base_esn", horizon, res, None)
            dmp_vals = _values_over_seeds(best_index, seeds, ds, "dmp", horizon, res, default_prune_ratio)

            base_mean = _mean_std_ci95(base_vals)[0] if base_vals else None
            dmp_mean = _mean_std_ci95(dmp_vals)[0] if dmp_vals else None

            base_better = False
            dmp_better = False

            if base_mean is not None and dmp_mean is not None:
                base_better = base_mean < dmp_mean
                dmp_better = dmp_mean < base_mean

            base_cell = _stats_cell(base_vals, bold=base_better)
            dmp_cell = _stats_cell(dmp_vals, bold=dmp_better)

            lines.append(f"{_dataset_label(ds)} ")
            lines.append(f"& {base_cell} ")
            lines.append(f"& {dmp_cell} {NL}")
            lines.append("")

        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])

        for seed in seeds:
            lines.extend(
                [
                    "\\begin{table}[!h]",
                    "\\centering",
                    f"\\caption{{Seed {seed}: Base ESN vs DMP MSE at horizon $h={horizon}$, reservoir {res}, prune {int(default_prune_ratio * 100)}\\%.}}",
                    "\\small",
                    "\\begin{tabular}{lcc}",
                    "\\toprule",
                    f"Dataset & Base ESN & DMP {NL}",
                    "\\midrule",
                ]
            )

            for ds in datasets:
                base_entry = _lookup_entry(best_index, ds, "base_esn", horizon, res, seed, None)
                dmp_entry = _lookup_entry(best_index, ds, "dmp", horizon, res, seed, default_prune_ratio)

                b = base_entry["test_mse"] if base_entry else None
                o = dmp_entry["test_mse"] if dmp_entry else None

                b_txt = _fmt(b, digits=6)
                o_txt = _fmt(o, digits=6)

                if b is not None and o is not None:
                    if b <= o:
                        b_txt = f"\\textbf{{{b_txt}}}"
                    else:
                        o_txt = f"\\textbf{{{o_txt}}}"

                lines.append(f"{_dataset_label(ds)} & {b_txt} & {o_txt} {NL}")

            lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

        _write_text(save_dir / f"dmp_ci_base_vs_dmp_res{res}_h{horizon}.tex", "\n".join(lines))


def _clear_tex_outputs(save_dir: Path) -> None:
    if not save_dir.exists():
        return

    for p in save_dir.glob("*.tex"):
        p.unlink()


# ==============================================================
# Main public function
# ==============================================================

def save_all_ours_latex_tables(
    save_dir: str | Path,
    results_root: str = "results",
    results_root_template: str = "results_unified_jacobian",
    model_name: str = "DMP",
    benchmark_reservoirs: Iterable[int] = (300, 500, 700, 1000),
    scaling_sizes: Iterable[int] = (100, 200, 300, 500, 700, 1000),
    horizons: Iterable[int] = (10, 20, 30),
    prune_ratios: Iterable[float] = (0.10, 0.20, 0.30),
    default_horizon: int = 20,
    default_prune_ratio: float = 0.20,
    clear_existing_tex: bool = False,
) -> pd.DataFrame:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if clear_existing_tex:
        _clear_tex_outputs(save_dir)

    lookup_path = save_dir / "metrics_lookup.csv"
    lookup_df = _load_lookup_df(lookup_path)

    base_df = load_model_metrics("base_esn", results_root, results_root_template)
    leaky_df = load_model_metrics("leaky_esn", results_root, results_root_template)
    deep_df = load_model_metrics("deep_esn", results_root, results_root_template)
    betw_df = load_model_metrics("betweenness_esn", results_root, results_root_template)
    close_df = load_model_metrics("closeness_esn", results_root, results_root_template)
    ours_df = load_model_metrics(model_name, results_root, results_root_template)

    frames = [lookup_df, base_df, leaky_df, deep_df, betw_df, close_df, ours_df]

    best_index = _build_best_index(frames)
    merged_df = _best_index_to_df(best_index)

    if not merged_df.empty:
        merged_df = merged_df.sort_values(
            by=["dataset", "model", "horizon", "reservoir", "seed", "prune_ratio"],
            kind="stable",
        )

    merged_df.to_csv(lookup_path, index=False)

    datasets = _get_dataset_order([merged_df])
    seeds = _get_seed_order([merged_df])

    observed_res = _observed_reservoirs(best_index)
    pref_res = [int(x) for x in benchmark_reservoirs]

    reservoirs = [r for r in pref_res if r in observed_res] + [
        r for r in observed_res if r not in set(pref_res)
    ]

    if not reservoirs:
        reservoirs = sorted(set(pref_res))

    observed_h = _observed_horizons(best_index)
    pref_h = sorted({int(x) for x in horizons})

    horizons_list = [h for h in pref_h if h in observed_h] + [
        h for h in observed_h if h not in set(pref_h)
    ]

    if not horizons_list:
        horizons_list = pref_h

    observed_dmp_prunes = _observed_prunes_for_model(best_index, model_name)
    pref_prunes = sorted(
        {
            _normalize_ratio_key(float(x))
            for x in prune_ratios
            if _normalize_ratio_key(float(x)) is not None
        }
    )

    prune_list = [p for p in pref_prunes if p in observed_dmp_prunes] + [
        p for p in observed_dmp_prunes if p not in set(pref_prunes)
    ]

    if not prune_list:
        prune_list = pref_prunes

    default_h = int(default_horizon)

    if observed_h and default_h not in set(observed_h):
        default_h = int(horizons_list[0])

    default_p = _normalize_ratio_key(default_prune_ratio)

    if default_p is None:
        default_p = 0.2

    if prune_list and default_p not in set(prune_list):
        default_p = float(prune_list[0])

    def _has_any(model: str, horizon: int, reservoir: int, prune_ratio: Optional[float]) -> bool:
        for ds in datasets:
            for sd in seeds:
                if _lookup_entry(
                    best_index,
                    ds,
                    model,
                    int(horizon),
                    int(reservoir),
                    int(sd),
                    prune_ratio,
                ) is not None:
                    return True
        return False

    comparison_res = [
        r
        for r in reservoirs
        if any(
            _has_any(mk, default_h, r, _resolve_model_prune(pf, default_p))
            for mk, _ml, pf in MODEL_ROWS
        )
    ]

    if not comparison_res:
        comparison_res = reservoirs

    scaling_res = [
        r
        for r in reservoirs
        if _has_any("base_esn", default_h, r, None)
        or _has_any("dmp", default_h, r, default_p)
    ]

    if not scaling_res:
        scaling_res = reservoirs

    prune_res = [
        r
        for r in reservoirs
        if _has_any("dmp", default_h, r, None)
        or any(_has_any("dmp", default_h, r, pr) for pr in prune_list)
    ]

    if not prune_res:
        prune_res = reservoirs

    horizon_res = [
        r
        for r in reservoirs
        if any(_has_any("dmp", h, r, default_p) for h in horizons_list)
    ]

    if not horizon_res:
        horizon_res = reservoirs

    ci_res = [
        r
        for r in reservoirs
        if _has_any("base_esn", default_h, r, None)
        and _has_any("dmp", default_h, r, default_p)
    ]

    if not ci_res:
        ci_res = scaling_res

    _render_main_comparison_seedwise(
        save_dir=save_dir,
        best_index=best_index,
        datasets=datasets,
        seeds=seeds,
        reservoirs=comparison_res,
        horizon=default_h,
        default_prune_ratio=float(default_p),
    )

    _render_old_vs_ours_scaling_seedwise(
        save_dir=save_dir,
        best_index=best_index,
        datasets=datasets,
        seeds=seeds,
        reservoirs=scaling_res,
        horizon=default_h,
        default_prune_ratio=float(default_p),
    )

    _render_ours_prune_ratio_seedwise(
        save_dir=save_dir,
        best_index=best_index,
        datasets=datasets,
        seeds=seeds,
        reservoirs=prune_res,
        horizon=default_h,
        prune_ratios=prune_list,
    )

    _render_ours_horizon_seedwise(
        save_dir=save_dir,
        best_index=best_index,
        datasets=datasets,
        seeds=seeds,
        reservoirs=horizon_res,
        horizons=horizons_list,
        default_prune_ratio=float(default_p),
    )

    _render_old_vs_ours_ci_with_seed_tables(
        save_dir=save_dir,
        best_index=best_index,
        datasets=datasets,
        seeds=seeds,
        reservoirs=ci_res,
        horizon=default_h,
        default_prune_ratio=float(default_p),
    )

    return aggregate_ours_metrics(ours_df)


# ==============================================================
# Compatibility wrappers retained for external callers
# ==============================================================

def save_ours_benchmark_tables_per_dataset(*args, **kwargs) -> None:
    return None


def save_ours_scaling_tables_per_dataset(*args, **kwargs) -> None:
    return None


def save_ours_horizon_tables_by_reservoir(*args, **kwargs) -> None:
    return None


def save_ours_prune_ratio_tables_by_reservoir(*args, **kwargs) -> None:
    return None


def save_main_comparison_tables_by_reservoir(*args, **kwargs) -> None:
    return None


def save_old_vs_ours_ci_tables_by_reservoir(*args, **kwargs) -> None:
    return None


__all__ = [
    "load_ours_metrics",
    "load_model_metrics",
    "aggregate_ours_metrics",
    "save_ours_benchmark_tables_per_dataset",
    "save_ours_scaling_tables_per_dataset",
    "save_ours_horizon_tables_by_reservoir",
    "save_ours_prune_ratio_tables_by_reservoir",
    "save_main_comparison_tables_by_reservoir",
    "save_old_vs_ours_ci_tables_by_reservoir",
    "save_all_ours_latex_tables",
]