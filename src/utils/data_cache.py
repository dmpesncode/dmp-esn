import torch


def hydrate_esn_series(esn, series_raw: torch.Tensor) -> None:
    """
    Attach an already-loaded raw time series to an ESN instance.

    The ESN still applies its own preprocessing/normalization through
    _preprocess_data(), so behavior remains identical to load_data().
    """
    if series_raw is None:
        raise ValueError("series_raw must not be None.")

    if not isinstance(series_raw, torch.Tensor):
        series_raw = torch.as_tensor(series_raw)

    esn.data = esn._preprocess_data(series_raw.clone())


__all__ = [
    "hydrate_esn_series",
]