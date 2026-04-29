from .dmp_esn import (
    DMPConfig,
    DMPESN,
    DynamicalModePruner,
    rescale_reservoir_to_target_rho,
)

from .dmp_results_save import (
    DMPResultSaveConfig,
    DMPResultsSaver,
    save_dmp_artifacts,
    OursJacobianResultsSaver,
    OursResultSaveConfig,
    save_jacobian_artifacts,
)


# ==============================================================
# Backward-compatible aliases for older training scripts
# ==============================================================

ImprovedJacobianPruningConfig = DMPConfig
ImprovedJacobianPruningESN = DMPESN
StableTaskAwareJacobianPruner = DynamicalModePruner

JacobianPruningConfig = DMPConfig
JacobianPruningESN = DMPESN
DynamicalLeveragePruning = DynamicalModePruner

rescale_reservoir = rescale_reservoir_to_target_rho


__all__ = [
    # New DMP names
    "DMPConfig",
    "DMPESN",
    "DynamicalModePruner",
    "DMPResultSaveConfig",
    "DMPResultsSaver",
    "save_dmp_artifacts",
    "rescale_reservoir_to_target_rho",

    # Backward-compatible pruning aliases
    "ImprovedJacobianPruningConfig",
    "ImprovedJacobianPruningESN",
    "StableTaskAwareJacobianPruner",
    "JacobianPruningConfig",
    "JacobianPruningESN",
    "DynamicalLeveragePruning",
    "rescale_reservoir",

    # Backward-compatible result-saving aliases
    "OursResultSaveConfig",
    "OursJacobianResultsSaver",
    "save_jacobian_artifacts",
]