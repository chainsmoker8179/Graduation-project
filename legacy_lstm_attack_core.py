from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from legacy_lstm_feature_bridge import LegacyLSTMFeatureBridge
from legacy_lstm_predictor import load_legacy_lstm_from_files
from legacy_lstm_preprocess import FillnaLayer, RobustZScoreNormLayer
from whitebox_attack_core import (
    CleanGateMetrics,
    CleanGateThresholds,
    RawFeatureAttackPipeline,
    compute_input_gradients,
    fgsm_maximize_mse,
    pgd_maximize_mse,
    project_relative_box,
    relative_budget,
    run_clean_gate,
    spearman_correlation,
    usage_ratio,
    validate_clean_gate,
)


class LegacyRawLSTMPipeline(RawFeatureAttackPipeline):
    def __init__(
        self,
        normalization_stats: dict[str, Any],
        *,
        state_dict_path: Path,
        config_path: Path,
    ) -> None:
        bridge = LegacyLSTMFeatureBridge()
        norm = RobustZScoreNormLayer(
            center=torch.tensor(normalization_stats["center"], dtype=torch.float32),
            scale=torch.tensor(normalization_stats["scale"], dtype=torch.float32),
            clip_outlier=bool(normalization_stats["clip_outlier"]),
        )
        fillna = FillnaLayer()
        predictor = load_legacy_lstm_from_files(config_path=config_path, state_dict_path=state_dict_path)
        super().__init__(bridge=bridge, norm=norm, fillna=fillna, model=predictor)
        self.predictor = predictor


__all__ = [
    "CleanGateMetrics",
    "CleanGateThresholds",
    "LegacyRawLSTMPipeline",
    "RawFeatureAttackPipeline",
    "compute_input_gradients",
    "fgsm_maximize_mse",
    "pgd_maximize_mse",
    "project_relative_box",
    "relative_budget",
    "run_clean_gate",
    "spearman_correlation",
    "usage_ratio",
    "validate_clean_gate",
]
