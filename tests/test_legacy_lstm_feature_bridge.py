import pytest
import torch
from qlib.contrib.data.loader import Alpha158DL

from legacy_lstm_feature_bridge import (
    LEGACY_LSTM_FEATURES,
    LegacyLSTMFeatureBridge,
)


def test_legacy_lstm_feature_list_matches_export_order():
    requested = [
        "RESI5",
        "WVMA5",
        "RSQR5",
        "KLEN",
        "RSQR10",
        "CORR5",
        "CORD5",
        "CORR10",
        "ROC60",
        "RESI10",
        "VSTD5",
        "RSQR60",
        "CORR60",
        "WVMA60",
        "STD5",
        "RSQR20",
        "CORD60",
        "CORD10",
        "CORR20",
        "KLOW",
    ]
    _, alpha158_names = Alpha158DL.get_feature_config(
        {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
            "rolling": {},
        }
    )
    expected = [name for name in alpha158_names if name in requested]

    assert LEGACY_LSTM_FEATURES == expected


def test_legacy_lstm_feature_bridge_returns_20_steps_for_80_step_input():
    bridge = LegacyLSTMFeatureBridge()
    x = torch.rand(3, 80, 5)
    y = bridge(x)
    assert y.shape == (3, 20, 20)


def test_legacy_lstm_feature_bridge_rejects_short_input_windows():
    bridge = LegacyLSTMFeatureBridge()
    x = torch.rand(2, 79, 5)
    with pytest.raises(ValueError, match="80"):
        bridge(x)
