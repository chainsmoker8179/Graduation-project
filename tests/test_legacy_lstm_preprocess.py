import torch

from legacy_lstm_preprocess import FillnaLayer, RobustZScoreNormLayer


def test_robust_zscore_layer_matches_expected_formula():
    layer = RobustZScoreNormLayer(
        center=torch.tensor([1.0, 2.0]),
        scale=torch.tensor([2.0, 4.0]),
        clip_outlier=False,
    )
    x = torch.tensor([[[3.0, 6.0]]])
    y = layer(x)
    assert torch.allclose(y, torch.tensor([[[1.0, 1.0]]]))


def test_robust_zscore_layer_applies_optional_clipping():
    layer = RobustZScoreNormLayer(
        center=torch.tensor([0.0]),
        scale=torch.tensor([1.0]),
        clip_outlier=True,
    )
    x = torch.tensor([[[5.0], [-5.0]]])
    y = layer(x)
    assert torch.allclose(y, torch.tensor([[[3.0], [-3.0]]]))


def test_fillna_layer_replaces_nan_with_zero_by_default():
    layer = FillnaLayer()
    x = torch.tensor([[[1.0, float("nan")]]])
    y = layer(x)
    assert torch.allclose(y, torch.tensor([[[1.0, 0.0]]]))


def test_robust_zscore_layer_maps_nan_input_to_zero_feature_value() -> None:
    layer = RobustZScoreNormLayer(
        center=torch.tensor([2.0]),
        scale=torch.tensor([4.0]),
        clip_outlier=False,
    )
    x = torch.tensor([[[float("nan")]]])

    y = layer(x)

    assert torch.allclose(y, torch.tensor([[[0.0]]]))
