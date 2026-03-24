from types import SimpleNamespace

import pandas as pd
import torch

from scripts.export_lstm_attack_assets import (
    build_alpha158_handler_kwargs,
    build_matched_reference,
    export_matched_feature_windows,
    prepare_feature_test_split,
    extract_normalization_stats,
    filter_matched_reference_by_keys,
    select_matched_rows_by_keys,
)
from scripts.export_whitebox_attack_assets import build_matched_reference as unified_build_matched_reference


def test_build_matched_reference_keeps_only_intersection():
    pred_index = pd.MultiIndex.from_tuples(
        [
            ("2025-01-02", "AAA"),
            ("2025-01-02", "BBB"),
        ],
        names=["datetime", "instrument"],
    )
    label_index = pd.MultiIndex.from_tuples(
        [
            ("2025-01-02", "BBB"),
            ("2025-01-03", "CCC"),
        ],
        names=["datetime", "instrument"],
    )
    pred = pd.DataFrame({"score": [0.1, 0.2]}, index=pred_index)
    label = pd.DataFrame({"label": [1.2, 1.3]}, index=label_index)

    matched = build_matched_reference(pred, label)

    assert list(matched.index) == [(pd.Timestamp("2025-01-02"), "BBB")]
    assert matched.loc[(pd.Timestamp("2025-01-02"), "BBB"), "score"] == 0.2
    assert matched.loc[(pd.Timestamp("2025-01-02"), "BBB"), "label"] == 1.2


def test_unified_export_entry_reuses_matched_reference_builder():
    pred_index = pd.MultiIndex.from_tuples(
        [
            ("2025-01-02", "AAA"),
            ("2025-01-02", "BBB"),
        ],
        names=["datetime", "instrument"],
    )
    label_index = pd.MultiIndex.from_tuples(
        [
            ("2025-01-02", "BBB"),
            ("2025-01-03", "CCC"),
        ],
        names=["datetime", "instrument"],
    )
    pred = pd.DataFrame({"score": [0.1, 0.2]}, index=pred_index)
    label = pd.DataFrame({"label": [1.2, 1.3]}, index=label_index)

    old_matched = build_matched_reference(pred, label)
    new_matched = unified_build_matched_reference(pred, label)

    pd.testing.assert_frame_equal(new_matched, old_matched)


def test_build_matched_reference_respects_optional_date_range():
    pred_index = pd.MultiIndex.from_tuples(
        [
            ("2025-01-02", "AAA"),
            ("2025-11-02", "BBB"),
        ],
        names=["datetime", "instrument"],
    )
    label_index = pred_index
    pred = pd.DataFrame({"score": [0.1, 0.2]}, index=pred_index)
    label = pd.DataFrame({"label": [1.1, 1.2]}, index=label_index)

    matched = build_matched_reference(
        pred,
        label,
        date_from="2025-01-01",
        date_to="2025-10-31",
    )

    assert list(matched.index) == [(pd.Timestamp("2025-01-02"), "AAA")]


def test_extract_normalization_stats_preserves_feature_order():
    processor = SimpleNamespace(
        cols=[("feature", "B"), ("feature", "A")],
        mean_train=[2.0, 1.0],
        std_train=[4.0, 3.0],
        clip_outlier=True,
    )

    stats = extract_normalization_stats(processor=processor, expected_features=["A", "B"])

    assert stats["feature_names"] == ["A", "B"]
    assert stats["center"] == [1.0, 2.0]
    assert stats["scale"] == [3.0, 4.0]
    assert stats["clip_outlier"] is True


def test_export_matched_feature_windows_aligns_windows_by_key():
    matched_index = pd.MultiIndex.from_tuples(
        [
            ("2025-01-02", "AAA"),
            ("2025-01-03", "BBB"),
        ],
        names=["datetime", "instrument"],
    )
    matched_reference = pd.DataFrame(
        {
            "score": [0.1, 0.2],
            "label": [0.3, 0.4],
        },
        index=matched_index,
    )

    class FakeFeatureSplit:
        def __init__(self):
            self._index = pd.MultiIndex.from_tuples(
                [
                    ("2025-01-01", "AAA"),
                    ("2025-01-02", "AAA"),
                    ("2025-01-03", "BBB"),
                ],
                names=["datetime", "instrument"],
            )
            self._items = [
                torch.zeros(20, 21, dtype=torch.float32).numpy(),
                torch.full((20, 21), 1.0, dtype=torch.float32).numpy(),
                torch.full((20, 21), 2.0, dtype=torch.float32).numpy(),
            ]

        def get_index(self):
            return self._index

        def __getitem__(self, idx):
            return self._items[idx]

    feature_split = FakeFeatureSplit()

    asset = export_matched_feature_windows(matched_reference, feature_split, feature_dim=20)

    assert asset["features"].shape == (2, 20, 20)
    assert torch.allclose(asset["features"][0], torch.full((20, 20), 1.0))
    assert torch.allclose(asset["features"][1], torch.full((20, 20), 2.0))
    assert asset["missing_keys"] == []


def test_export_matched_raw_windows_skips_non_finite_samples():
    from scripts.export_lstm_attack_assets import export_matched_raw_windows

    matched_index = pd.MultiIndex.from_tuples(
        [
            ("2025-01-02", "AAA"),
            ("2025-01-03", "BBB"),
        ],
        names=["datetime", "instrument"],
    )
    matched_reference = pd.DataFrame(
        {
            "score": [0.1, 0.2],
            "label": [0.3, 0.4],
        },
        index=matched_index,
    )

    class FakeRawSplit:
        def __init__(self):
            self._index = pd.MultiIndex.from_tuples(
                [
                    ("2025-01-02", "AAA"),
                    ("2025-01-03", "BBB"),
                ],
                names=["datetime", "instrument"],
            )
            good = torch.ones(80, 6, dtype=torch.float32).numpy()
            bad = torch.ones(80, 6, dtype=torch.float32).numpy()
            bad[:5, :5] = float("nan")
            self._items = [good, bad]

        def get_index(self):
            return self._index

        def __getitem__(self, idx):
            return self._items[idx]

    asset = export_matched_raw_windows(matched_reference, FakeRawSplit())

    assert asset["ohlcv"].shape == (1, 80, 5)
    assert asset["keys"] == [("2025-01-02 00:00:00", "AAA")]
    assert asset["missing_keys"] == [("2025-01-03 00:00:00", "BBB")]


def test_build_alpha158_handler_kwargs_matches_legacy_pipeline():
    args = SimpleNamespace(
        start_time="2019-01-01",
        end_time="2025-12-31",
        fit_start_time="2019-01-01",
        fit_end_time="2023-12-31",
        market="all",
        label_expr="Ref($close, -2) / Ref($close, -1) - 1",
    )

    kwargs = build_alpha158_handler_kwargs(args)

    assert kwargs["start_time"] == "2019-01-01"
    assert kwargs["fit_end_time"] == "2023-12-31"
    assert kwargs["instruments"] == "all"
    assert kwargs["label"] == ["Ref($close, -2) / Ref($close, -1) - 1"]
    assert kwargs["infer_processors"] == [
        {"class": "FilterCol", "kwargs": {"fields_group": "feature", "col_list": kwargs["infer_processors"][0]["kwargs"]["col_list"]}},
        {
            "class": "RobustZScoreNorm",
            "kwargs": {
                "fields_group": "feature",
                "clip_outlier": True,
                "fit_start_time": "2019-01-01",
                "fit_end_time": "2023-12-31",
            },
        },
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]
    assert kwargs["learn_processors"] == [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ]


def test_prepare_feature_test_split_uses_dk_i_and_ffill_bfill():
    class FakeSplit:
        def __init__(self):
            self.config_calls = []

        def config(self, **kwargs):
            self.config_calls.append(kwargs)

    class FakeDataset:
        def __init__(self):
            self.prepare_calls = []
            self.split = FakeSplit()

        def prepare(self, *args, **kwargs):
            self.prepare_calls.append((args, kwargs))
            return self.split

    dataset = FakeDataset()

    split = prepare_feature_test_split(dataset)

    assert split is dataset.split
    assert dataset.prepare_calls == [
        (
            ("test",),
            {
                "col_set": ["feature", "label"],
                "data_key": "infer",
            },
        )
    ]
    assert dataset.split.config_calls == [{"fillna_type": "ffill+bfill"}]


def test_select_matched_rows_by_keys_returns_requested_subset_in_order():
    sample_asset = {
        "keys": [
            ("2025-01-02 00:00:00", "AAA"),
            ("2025-01-02 00:00:00", "BBB"),
            ("2025-01-03 00:00:00", "CCC"),
        ],
        "ohlcv": torch.arange(3 * 2 * 5, dtype=torch.float32).reshape(3, 2, 5),
        "label": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        "score": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
    }

    subset = select_matched_rows_by_keys(
        sample_asset,
        requested_keys=[
            ("2025-01-03 00:00:00", "CCC"),
            ("2025-01-02 00:00:00", "AAA"),
        ],
    )

    assert subset["keys"] == [
        ("2025-01-03 00:00:00", "CCC"),
        ("2025-01-02 00:00:00", "AAA"),
    ]
    assert subset["ohlcv"].shape == (2, 2, 5)
    assert torch.allclose(subset["label"], torch.tensor([0.3, 0.1], dtype=torch.float32))
    assert torch.allclose(subset["score"], torch.tensor([3.0, 1.0], dtype=torch.float32))


def test_select_matched_rows_by_keys_accepts_timestamp_requested_keys():
    sample_asset = {
        "keys": [
            ("2025-01-02 00:00:00", "AAA"),
            ("2025-01-03 00:00:00", "CCC"),
        ],
        "ohlcv": torch.arange(2 * 2 * 5, dtype=torch.float32).reshape(2, 2, 5),
        "label": torch.tensor([0.1, 0.3], dtype=torch.float32),
        "score": torch.tensor([1.0, 3.0], dtype=torch.float32),
    }

    subset = select_matched_rows_by_keys(
        sample_asset,
        requested_keys=[
            (pd.Timestamp("2025-01-03"), "CCC"),
            (pd.Timestamp("2025-01-02"), "AAA"),
        ],
    )

    assert subset["keys"] == [
        ("2025-01-03 00:00:00", "CCC"),
        ("2025-01-02 00:00:00", "AAA"),
    ]


def test_filter_matched_reference_by_keys_preserves_requested_order():
    matched_reference = pd.DataFrame(
        {"score": [0.1, 0.2, 0.3], "label": [1.0, 2.0, 3.0]},
        index=pd.MultiIndex.from_tuples(
            [
                ("2025-01-02", "AAA"),
                ("2025-01-02", "BBB"),
                ("2025-01-03", "CCC"),
            ],
            names=["datetime", "instrument"],
        ),
    )

    filtered = filter_matched_reference_by_keys(
        matched_reference,
        requested_keys=[
            (pd.Timestamp("2025-01-03"), "CCC"),
            (pd.Timestamp("2025-01-02"), "AAA"),
        ],
    )

    assert list(filtered.index) == [
        (pd.Timestamp("2025-01-03"), "CCC"),
        (pd.Timestamp("2025-01-02"), "AAA"),
    ]
    assert filtered["score"].tolist() == [0.3, 0.1]
