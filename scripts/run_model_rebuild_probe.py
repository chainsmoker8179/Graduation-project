#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whitebox_model_probe import load_feature_model_from_config, load_probe_asset, run_clean_forward_probe


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe transformer/tcn rebuild path with clean forward only.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args(argv)


def _choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _render_readme(model_name: str, summary: dict[str, float]) -> str:
    return "\n".join(
        [
            f"# {model_name} Model Rebuild Probe",
            "",
            "## Summary",
            f"- sample_count: `{summary['sample_count']}`",
            f"- pred_finite_rate: `{summary['pred_finite_rate']:.6f}`",
            f"- pred_mean: `{summary['pred_mean']:.6f}`",
            f"- pred_std: `{summary['pred_std']:.6f}`",
            f"- mae_to_reference: `{summary['mae_to_reference']:.6f}`",
            f"- mse_to_reference: `{summary['mse_to_reference']:.6f}`",
            f"- spearman_to_reference: `{summary['spearman_to_reference']:.6f}`",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.out_dir is None:
        args.out_dir = Path("reports") / f"{args.model_name}_model_probe"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _choose_device(args.device)

    model = load_feature_model_from_config(
        config_path=args.config_path,
        state_dict_path=args.state_dict_path,
        device=device,
    )
    keys, features, reference_scores = load_probe_asset(args.asset_dir, max_samples=args.max_samples)
    summary, pred_df = run_clean_forward_probe(
        model=model,
        keys=keys,
        feature_windows=features,
        reference_scores=reference_scores,
        device=device,
    )

    summary_payload = {
        "model_name": args.model_name,
        "config_path": str(args.config_path),
        "state_dict_path": str(args.state_dict_path),
        "asset_dir": str(args.asset_dir),
        **summary,
    }

    summary_path = args.out_dir / "probe_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pred_df.to_csv(args.out_dir / "probe_predictions.csv", index=False)
    (args.out_dir / "README.md").write_text(_render_readme(args.model_name, summary_payload), encoding="utf-8")

    print(f"probe_summary={summary_path}")


if __name__ == "__main__":
    main()
