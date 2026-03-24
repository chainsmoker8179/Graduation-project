from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.lstm_attack_daily_panel import load_multiseed_daily_panel
from scripts.analysis.lstm_attack_significance import bootstrap_path_metric_deltas, build_significance_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build significance statistics for LSTM white-box attack outputs.")
    parser.add_argument("--report-root", type=Path, default=Path("reports"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/lstm_single_model_evidence"))
    parser.add_argument("--seed-root-path", type=Path, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    panel = load_multiseed_daily_panel(args.report_root, seed_root_path=args.seed_root_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    significance_df = build_significance_table(panel)
    significance_path = args.out_dir / "significance_daily_metrics.csv"
    significance_df.to_csv(significance_path, index=False)

    bootstrap_summary = {
        "partial_fgsm_vs_partial_clean": bootstrap_path_metric_deltas(
            panel,
            attacked_prefix="fgsm",
            n_bootstrap=args.bootstrap_reps,
            block_size=args.block_size,
            random_seed=args.random_seed,
        ),
        "partial_pgd_vs_partial_clean": bootstrap_path_metric_deltas(
            panel,
            attacked_prefix="pgd",
            n_bootstrap=args.bootstrap_reps,
            block_size=args.block_size,
            random_seed=args.random_seed,
        ),
    }
    bootstrap_path = args.out_dir / "significance_block_bootstrap.json"
    bootstrap_path.write_text(json.dumps(bootstrap_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"significance_csv={significance_path}")
    print(f"bootstrap_json={bootstrap_path}")


if __name__ == "__main__":
    main()
