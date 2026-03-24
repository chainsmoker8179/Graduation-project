#!/usr/bin/env python3

from __future__ import annotations

from scripts.export_large_sample_attack_assets import main
from scripts.export_lstm_attack_assets import build_matched_reference

__all__ = ["build_matched_reference", "main"]


if __name__ == "__main__":
    main()
