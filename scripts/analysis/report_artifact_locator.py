from __future__ import annotations

import csv
from pathlib import Path


def resolve_report_artifact(repo_root: str | Path, relative_path: str | Path) -> Path:
    repo_root = Path(repo_root).resolve()
    relative_path = Path(relative_path)
    direct_path = repo_root / relative_path
    if direct_path.exists():
        return direct_path

    manifest_path = repo_root / "reports" / "archive_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"artifact not found and archive manifest missing: {relative_path}")

    directory_candidates: list[Path] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            entry_type = row[2]
            archived_rel_path = row[4]
            archived_abs_path = row[5]
            if archived_rel_path == str(relative_path):
                archived_path = Path(archived_abs_path)
                if archived_path.exists():
                    return archived_path
                raise FileNotFoundError(f"archived artifact missing on disk: {archived_path}")
            if entry_type == "dir" and str(relative_path.parent).rstrip("/") + "/" == archived_rel_path:
                directory_candidates.append(Path(archived_abs_path))
    for directory in directory_candidates:
        candidate = directory / relative_path.name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"artifact not found in reports or archive manifest: {relative_path}")
