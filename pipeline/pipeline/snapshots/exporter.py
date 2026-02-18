"""Write snapshot data to disk as JSON files."""

import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path

from pipeline.snapshots.generator import Snapshot
from pipeline.snapshots.git_repo import ensure_submodule, get_dates_with_commits

logger = logging.getLogger(__name__)


def _write_json(path: Path, data: object) -> None:
    """Write data as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_index(snapshots_dir: Path) -> dict | None:
    """Load snapshots/index.json. Returns None if missing or corrupt."""
    index_path = snapshots_dir / "index.json"
    if not index_path.exists():
        return None
    try:
        data = json.loads(index_path.read_text())
        if "snapshots" not in data or "latest" not in data:
            return None
        return data
    except (json.JSONDecodeError, KeyError):
        return None


def write_snapshot(snapshots_dir: Path, snapshot: Snapshot) -> None:
    """Write a single snapshot's files to disk."""
    date_str = snapshot.snapshot_date.isoformat()
    snap_dir = snapshots_dir / date_str

    benchmarks_data = [e.to_dict() for e in snapshot.benchmarks]
    sota_data = [e.to_dict() for e in snapshot.sota_scores]

    _write_json(snap_dir / "benchmarks.json", benchmarks_data)
    _write_json(snap_dir / "sota_scores.json", sota_data)

    logger.info("Wrote snapshot %s (%d entries)", date_str, len(benchmarks_data))


def write_index(snapshots_dir: Path, snapshot_dates: list[date]) -> None:
    """Write or update the index.json file."""
    sorted_dates = sorted(snapshot_dates)
    index = {
        "snapshots": [d.isoformat() for d in sorted_dates],
        "latest": sorted_dates[-1].isoformat() if sorted_dates else None,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(snapshots_dir / "index.json", index)
    logger.info(
        "Wrote index.json: %d snapshots, latest=%s",
        len(sorted_dates),
        index["latest"],
    )


def run_snapshot_export(
    repo_path: Path,
    snapshots_dir: Path,
    *,
    force: bool = False,
) -> int:
    """Run the full snapshot export pipeline.

    Returns the number of new snapshots generated.
    """
    from pipeline.snapshots.generator import generate_snapshot

    ensure_submodule(repo_path)

    # Get all dates with commits
    all_dates = get_dates_with_commits(repo_path)
    if not all_dates:
        logger.warning("No commit dates found in repo")
        return 0

    # Load existing index (unless forcing full regen)
    existing_dates: set[str] = set()
    if not force:
        index = load_index(snapshots_dir)
        if index is not None:
            # Verify all listed snapshots still have directories on disk
            missing = [
                d
                for d in index["snapshots"]
                if not (snapshots_dir / d / "benchmarks.json").exists()
            ]
            if missing:
                logger.warning(
                    "Missing snapshot dirs for %d dates, forcing full regen",
                    len(missing),
                )
                existing_dates = set()
            else:
                existing_dates = set(index["snapshots"])

    # Determine which dates need generation
    new_dates = [d for d in all_dates if d.isoformat() not in existing_dates]

    if not new_dates and existing_dates:
        logger.info("All %d snapshots up to date, nothing to generate", len(existing_dates))
        return 0

    logger.info(
        "%d new snapshot(s) to generate (of %d total dates)",
        len(new_dates),
        len(all_dates),
    )

    # Generate new snapshots
    generated_dates: list[date] = []
    for day in new_dates:
        snapshot = generate_snapshot(repo_path, day)
        if snapshot is not None:
            write_snapshot(snapshots_dir, snapshot)
            generated_dates.append(day)

    # Combine existing + new for index
    all_snapshot_dates = sorted(
        {date.fromisoformat(d) for d in existing_dates} | set(generated_dates)
    )

    if all_snapshot_dates:
        write_index(snapshots_dir, all_snapshot_dates)

    logger.info("Generated %d new snapshot(s)", len(generated_dates))
    return len(generated_dates)
