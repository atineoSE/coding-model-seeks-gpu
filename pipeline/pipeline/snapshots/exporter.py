"""Write snapshot data to disk as JSON files."""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from pipeline.snapshots.constants import BENCHMARK_MAP, OVERALL_NAME
from pipeline.snapshots.generator import Snapshot
from pipeline.snapshots.git_repo import ensure_submodule, get_dates_with_commits

logger = logging.getLogger(__name__)

# All benchmark categories (excluding the derived "overall")
ALL_CATEGORIES: set[str] = {name for name, _display in BENCHMARK_MAP.values()}


@dataclass
class NewSnapshotInfo:
    """Coverage summary for a single newly generated snapshot."""

    snapshot_date: date
    model_coverage: dict[str, list[str]] = field(default_factory=dict)
    """model_name → list of covered category benchmark_names."""

    model_missing: dict[str, list[str]] = field(default_factory=dict)
    """model_name → list of missing category benchmark_names."""


def extract_coverage(snapshot: Snapshot) -> NewSnapshotInfo:
    """Extract per-model category coverage from a snapshot."""
    covered: dict[str, set[str]] = defaultdict(set)

    for entry in snapshot.benchmarks:
        if entry.benchmark_name == OVERALL_NAME:
            continue
        covered[entry.model_name].add(entry.benchmark_name)

    info = NewSnapshotInfo(snapshot_date=snapshot.snapshot_date)
    for model in sorted(covered):
        cats = covered[model]
        info.model_coverage[model] = sorted(cats)
        missing = ALL_CATEGORIES - cats
        if missing:
            info.model_missing[model] = sorted(missing)

    return info


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
) -> list[NewSnapshotInfo]:
    """Run the full snapshot export pipeline.

    Returns a list of NewSnapshotInfo for each newly generated snapshot,
    containing per-model category coverage details.
    """
    from pipeline.snapshots.generator import generate_snapshot

    ensure_submodule(repo_path)

    # Get all dates with commits
    all_dates = get_dates_with_commits(repo_path)
    if not all_dates:
        logger.warning("No commit dates found in repo")
        return []

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
        return []

    logger.info(
        "%d new snapshot(s) to generate (of %d total dates)",
        len(new_dates),
        len(all_dates),
    )

    # Generate new snapshots
    generated_dates: list[date] = []
    new_snapshot_infos: list[NewSnapshotInfo] = []
    for day in new_dates:
        snapshot = generate_snapshot(repo_path, day)
        if snapshot is not None:
            write_snapshot(snapshots_dir, snapshot)
            generated_dates.append(day)
            new_snapshot_infos.append(extract_coverage(snapshot))

    # Combine existing + new for index
    all_snapshot_dates = sorted(
        {date.fromisoformat(d) for d in existing_dates} | set(generated_dates)
    )

    if all_snapshot_dates:
        write_index(snapshots_dir, all_snapshot_dates)

    logger.info("Generated %d new snapshot(s)", len(generated_dates))
    return new_snapshot_infos
