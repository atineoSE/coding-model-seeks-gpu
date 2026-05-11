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

    new_models: set[str] | None = None
    """Model names appearing for the first time in this snapshot. None = not computed (show all)."""

    gained_categories: dict[str, list[str]] = field(default_factory=dict)
    """model_name → categories gained since the previous snapshot (existing models only)."""

    score_changes: dict[str, list[tuple[str, float, float]]] = field(default_factory=dict)
    """model_name → list of (benchmark_name, old_score, new_score) for benchmarks where the
    score changed but coverage did not (i.e. excludes new_models and gained_categories)."""


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


def _load_snapshot_state(
    snap_dir: Path,
) -> tuple[
    list[dict] | None,
    list[dict] | None,
    dict[str, set[str]],
    dict[tuple[str, str], float],
]:
    """Read a snapshot dir into (benchmarks, sota, coverage, scores).

    Returns Nones / empties for any file that's missing or unparsable so callers
    can treat that as "no baseline" and fall through to writing.
    """
    benchmarks_path = snap_dir / "benchmarks.json"
    sota_path = snap_dir / "sota_scores.json"
    benchmarks_data: list[dict] | None = None
    sota_data: list[dict] | None = None
    coverage: dict[str, set[str]] = {}
    scores: dict[tuple[str, str], float] = {}
    if benchmarks_path.exists():
        try:
            benchmarks_data = json.loads(benchmarks_path.read_text())
            for entry in benchmarks_data:
                if entry.get("benchmark_name") != OVERALL_NAME:
                    name = entry["model_name"]
                    bench = entry["benchmark_name"]
                    coverage.setdefault(name, set()).add(bench)
                    scores[(name, bench)] = entry["score"]
        except (json.JSONDecodeError, KeyError):
            benchmarks_data = None
            coverage = {}
            scores = {}
    if sota_path.exists():
        try:
            sota_data = json.loads(sota_path.read_text())
        except json.JSONDecodeError:
            sota_data = None
    return benchmarks_data, sota_data, coverage, scores


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

    # Walk existing + new dates in chronological order so each new date's
    # dedup/diff baseline is the snapshot that *chronologically* precedes it,
    # which may be an existing snapshot or a just-generated new one. Seeding
    # from max(existing_dates) is wrong when new dates land *between* existing
    # ones — e.g. after regenerating with a corrected timezone — because that
    # compares the new snapshot against one taken after it.
    existing_date_objs: set[date] = {date.fromisoformat(d) for d in existing_dates}
    walk_dates = sorted(existing_date_objs | set(new_dates))

    prev_benchmarks_data: list[dict] | None = None
    prev_sota_data: list[dict] | None = None
    prev_coverage: dict[str, set[str]] = {}
    prev_scores: dict[tuple[str, str], float] = {}
    pending_existing: date | None = None

    generated_dates: list[date] = []
    new_snapshot_infos: list[NewSnapshotInfo] = []
    for day in walk_dates:
        if day in existing_date_objs:
            pending_existing = day
            continue

        # New date — if an existing snapshot has been seen since we last
        # refreshed the baseline, load it now (it's the chronological prior).
        if pending_existing is not None:
            (
                prev_benchmarks_data,
                prev_sota_data,
                prev_coverage,
                prev_scores,
            ) = _load_snapshot_state(snapshots_dir / pending_existing.isoformat())
            pending_existing = None

        snapshot = generate_snapshot(repo_path, day)
        if snapshot is None:
            continue

        benchmarks_data = [e.to_dict() for e in snapshot.benchmarks]
        sota_data = [e.to_dict() for e in snapshot.sota_scores]

        # Dedup: skip when content is byte-identical to the previous snapshot.
        # An upstream commit-day with no leaderboard delta should not mint a new snapshot.
        if (
            prev_benchmarks_data is not None
            and prev_sota_data is not None
            and benchmarks_data == prev_benchmarks_data
            and sota_data == prev_sota_data
        ):
            logger.info("Snapshot %s identical to previous; skipping", day.isoformat())
            continue

        info = extract_coverage(snapshot)
        info.new_models = {m for m in info.model_coverage if m not in prev_coverage}

        curr_scores = {
            (e.model_name, e.benchmark_name): e.score
            for e in snapshot.benchmarks
            if e.benchmark_name != OVERALL_NAME
        }

        for model, cats in info.model_coverage.items():
            if model not in prev_coverage:
                continue
            gained = sorted(set(cats) - prev_coverage[model])
            if gained:
                info.gained_categories[model] = gained
            shared = sorted(set(cats) & prev_coverage[model])
            changes: list[tuple[str, float, float]] = []
            for bench in shared:
                old = prev_scores.get((model, bench))
                new = curr_scores[(model, bench)]
                if old is not None and old != new:
                    changes.append((bench, old, new))
            if changes:
                info.score_changes[model] = changes

        prev_benchmarks_data = benchmarks_data
        prev_sota_data = sota_data
        prev_coverage = {m: set(cats) for m, cats in info.model_coverage.items()}
        prev_scores = dict(curr_scores)

        write_snapshot(snapshots_dir, snapshot)
        generated_dates.append(day)
        new_snapshot_infos.append(info)

    # Combine existing + new for index
    all_snapshot_dates = sorted(
        {date.fromisoformat(d) for d in existing_dates} | set(generated_dates)
    )

    if all_snapshot_dates:
        write_index(snapshots_dir, all_snapshot_dates)

    logger.info("Generated %d new snapshot(s)", len(generated_dates))
    return new_snapshot_infos
