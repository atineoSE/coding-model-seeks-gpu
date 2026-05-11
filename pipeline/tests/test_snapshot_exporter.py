"""Tests for snapshot export (file writing and index management)."""

import json
from datetime import date
from unittest.mock import patch

from pipeline.snapshots.exporter import (
    ALL_CATEGORIES,
    NewSnapshotInfo,
    extract_coverage,
    load_index,
    run_snapshot_export,
    write_index,
    write_snapshot,
)
from pipeline.snapshots.generator import BenchmarkEntry, Snapshot, SotaEntry


def test_write_snapshot_creates_files(tmp_path):
    """write_snapshot should create benchmarks.json and sota_scores.json."""
    snapshot = Snapshot(
        snapshot_date=date(2026, 1, 15),
        benchmarks=[
            BenchmarkEntry(
                model_name="test-model",
                benchmark_name="issue_resolution",
                benchmark_display_name="Issue Resolution",
                score=75.0,
                rank=1,
                cost_per_task=1.5,
            ),
        ],
        sota_scores=[
            SotaEntry(
                benchmark_name="issue_resolution",
                benchmark_display_name="Issue Resolution",
                sota_model_name="test-model",
                sota_score=75.0,
            ),
        ],
    )

    write_snapshot(tmp_path, snapshot)

    bench_path = tmp_path / "2026-01-15" / "benchmarks.json"
    sota_path = tmp_path / "2026-01-15" / "sota_scores.json"

    assert bench_path.exists()
    assert sota_path.exists()

    benchmarks = json.loads(bench_path.read_text())
    assert len(benchmarks) == 1
    assert benchmarks[0]["model_name"] == "test-model"
    assert benchmarks[0]["score"] == 75.0
    assert benchmarks[0]["rank"] == 1

    sota = json.loads(sota_path.read_text())
    assert len(sota) == 1
    assert sota[0]["sota_model_name"] == "test-model"


def test_write_index_format(tmp_path):
    """write_index should create a properly formatted index.json."""
    dates = [date(2026, 1, 15), date(2025, 12, 19), date(2026, 1, 20)]
    write_index(tmp_path, dates)

    index_path = tmp_path / "index.json"
    assert index_path.exists()

    data = json.loads(index_path.read_text())
    assert data["snapshots"] == ["2025-12-19", "2026-01-15", "2026-01-20"]
    assert data["latest"] == "2026-01-20"
    assert "generated_at" in data


def test_load_index_missing(tmp_path):
    """load_index returns None when index.json doesn't exist."""
    assert load_index(tmp_path) is None


def test_load_index_corrupt(tmp_path):
    """load_index returns None for corrupt JSON."""
    (tmp_path / "index.json").write_text("not json")
    assert load_index(tmp_path) is None


def test_load_index_valid(tmp_path):
    """load_index returns the parsed data for valid index.json."""
    data = {
        "snapshots": ["2026-01-15"],
        "latest": "2026-01-15",
        "generated_at": "2026-01-15T00:00:00Z",
    }
    (tmp_path / "index.json").write_text(json.dumps(data))
    result = load_index(tmp_path)
    assert result is not None
    assert result["latest"] == "2026-01-15"


def _make_entry(model: str, bench: str, display: str = "", score: float = 50.0) -> BenchmarkEntry:
    """Helper to create a BenchmarkEntry for testing."""
    return BenchmarkEntry(
        model_name=model,
        benchmark_name=bench,
        benchmark_display_name=display or bench,
        score=score,
        rank=1,
        cost_per_task=None,
    )


def test_extract_coverage_complete_model():
    """A model with all 5 categories should have no missing entries."""
    snapshot = Snapshot(
        snapshot_date=date(2026, 4, 20),
        benchmarks=[
            _make_entry("model-a", "frontend"),
            _make_entry("model-a", "greenfield"),
            _make_entry("model-a", "issue_resolution"),
            _make_entry("model-a", "testing"),
            _make_entry("model-a", "information_gathering"),
            _make_entry("model-a", "overall"),  # should be ignored
        ],
    )

    info = extract_coverage(snapshot)

    assert info.snapshot_date == date(2026, 4, 20)
    assert set(info.model_coverage["model-a"]) == ALL_CATEGORIES
    assert "model-a" not in info.model_missing


def test_extract_coverage_partial_model():
    """A model missing some categories should appear in model_missing."""
    snapshot = Snapshot(
        snapshot_date=date(2026, 4, 20),
        benchmarks=[
            _make_entry("model-b", "frontend"),
            _make_entry("model-b", "greenfield"),
        ],
    )

    info = extract_coverage(snapshot)

    assert info.model_coverage["model-b"] == ["frontend", "greenfield"]
    assert set(info.model_missing["model-b"]) == {
        "information_gathering",
        "issue_resolution",
        "testing",
    }


def test_extract_coverage_multiple_models():
    """Coverage is tracked independently per model."""
    snapshot = Snapshot(
        snapshot_date=date(2026, 4, 20),
        benchmarks=[
            _make_entry("model-a", "frontend"),
            _make_entry("model-a", "greenfield"),
            _make_entry("model-a", "issue_resolution"),
            _make_entry("model-a", "testing"),
            _make_entry("model-a", "information_gathering"),
            _make_entry("model-b", "frontend"),
        ],
    )

    info = extract_coverage(snapshot)

    assert "model-a" not in info.model_missing
    assert set(info.model_missing["model-b"]) == ALL_CATEGORIES - {"frontend"}


def test_extract_coverage_empty_snapshot():
    """An empty snapshot should produce empty coverage."""
    snapshot = Snapshot(snapshot_date=date(2026, 4, 20), benchmarks=[])

    info = extract_coverage(snapshot)

    assert info.model_coverage == {}
    assert info.model_missing == {}


# ---------------------------------------------------------------------------
# run_snapshot_export — new_models and gained_categories diff logic
# ---------------------------------------------------------------------------

def _make_snapshot(
    d: date,
    model_cats: dict[str, list[str]],
    scores: dict[tuple[str, str], float] | None = None,
) -> Snapshot:
    """Build a minimal Snapshot with the given model→categories mapping.

    Optional `scores` overrides the default 50.0 for specific (model, benchmark) pairs.
    """
    scores = scores or {}
    entries = [
        _make_entry(model, bench, score=scores.get((model, bench), 50.0))
        for model, cats in model_cats.items()
        for bench in cats
    ]
    return Snapshot(snapshot_date=d, benchmarks=entries)


def _write_baseline(
    snapshots_dir,
    snap_date: date,
    model_cats: dict[str, list[str]],
    scores: dict[tuple[str, str], float] | None = None,
) -> None:
    """Write a fake existing snapshot and index so run_snapshot_export treats it as known."""
    scores = scores or {}
    date_str = snap_date.isoformat()
    snap_dir = snapshots_dir / date_str
    snap_dir.mkdir(parents=True)
    rows = [
        {"model_name": m, "benchmark_name": b, "score": scores.get((m, b), 50.0)}
        for m, cats in model_cats.items()
        for b in cats
    ]
    (snap_dir / "benchmarks.json").write_text(json.dumps(rows))
    (snap_dir / "sota_scores.json").write_text("[]")
    index = {"snapshots": [date_str], "latest": date_str, "generated_at": "2026-01-01T00:00:00Z"}
    (snapshots_dir / "index.json").write_text(json.dumps(index))


def test_new_model_detected(tmp_path):
    """A model absent from the previous snapshot appears in new_models."""
    _write_baseline(tmp_path, date(2026, 4, 27), {"existing-model": ["frontend"]})
    new_snap = _make_snapshot(date(2026, 4, 28), {
        "existing-model": ["frontend"],
        "brand-new-model": ["frontend", "testing"],
    })

    with (
        patch("pipeline.snapshots.exporter.ensure_submodule"),
        patch("pipeline.snapshots.exporter.get_dates_with_commits", return_value=[date(2026, 4, 28)]),
        patch("pipeline.snapshots.generator.generate_snapshot", return_value=new_snap),
    ):
        infos = run_snapshot_export(tmp_path, tmp_path)

    assert len(infos) == 1
    assert infos[0].new_models == {"brand-new-model"}
    assert "existing-model" not in infos[0].new_models


def test_gained_categories_detected(tmp_path):
    """An existing model that gains a new category appears in gained_categories."""
    _write_baseline(tmp_path, date(2026, 4, 27), {"model-a": ["frontend"]})
    new_snap = _make_snapshot(date(2026, 4, 28), {"model-a": ["frontend", "testing"]})

    with (
        patch("pipeline.snapshots.exporter.ensure_submodule"),
        patch("pipeline.snapshots.exporter.get_dates_with_commits", return_value=[date(2026, 4, 28)]),
        patch("pipeline.snapshots.generator.generate_snapshot", return_value=new_snap),
    ):
        infos = run_snapshot_export(tmp_path, tmp_path)

    assert infos[0].gained_categories == {"model-a": ["testing"]}


def test_unchanged_model_has_no_gained_categories(tmp_path):
    """A model with identical coverage produces no gained_categories entry."""
    _write_baseline(tmp_path, date(2026, 4, 27), {"model-a": ["frontend", "testing"]})
    new_snap = _make_snapshot(date(2026, 4, 28), {"model-a": ["frontend", "testing"]})

    with (
        patch("pipeline.snapshots.exporter.ensure_submodule"),
        patch("pipeline.snapshots.exporter.get_dates_with_commits", return_value=[date(2026, 4, 28)]),
        patch("pipeline.snapshots.generator.generate_snapshot", return_value=new_snap),
    ):
        infos = run_snapshot_export(tmp_path, tmp_path)

    assert infos[0].new_models == set()
    assert infos[0].gained_categories == {}


def test_score_change_detected(tmp_path):
    """A model whose score changes on an existing benchmark appears in score_changes."""
    _write_baseline(
        tmp_path,
        date(2026, 4, 27),
        {"model-a": ["frontend", "testing"]},
        scores={("model-a", "frontend"): 40.0, ("model-a", "testing"): 60.0},
    )
    new_snap = _make_snapshot(
        date(2026, 4, 28),
        {"model-a": ["frontend", "testing"]},
        scores={("model-a", "frontend"): 45.0, ("model-a", "testing"): 60.0},
    )

    dates = [date(2026, 4, 28)]
    with (
        patch("pipeline.snapshots.exporter.ensure_submodule"),
        patch("pipeline.snapshots.exporter.get_dates_with_commits", return_value=dates),
        patch("pipeline.snapshots.generator.generate_snapshot", return_value=new_snap),
    ):
        infos = run_snapshot_export(tmp_path, tmp_path)

    assert infos[0].score_changes == {"model-a": [("frontend", 40.0, 45.0)]}
    assert infos[0].new_models == set()
    assert infos[0].gained_categories == {}


def test_score_change_excludes_new_models(tmp_path):
    """A new model's scores belong under new_models, not score_changes."""
    _write_baseline(tmp_path, date(2026, 4, 27), {"old-model": ["frontend"]})
    new_snap = _make_snapshot(
        date(2026, 4, 28),
        {"old-model": ["frontend"], "new-model": ["frontend"]},
    )

    dates = [date(2026, 4, 28)]
    with (
        patch("pipeline.snapshots.exporter.ensure_submodule"),
        patch("pipeline.snapshots.exporter.get_dates_with_commits", return_value=dates),
        patch("pipeline.snapshots.generator.generate_snapshot", return_value=new_snap),
    ):
        infos = run_snapshot_export(tmp_path, tmp_path)

    assert infos[0].new_models == {"new-model"}
    assert "new-model" not in infos[0].score_changes


def test_score_change_excludes_gained_categories(tmp_path):
    """A newly gained category goes in gained_categories, not score_changes."""
    _write_baseline(
        tmp_path,
        date(2026, 4, 27),
        {"model-a": ["frontend"]},
        scores={("model-a", "frontend"): 40.0},
    )
    new_snap = _make_snapshot(
        date(2026, 4, 28),
        {"model-a": ["frontend", "testing"]},
        scores={("model-a", "frontend"): 40.0, ("model-a", "testing"): 50.0},
    )

    dates = [date(2026, 4, 28)]
    with (
        patch("pipeline.snapshots.exporter.ensure_submodule"),
        patch("pipeline.snapshots.exporter.get_dates_with_commits", return_value=dates),
        patch("pipeline.snapshots.generator.generate_snapshot", return_value=new_snap),
    ):
        infos = run_snapshot_export(tmp_path, tmp_path)

    assert infos[0].gained_categories == {"model-a": ["testing"]}
    assert infos[0].score_changes == {}


def test_identical_snapshot_skipped(tmp_path):
    """A snapshot byte-identical to the previous is not written or returned."""
    # Use write_snapshot to seed a real prior so to_dict()-shaped comparison can match.
    prior = Snapshot(
        snapshot_date=date(2026, 4, 27),
        benchmarks=[
            BenchmarkEntry(
                model_name="model-a",
                benchmark_name="frontend",
                benchmark_display_name="Frontend",
                score=42.0,
                rank=1,
                cost_per_task=None,
            ),
        ],
        sota_scores=[
            SotaEntry(
                benchmark_name="frontend",
                benchmark_display_name="Frontend",
                sota_model_name="model-a",
                sota_score=42.0,
            ),
        ],
    )
    write_snapshot(tmp_path, prior)
    write_index(tmp_path, [date(2026, 4, 27)])

    # New snapshot for a later date with identical content (only the snapshot_date differs)
    duplicate = Snapshot(
        snapshot_date=date(2026, 4, 28),
        benchmarks=prior.benchmarks,
        sota_scores=prior.sota_scores,
    )

    with (
        patch("pipeline.snapshots.exporter.ensure_submodule"),
        patch("pipeline.snapshots.exporter.get_dates_with_commits",
              return_value=[date(2026, 4, 27), date(2026, 4, 28)]),
        patch("pipeline.snapshots.generator.generate_snapshot", return_value=duplicate),
    ):
        infos = run_snapshot_export(tmp_path, tmp_path)

    assert infos == []
    assert not (tmp_path / "2026-04-28").exists()
    # Index unchanged: still only the original date
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["snapshots"] == ["2026-04-27"]


def test_score_change_mints_snapshot(tmp_path):
    """A snapshot with only score changes (no roster change) is still written."""
    prior = Snapshot(
        snapshot_date=date(2026, 4, 27),
        benchmarks=[
            BenchmarkEntry(
                model_name="model-a",
                benchmark_name="frontend",
                benchmark_display_name="Frontend",
                score=42.0,
                rank=1,
                cost_per_task=None,
            ),
        ],
        sota_scores=[],
    )
    write_snapshot(tmp_path, prior)
    write_index(tmp_path, [date(2026, 4, 27)])

    bumped = Snapshot(
        snapshot_date=date(2026, 4, 28),
        benchmarks=[
            BenchmarkEntry(
                model_name="model-a",
                benchmark_name="frontend",
                benchmark_display_name="Frontend",
                score=48.0,
                rank=1,
                cost_per_task=None,
            ),
        ],
        sota_scores=[],
    )

    with (
        patch("pipeline.snapshots.exporter.ensure_submodule"),
        patch("pipeline.snapshots.exporter.get_dates_with_commits",
              return_value=[date(2026, 4, 27), date(2026, 4, 28)]),
        patch("pipeline.snapshots.generator.generate_snapshot", return_value=bumped),
    ):
        infos = run_snapshot_export(tmp_path, tmp_path)

    assert len(infos) == 1
    assert infos[0].score_changes == {"model-a": [("frontend", 42.0, 48.0)]}
    assert (tmp_path / "2026-04-28" / "benchmarks.json").exists()


def _write_multi_baseline(
    snapshots_dir,
    snapshots: dict[date, dict[str, list[str]]],
    scores: dict[date, dict[tuple[str, str], float]] | None = None,
) -> None:
    """Write several existing snapshots (via the real writer so bytes match
    what generate_snapshot would produce) + an index referencing them all."""
    scores = scores or {}
    for snap_date, model_cats in snapshots.items():
        snap = _make_snapshot(snap_date, model_cats, scores=scores.get(snap_date))
        write_snapshot(snapshots_dir, snap)
    write_index(snapshots_dir, list(snapshots.keys()))


def test_backfill_identical_to_chronological_prior_is_skipped(tmp_path):
    """A new date that lands *between* two existing snapshots must dedup against
    the chronologically prior one (not the latest). Regression for the case
    where regenerating with a corrected timezone produced an index with gaps,
    and a subsequent incremental run wrote duplicate snapshots for those gap
    dates because the dedup compared them against the later latest snapshot."""
    _write_multi_baseline(
        tmp_path,
        {
            date(2026, 5, 1): {"model-a": ["frontend"]},
            date(2026, 5, 7): {"model-a": ["frontend"], "model-b": ["frontend"]},
        },
    )
    # Identical to 05-01; chronologically prior to 05-07
    backfill = _make_snapshot(date(2026, 5, 2), {"model-a": ["frontend"]})

    with (
        patch("pipeline.snapshots.exporter.ensure_submodule"),
        patch(
            "pipeline.snapshots.exporter.get_dates_with_commits",
            return_value=[date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 7)],
        ),
        patch("pipeline.snapshots.generator.generate_snapshot", return_value=backfill),
    ):
        infos = run_snapshot_export(tmp_path, tmp_path)

    assert infos == []
    assert not (tmp_path / "2026-05-02").exists()
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["snapshots"] == ["2026-05-01", "2026-05-07"]


def test_backfill_diff_against_chronological_prior(tmp_path):
    """A backfill snapshot's diff (new_models, score_changes) is computed
    against the chronological prior existing snapshot, not the latest."""
    _write_multi_baseline(
        tmp_path,
        {
            date(2026, 5, 1): {"model-a": ["frontend"]},
            date(2026, 5, 7): {
                "model-a": ["frontend"],
                "future-model": ["frontend"],
            },
        },
        scores={
            date(2026, 5, 1): {("model-a", "frontend"): 40.0},
            date(2026, 5, 7): {("model-a", "frontend"): 60.0, ("future-model", "frontend"): 70.0},
        },
    )
    # On 05-02: model-a score bumped to 45 (no new models yet relative to 05-01)
    backfill = _make_snapshot(
        date(2026, 5, 2),
        {"model-a": ["frontend"]},
        scores={("model-a", "frontend"): 45.0},
    )

    with (
        patch("pipeline.snapshots.exporter.ensure_submodule"),
        patch(
            "pipeline.snapshots.exporter.get_dates_with_commits",
            return_value=[date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 7)],
        ),
        patch("pipeline.snapshots.generator.generate_snapshot", return_value=backfill),
    ):
        infos = run_snapshot_export(tmp_path, tmp_path)

    assert len(infos) == 1
    # Compared against 05-01 (prior), not 05-07 (latest)
    assert infos[0].new_models == set()
    assert infos[0].score_changes == {"model-a": [("frontend", 40.0, 45.0)]}


def test_prev_coverage_advances_between_snapshots(tmp_path):
    """Models added in snapshot N are treated as existing in snapshot N+1."""
    _write_baseline(tmp_path, date(2026, 4, 26), {"old-model": ["frontend"]})

    snap_apr27 = _make_snapshot(date(2026, 4, 27), {
        "old-model": ["frontend"],
        "mid-model": ["frontend"],
    })
    snap_apr28 = _make_snapshot(date(2026, 4, 28), {
        "old-model": ["frontend"],
        "mid-model": ["frontend"],
        "newest-model": ["frontend"],
    })

    def fake_generate(repo_path, day):
        return snap_apr27 if day == date(2026, 4, 27) else snap_apr28

    with (
        patch("pipeline.snapshots.exporter.ensure_submodule"),
        patch("pipeline.snapshots.exporter.get_dates_with_commits",
              return_value=[date(2026, 4, 27), date(2026, 4, 28)]),
        patch("pipeline.snapshots.generator.generate_snapshot", side_effect=fake_generate),
    ):
        infos = run_snapshot_export(tmp_path, tmp_path)

    assert len(infos) == 2
    assert infos[0].new_models == {"mid-model"}
    # mid-model was added in Apr 27, so it must NOT be new in Apr 28
    assert infos[1].new_models == {"newest-model"}
    assert "mid-model" not in infos[1].new_models
