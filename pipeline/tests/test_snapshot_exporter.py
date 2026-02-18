"""Tests for snapshot export (file writing and index management)."""

import json
from datetime import date

from pipeline.snapshots.exporter import load_index, write_index, write_snapshot
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
