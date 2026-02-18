"""Tests for snapshot generation logic."""

from datetime import date
from unittest.mock import patch

from pipeline.snapshots.generator import generate_snapshot
from pipeline.snapshots.reader import ModelData, ScoreEntry


def _make_model(name: str, scores: list[tuple[str, float, float | None]]) -> ModelData:
    """Helper to create a ModelData with score entries."""
    return ModelData(
        model_name=name,
        scores=[
            ScoreEntry(benchmark=bench, score=score, cost_per_instance=cost)
            for bench, score, cost in scores
        ],
    )


SAMPLE_MODELS = [
    _make_model(
        "claude-opus-4-5",
        [
            ("swe-bench", 76.6, 1.82),
            ("commit0", 37.5, 4.65),
            ("gaia", 69.1, 0.55),
            ("swt-bench", 78.5, 1.38),
            ("swe-bench-multimodal", 41.2, 2.54),
        ],
    ),
    _make_model(
        "GPT-5.2",
        [
            ("swe-bench", 72.0, 1.50),
            ("commit0", 30.0, 3.00),
        ],
    ),
]


@patch("pipeline.snapshots.generator.read_all_models")
@patch("pipeline.snapshots.generator.get_last_commit_of_day")
def test_generates_all_categories_plus_overall(mock_commit, mock_read):
    """Snapshot should have 5 category benchmarks + overall."""
    mock_commit.return_value = "abc123"
    mock_read.return_value = SAMPLE_MODELS

    from pathlib import Path

    snapshot = generate_snapshot(Path("/fake"), date(2026, 2, 15))
    assert snapshot is not None

    bench_names = {e.benchmark_name for e in snapshot.benchmarks}
    assert "overall" in bench_names
    assert "issue_resolution" in bench_names
    assert "frontend" in bench_names
    assert "greenfield" in bench_names
    assert "testing" in bench_names
    assert "information_gathering" in bench_names


@patch("pipeline.snapshots.generator.read_all_models")
@patch("pipeline.snapshots.generator.get_last_commit_of_day")
def test_ranks_are_assigned(mock_commit, mock_read):
    """Models should be ranked by score descending within each category."""
    mock_commit.return_value = "abc123"
    mock_read.return_value = SAMPLE_MODELS

    from pathlib import Path

    snapshot = generate_snapshot(Path("/fake"), date(2026, 2, 15))
    assert snapshot is not None

    # In issue_resolution (swe-bench), claude should be rank 1
    ir_entries = [e for e in snapshot.benchmarks if e.benchmark_name == "issue_resolution"]
    ir_entries.sort(key=lambda e: e.rank)
    assert ir_entries[0].model_name == "claude-opus-4-5"
    assert ir_entries[0].rank == 1
    assert ir_entries[1].model_name == "GPT-5.2"
    assert ir_entries[1].rank == 2


@patch("pipeline.snapshots.generator.read_all_models")
@patch("pipeline.snapshots.generator.get_last_commit_of_day")
def test_sota_extraction(mock_commit, mock_read):
    """SOTA should pick highest score per category."""
    mock_commit.return_value = "abc123"
    mock_read.return_value = SAMPLE_MODELS

    from pathlib import Path

    snapshot = generate_snapshot(Path("/fake"), date(2026, 2, 15))
    assert snapshot is not None

    sota_map = {s.benchmark_name: s for s in snapshot.sota_scores}
    assert sota_map["issue_resolution"].sota_model_name == "claude-opus-4-5"
    assert sota_map["issue_resolution"].sota_score == 76.6
    assert sota_map["greenfield"].sota_model_name == "claude-opus-4-5"
    assert sota_map["greenfield"].sota_score == 37.5


@patch("pipeline.snapshots.generator.read_all_models")
@patch("pipeline.snapshots.generator.get_last_commit_of_day")
def test_overall_is_mean_of_available(mock_commit, mock_read):
    """Overall only includes models with all categories; score is the mean."""
    mock_commit.return_value = "abc123"
    mock_read.return_value = SAMPLE_MODELS

    from pathlib import Path

    snapshot = generate_snapshot(Path("/fake"), date(2026, 2, 15))
    assert snapshot is not None

    overall_entries = {
        e.model_name: e for e in snapshot.benchmarks if e.benchmark_name == "overall"
    }

    # claude has 5 categories: mean(76.6, 37.5, 69.1, 78.5, 41.2) = 60.58 → 60.6
    claude_overall = overall_entries["claude-opus-4-5"]
    expected = round((76.6 + 37.5 + 69.1 + 78.5 + 41.2) / 5, 1)
    assert claude_overall.score == expected

    # GPT only has 2 of 5 categories → excluded from overall
    assert "GPT-5.2" not in overall_entries


@patch("pipeline.snapshots.generator.read_all_models")
@patch("pipeline.snapshots.generator.get_last_commit_of_day")
def test_no_commit_returns_none(mock_commit, mock_read):
    """If no commit exists for a date, returns None."""
    mock_commit.return_value = None

    from pathlib import Path

    snapshot = generate_snapshot(Path("/fake"), date(2026, 2, 15))
    assert snapshot is None


@patch("pipeline.snapshots.generator.read_all_models")
@patch("pipeline.snapshots.generator.get_last_commit_of_day")
def test_benchmark_map_filters_unknown(mock_commit, mock_read):
    """Unknown benchmark names from the repo are ignored."""
    mock_commit.return_value = "abc123"
    mock_read.return_value = [
        _make_model("test-model", [("unknown-bench", 50.0, 1.0)]),
    ]

    from pathlib import Path

    snapshot = generate_snapshot(Path("/fake"), date(2026, 2, 15))
    # Should still produce an overall from whatever categories matched (none)
    # → no entries
    assert snapshot is not None
    assert len(snapshot.benchmarks) == 0
