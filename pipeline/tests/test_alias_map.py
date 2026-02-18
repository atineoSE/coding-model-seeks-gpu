"""Tests for the model alias/rename resolution."""

from datetime import date

from pipeline.snapshots.alias_map import resolve_model_name


def test_no_rename_needed():
    """Models without renames return unchanged."""
    assert resolve_model_name("claude-opus-4-5", date(2026, 2, 15)) == "claude-opus-4-5"


def test_rename_before_effective_date():
    """Before a rename's effective date, the old name stays."""
    assert resolve_model_name("claude-4.5-opus", date(2026, 2, 9)) == "claude-4.5-opus"


def test_rename_on_effective_date():
    """On the rename date, the new name is used."""
    assert resolve_model_name("claude-4.5-opus", date(2026, 2, 10)) == "claude-opus-4-5"


def test_rename_after_effective_date():
    """After the rename date, the new name is used."""
    assert resolve_model_name("claude-4.5-opus", date(2026, 2, 15)) == "claude-opus-4-5"


def test_chained_rename_jade_spark():
    """jade-spark-2862 → Minimax-2.5 → MiniMax-M2.5 when both renames have happened."""
    # Before any rename
    assert resolve_model_name("jade-spark-2862", date(2026, 2, 10)) == "jade-spark-2862"
    # After first rename only
    assert resolve_model_name("jade-spark-2862", date(2026, 2, 11)) == "Minimax-2.5"
    # After both renames
    assert resolve_model_name("jade-spark-2862", date(2026, 2, 12)) == "MiniMax-M2.5"


def test_unknown_model_unchanged():
    """A model name not in the rename table passes through."""
    assert resolve_model_name("some-unknown-model", date(2026, 12, 31)) == "some-unknown-model"


def test_early_model_name_format():
    """Early model names (pre-rename) resolve correctly at old dates."""
    assert (
        resolve_model_name("claude-opus-4-5-20251101", date(2025, 12, 22))
        == "claude-opus-4-5-20251101"
    )
    assert resolve_model_name("claude-opus-4-5-20251101", date(2026, 2, 10)) == "claude-opus-4-5"
