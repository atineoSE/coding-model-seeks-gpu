"""Tests for check_missing_mappings() in main.py."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.main import check_missing_mappings


def _write_snapshot(tmp_path: Path, benchmarks: list[dict]) -> Path:
    """Write a minimal snapshot index + benchmarks.json under tmp_path."""
    snapshot_dir = tmp_path / "2026-01-01"
    snapshot_dir.mkdir()
    (snapshot_dir / "benchmarks.json").write_text(json.dumps(benchmarks))
    (tmp_path / "index.json").write_text(json.dumps({"latest": "2026-01-01", "snapshots": ["2026-01-01"]}))
    return tmp_path


class TestCheckMissingMappings:
    def test_model_in_MODEL_NAME_TO_HF_ID_is_not_flagged(self, tmp_path):
        benchmarks = [{"model_name": "KnownModel", "openness": "open_weights"}]
        snapshots_dir = _write_snapshot(tmp_path, benchmarks)

        with patch("pipeline.main.MODEL_NAME_TO_HF_ID", {"KnownModel": "org/repo"}), \
             patch("pipeline.main.MODEL_ARCH_SOURCE_HF_ID", {}), \
             patch("pipeline.main.notify_missing_mapping") as mock_notify, \
             patch("pipeline.main.is_enabled", return_value=True):
            check_missing_mappings(snapshots_dir)

        mock_notify.assert_not_called()

    def test_model_in_MODEL_ARCH_SOURCE_HF_ID_is_not_flagged(self, tmp_path):
        """Models handled via MODEL_ARCH_SOURCE_HF_ID must not trigger the alert."""
        benchmarks = [{"model_name": "MiniMax-M2.7", "openness": "open_weights"}]
        snapshots_dir = _write_snapshot(tmp_path, benchmarks)

        with patch("pipeline.main.MODEL_NAME_TO_HF_ID", {}), \
             patch("pipeline.main.MODEL_ARCH_SOURCE_HF_ID", {"MiniMax-M2.7": "MiniMaxAI/MiniMax-M2.5"}), \
             patch("pipeline.main.notify_missing_mapping") as mock_notify, \
             patch("pipeline.main.is_enabled", return_value=True):
            check_missing_mappings(snapshots_dir)

        mock_notify.assert_not_called()

    def test_truly_unknown_model_is_flagged(self, tmp_path):
        benchmarks = [{"model_name": "UnknownModel-7B", "openness": "open_weights"}]
        snapshots_dir = _write_snapshot(tmp_path, benchmarks)

        with patch("pipeline.main.MODEL_NAME_TO_HF_ID", {}), \
             patch("pipeline.main.MODEL_ARCH_SOURCE_HF_ID", {}), \
             patch("pipeline.main.notify_missing_mapping") as mock_notify, \
             patch("pipeline.main.is_enabled", return_value=True):
            check_missing_mappings(snapshots_dir)

        mock_notify.assert_called_once_with("UnknownModel-7B")

    def test_closed_source_model_is_never_flagged(self, tmp_path):
        benchmarks = [{"model_name": "GPT-5", "openness": "closed"}]
        snapshots_dir = _write_snapshot(tmp_path, benchmarks)

        with patch("pipeline.main.MODEL_NAME_TO_HF_ID", {}), \
             patch("pipeline.main.MODEL_ARCH_SOURCE_HF_ID", {}), \
             patch("pipeline.main.notify_missing_mapping") as mock_notify, \
             patch("pipeline.main.is_enabled", return_value=True):
            check_missing_mappings(snapshots_dir)

        mock_notify.assert_not_called()

    def test_notify_not_called_when_disabled(self, tmp_path):
        benchmarks = [{"model_name": "UnknownModel-7B", "openness": "open_weights"}]
        snapshots_dir = _write_snapshot(tmp_path, benchmarks)

        with patch("pipeline.main.MODEL_NAME_TO_HF_ID", {}), \
             patch("pipeline.main.MODEL_ARCH_SOURCE_HF_ID", {}), \
             patch("pipeline.main.notify_missing_mapping") as mock_notify, \
             patch("pipeline.main.is_enabled", return_value=False):
            check_missing_mappings(snapshots_dir)

        mock_notify.assert_not_called()
