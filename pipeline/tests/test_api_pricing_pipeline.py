"""Integration tests for run_api_pricing_pipeline in main.py."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.main import run_api_pricing_pipeline


def _write_snapshot(tmp_path: Path, benchmarks: list[dict]) -> Path:
    snapshots_dir = tmp_path / "snapshots"
    latest_dir = snapshots_dir / "2026-04-17"
    latest_dir.mkdir(parents=True)
    index = {"snapshots": ["2026-04-17"], "latest": "2026-04-17"}
    (snapshots_dir / "index.json").write_text(json.dumps(index))
    (latest_dir / "benchmarks.json").write_text(json.dumps(benchmarks))
    return snapshots_dir


SAMPLE_BENCHMARKS = [
    {"model_name": "claude-opus-4-6", "benchmark_name": "overall", "score": 66.7, "openness": "closed_api_available"},
    {"model_name": "GPT-5.4", "benchmark_name": "overall", "score": 63.8, "openness": "closed_api_available"},
    {"model_name": "Gemini-3.1-Pro", "benchmark_name": "overall", "score": 55.7, "openness": "closed_api_available"},
]

FULL_PRICING = {
    "claude-opus-4-6": {"lab": "anthropic", "model_name": "claude-opus-4-6", "litellm_id": "claude-opus-4-6", "context_window": 1000000},
    "GPT-5.4": {"lab": "openai", "model_name": "GPT-5.4", "litellm_id": "gpt-5.4", "context_window": 1050000},
    "Gemini-3.1-Pro": {"lab": "google", "model_name": "Gemini-3.1-Pro", "litellm_id": "gemini/gemini-3.1-pro-preview", "context_window": 1048576},
}


class TestRunApiPricingPipeline:
    def test_succeeds_when_all_required_labs_present(self, tmp_path):
        snapshots_dir = _write_snapshot(tmp_path, SAMPLE_BENCHMARKS)

        with (
            patch("pipeline.main.fetch_api_pricing", return_value=FULL_PRICING),
            patch("pipeline.main.export_api_pricing"),
        ):
            updates = run_api_pricing_pipeline(snapshots_dir=snapshots_dir)

        assert any("API pricing updated" in u for u in updates)

    def test_raises_when_required_lab_missing(self, tmp_path):
        snapshots_dir = _write_snapshot(tmp_path, SAMPLE_BENCHMARKS)
        # Only OpenAI resolved — Anthropic and Google missing
        partial = {"GPT-5.4": FULL_PRICING["GPT-5.4"]}

        with (
            patch("pipeline.main.fetch_api_pricing", return_value=partial),
            patch("pipeline.main.export_api_pricing") as mock_export,
        ):
            with pytest.raises(RuntimeError, match="Missing required API pricing"):
                run_api_pricing_pipeline(snapshots_dir=snapshots_dir)

        mock_export.assert_not_called()

    def test_raises_lists_all_missing_labs_in_message(self, tmp_path):
        snapshots_dir = _write_snapshot(tmp_path, SAMPLE_BENCHMARKS)
        partial = {"GPT-5.4": FULL_PRICING["GPT-5.4"]}

        with (
            patch("pipeline.main.fetch_api_pricing", return_value=partial),
            patch("pipeline.main.export_api_pricing"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                run_api_pricing_pipeline(snapshots_dir=snapshots_dir)

        msg = str(exc_info.value)
        assert "anthropic" in msg
        assert "google" in msg

    def test_sends_alert_email_for_missing_required_labs(self, tmp_path):
        snapshots_dir = _write_snapshot(tmp_path, SAMPLE_BENCHMARKS)
        partial = {"GPT-5.4": FULL_PRICING["GPT-5.4"]}

        with (
            patch("pipeline.main.fetch_api_pricing", return_value=partial),
            patch("pipeline.main.export_api_pricing"),
            patch("pipeline.main.is_enabled", return_value=True),
            patch("pipeline.main.notify_missing_required_api_pricing") as mock_notify,
        ):
            with pytest.raises(RuntimeError):
                run_api_pricing_pipeline(snapshots_dir=snapshots_dir)

        mock_notify.assert_called_once()
        missing = mock_notify.call_args[0][0]
        missing_labs = {lab for lab, _ in missing}
        assert missing_labs == {"anthropic", "google"}

    def test_skips_gracefully_when_no_snapshot(self, tmp_path):
        empty_dir = tmp_path / "snapshots"
        empty_dir.mkdir()
        updates = run_api_pricing_pipeline(snapshots_dir=empty_dir)
        assert updates == []

    def test_export_not_called_when_no_pricing(self, tmp_path):
        snapshots_dir = _write_snapshot(tmp_path, SAMPLE_BENCHMARKS)

        with (
            patch("pipeline.main.fetch_api_pricing", return_value={}),
            patch("pipeline.main.export_api_pricing") as mock_export,
        ):
            with pytest.raises(RuntimeError):
                run_api_pricing_pipeline(snapshots_dir=snapshots_dir)

        mock_export.assert_not_called()
