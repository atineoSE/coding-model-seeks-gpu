"""Tests for the email notification module."""

from datetime import date
from unittest.mock import MagicMock, patch

from pipeline.notify import (
    SUBJECT_PREFIX,
    format_snapshot_coverage,
    is_enabled,
    notify_breaking_format_change,
    notify_data_updated,
    notify_failure,
    notify_missing_mapping,
    notify_missing_required_api_pricing,
    notify_new_snapshots,
    send_email,
)
from pipeline.snapshots.exporter import NewSnapshotInfo


class TestIsEnabled:
    def test_enabled_when_all_vars_set(self, monkeypatch):
        monkeypatch.setattr("pipeline.notify.SMTP_USER", "user@gmail.com")
        monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", "secret")
        monkeypatch.setattr("pipeline.notify.NOTIFY_TO", "dest@example.com")
        assert is_enabled() is True

    def test_disabled_when_user_missing(self, monkeypatch):
        monkeypatch.setattr("pipeline.notify.SMTP_USER", None)
        monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", "secret")
        monkeypatch.setattr("pipeline.notify.NOTIFY_TO", "dest@example.com")
        assert is_enabled() is False

    def test_disabled_when_password_missing(self, monkeypatch):
        monkeypatch.setattr("pipeline.notify.SMTP_USER", "user@gmail.com")
        monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", None)
        monkeypatch.setattr("pipeline.notify.NOTIFY_TO", "dest@example.com")
        assert is_enabled() is False

    def test_disabled_when_notify_to_missing(self, monkeypatch):
        monkeypatch.setattr("pipeline.notify.SMTP_USER", "user@gmail.com")
        monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", "secret")
        monkeypatch.setattr("pipeline.notify.NOTIFY_TO", None)
        assert is_enabled() is False


class TestSendEmail:
    def test_calls_smtp(self, monkeypatch):
        monkeypatch.setattr("pipeline.notify.SMTP_USER", "user@gmail.com")
        monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", "secret")
        monkeypatch.setattr("pipeline.notify.NOTIFY_TO", "dest@example.com")

        mock_smtp_instance = MagicMock()
        mock_smtp_cls = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = MagicMock(return_value=False)

        with patch("pipeline.notify.smtplib.SMTP", mock_smtp_cls):
            send_email("Test Subject", "Test body")

        mock_smtp_cls.assert_called_once_with("smtp.gmail.com", 587)
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("user@gmail.com", "secret")
        mock_smtp_instance.send_message.assert_called_once()

        sent_msg = mock_smtp_instance.send_message.call_args[0][0]
        assert sent_msg["Subject"] == f"{SUBJECT_PREFIX} Test Subject"
        assert sent_msg["From"] == "user@gmail.com"
        assert sent_msg["To"] == "dest@example.com"

    def test_swallows_errors(self, monkeypatch):
        monkeypatch.setattr("pipeline.notify.SMTP_USER", "user@gmail.com")
        monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", "secret")
        monkeypatch.setattr("pipeline.notify.NOTIFY_TO", "dest@example.com")

        with patch("pipeline.notify.smtplib.SMTP", side_effect=OSError("connection failed")):
            # Should not raise
            send_email("Test", "body")

    def test_skips_when_disabled(self, monkeypatch):
        monkeypatch.setattr("pipeline.notify.SMTP_USER", None)
        monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", None)
        monkeypatch.setattr("pipeline.notify.NOTIFY_TO", None)

        with patch("pipeline.notify.smtplib.SMTP") as mock_smtp:
            send_email("Test", "body")
            mock_smtp.assert_not_called()


class TestNotifyMissingMapping:
    def test_subject_format(self, monkeypatch):
        monkeypatch.setattr("pipeline.notify.SMTP_USER", "user@gmail.com")
        monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", "secret")
        monkeypatch.setattr("pipeline.notify.NOTIFY_TO", "dest@example.com")

        mock_smtp_instance = MagicMock()
        mock_smtp_cls = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = MagicMock(return_value=False)

        with patch("pipeline.notify.smtplib.SMTP", mock_smtp_cls):
            notify_missing_mapping("NewModel-7B")

        sent_msg = mock_smtp_instance.send_message.call_args[0][0]
        expected = f"{SUBJECT_PREFIX} Missing HuggingFace mapping for NewModel-7B"
        assert sent_msg["Subject"] == expected


class TestNotifyFailure:
    def test_subject_format(self, monkeypatch):
        monkeypatch.setattr("pipeline.notify.SMTP_USER", "user@gmail.com")
        monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", "secret")
        monkeypatch.setattr("pipeline.notify.NOTIFY_TO", "dest@example.com")

        mock_smtp_instance = MagicMock()
        mock_smtp_cls = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = MagicMock(return_value=False)

        with patch("pipeline.notify.smtplib.SMTP", mock_smtp_cls):
            notify_failure(RuntimeError("boom"), "Traceback...\nRuntimeError: boom")

        sent_msg = mock_smtp_instance.send_message.call_args[0][0]
        assert sent_msg["Subject"] == f"{SUBJECT_PREFIX} Pipeline failed"
        body = sent_msg.get_payload()
        assert "boom" in body
        assert "Traceback" in body


class TestNotifyBreakingFormatChange:
    def test_subject_and_body(self, monkeypatch):
        monkeypatch.setattr("pipeline.notify.SMTP_USER", "user@gmail.com")
        monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", "secret")
        monkeypatch.setattr("pipeline.notify.NOTIFY_TO", "dest@example.com")

        mock_smtp_instance = MagicMock()
        mock_smtp_cls = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = MagicMock(return_value=False)

        with patch("pipeline.notify.smtplib.SMTP", mock_smtp_cls):
            notify_breaking_format_change("gpuhunt", "Missing attributes: ['gpu_name']")

        sent_msg = mock_smtp_instance.send_message.call_args[0][0]
        assert sent_msg["Subject"] == f"{SUBJECT_PREFIX} Data format breaking change"
        body = sent_msg.get_payload()
        assert "gpuhunt" in body
        assert "Missing attributes" in body


class TestNotifyMissingRequiredApiPricing:
    def _smtp_mocks(self, monkeypatch):
        monkeypatch.setattr("pipeline.notify.SMTP_USER", "user@gmail.com")
        monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", "secret")
        monkeypatch.setattr("pipeline.notify.NOTIFY_TO", "dest@example.com")
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_cls = MagicMock(return_value=mock_instance)
        return mock_cls, mock_instance

    def test_subject_signals_failure(self, monkeypatch):
        mock_cls, mock_instance = self._smtp_mocks(monkeypatch)
        with patch("pipeline.notify.smtplib.SMTP", mock_cls):
            notify_missing_required_api_pricing([("anthropic", "claude-opus-4-6")])
        sent = mock_instance.send_message.call_args[0][0]
        assert "Pipeline failed" in sent["Subject"]
        assert SUBJECT_PREFIX in sent["Subject"]

    def test_body_lists_all_missing_labs(self, monkeypatch):
        mock_cls, mock_instance = self._smtp_mocks(monkeypatch)
        missing = [("anthropic", "claude-opus-4-6"), ("google", "Gemini-3.1-Pro")]
        with patch("pipeline.notify.smtplib.SMTP", mock_cls):
            notify_missing_required_api_pricing(missing)
        body = mock_instance.send_message.call_args[0][0].get_payload()
        assert "anthropic" in body
        assert "claude-opus-4-6" in body
        assert "google" in body
        assert "Gemini-3.1-Pro" in body

    def test_body_includes_fix_instructions(self, monkeypatch):
        mock_cls, mock_instance = self._smtp_mocks(monkeypatch)
        with patch("pipeline.notify.smtplib.SMTP", mock_cls):
            notify_missing_required_api_pricing([("openai", "GPT-5.4")])
        body = mock_instance.send_message.call_args[0][0].get_payload()
        assert "LITELLM_ID_MAP" in body


class TestNotifyDataUpdated:
    def test_subject_and_body(self, monkeypatch):
        monkeypatch.setattr("pipeline.notify.SMTP_USER", "user@gmail.com")
        monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", "secret")
        monkeypatch.setattr("pipeline.notify.NOTIFY_TO", "dest@example.com")

        mock_smtp_instance = MagicMock()
        mock_smtp_cls = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = MagicMock(return_value=False)

        with patch("pipeline.notify.smtplib.SMTP", mock_smtp_cls):
            notify_data_updated(["GPU prices refreshed: 500 offerings", "Models enriched: 7"])

        sent_msg = mock_smtp_instance.send_message.call_args[0][0]
        assert sent_msg["Subject"] == f"{SUBJECT_PREFIX} Source data updated"
        body = sent_msg.get_payload()
        assert "500 offerings" in body
        assert "Models enriched: 7" in body


class TestFormatSnapshotCoverage:
    def test_complete_model(self):
        info = NewSnapshotInfo(snapshot_date=date(2026, 4, 24))
        info.model_coverage["Qwen4-72B"] = [
            "frontend", "greenfield", "information_gathering",
            "issue_resolution", "testing",
        ]
        # No missing categories
        result = format_snapshot_coverage([info])
        assert "2026-04-24" in result
        assert "Qwen4-72B" in result
        assert "(complete)" in result

    def test_model_with_missing_categories(self):
        info = NewSnapshotInfo(snapshot_date=date(2026, 4, 24))
        info.model_coverage["GPT-5.5"] = [
            "frontend", "greenfield", "information_gathering",
        ]
        info.model_missing["GPT-5.5"] = ["issue_resolution", "testing"]
        result = format_snapshot_coverage([info])
        assert "GPT-5.5" in result
        assert "Frontend" in result
        assert "(missing: Issue Resolution, Testing)" in result

    def test_multiple_models_sorted(self):
        info = NewSnapshotInfo(snapshot_date=date(2026, 4, 24))
        info.model_coverage["Zebra-7B"] = ["frontend"]
        info.model_missing["Zebra-7B"] = ["greenfield"]
        info.model_coverage["Alpha-7B"] = ["frontend"]
        info.model_missing["Alpha-7B"] = ["greenfield"]
        result = format_snapshot_coverage([info])
        alpha_pos = result.index("Alpha-7B")
        zebra_pos = result.index("Zebra-7B")
        assert alpha_pos < zebra_pos

    def test_multiple_snapshots(self):
        info1 = NewSnapshotInfo(snapshot_date=date(2026, 4, 23))
        info1.model_coverage["ModelA"] = ["frontend"]
        info2 = NewSnapshotInfo(snapshot_date=date(2026, 4, 24))
        info2.model_coverage["ModelB"] = ["testing"]
        result = format_snapshot_coverage([info1, info2])
        assert "2026-04-23" in result
        assert "2026-04-24" in result
        assert "ModelA" in result
        assert "ModelB" in result

    def test_empty_list(self):
        result = format_snapshot_coverage([])
        assert result == ""


class TestNotifyNewSnapshots:
    def _smtp_mocks(self, monkeypatch):
        monkeypatch.setattr("pipeline.notify.SMTP_USER", "user@gmail.com")
        monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", "secret")
        monkeypatch.setattr("pipeline.notify.NOTIFY_TO", "dest@example.com")
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_cls = MagicMock(return_value=mock_instance)
        return mock_cls, mock_instance

    def test_single_snapshot_subject(self, monkeypatch):
        mock_cls, mock_instance = self._smtp_mocks(monkeypatch)
        info = NewSnapshotInfo(snapshot_date=date(2026, 4, 24))
        info.model_coverage["ModelA"] = ["frontend"]
        with patch("pipeline.notify.smtplib.SMTP", mock_cls):
            notify_new_snapshots([info])
        sent = mock_instance.send_message.call_args[0][0]
        assert "New snapshot: 1 generated" in sent["Subject"]

    def test_multiple_snapshots_subject(self, monkeypatch):
        mock_cls, mock_instance = self._smtp_mocks(monkeypatch)
        infos = [
            NewSnapshotInfo(snapshot_date=date(2026, 4, 23)),
            NewSnapshotInfo(snapshot_date=date(2026, 4, 24)),
        ]
        with patch("pipeline.notify.smtplib.SMTP", mock_cls):
            notify_new_snapshots(infos)
        sent = mock_instance.send_message.call_args[0][0]
        assert "New snapshots: 2 generated" in sent["Subject"]

    def test_body_contains_coverage(self, monkeypatch):
        mock_cls, mock_instance = self._smtp_mocks(monkeypatch)
        info = NewSnapshotInfo(snapshot_date=date(2026, 4, 24))
        info.model_coverage["GPT-5.5"] = ["frontend", "greenfield"]
        info.model_missing["GPT-5.5"] = ["testing"]
        with patch("pipeline.notify.smtplib.SMTP", mock_cls):
            notify_new_snapshots([info])
        body = mock_instance.send_message.call_args[0][0].get_payload()
        assert "GPT-5.5" in body
        assert "Frontend" in body
        assert "missing: Testing" in body

    def test_empty_infos_skips_email(self, monkeypatch):
        mock_cls, mock_instance = self._smtp_mocks(monkeypatch)
        with patch("pipeline.notify.smtplib.SMTP", mock_cls):
            notify_new_snapshots([])
        mock_instance.send_message.assert_not_called()
