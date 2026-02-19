"""Tests for the email notification module."""

from unittest.mock import MagicMock, patch

from pipeline.notify import (
    SUBJECT_PREFIX,
    is_enabled,
    notify_breaking_format_change,
    notify_data_updated,
    notify_failure,
    notify_missing_mapping,
    send_email,
)


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
