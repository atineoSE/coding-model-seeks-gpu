"""Shared test fixtures for the pipeline test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def block_real_smtp(monkeypatch):
    """Disable email notifications by default in every test.

    Sets the SMTP config to None so `is_enabled()` returns False, and replaces
    `smtplib.SMTP` with a function that raises if invoked. Tests that exercise
    the email path opt back in by re-monkeypatching the same attributes (the
    test-local override wins for the duration of the test).
    """
    monkeypatch.setattr("pipeline.notify.SMTP_USER", None)
    monkeypatch.setattr("pipeline.notify.SMTP_PASSWORD", None)
    monkeypatch.setattr("pipeline.notify.NOTIFY_TO", None)

    def _refuse(*_args, **_kwargs):
        raise RuntimeError(
            "Real SMTP connection attempted in a test. "
            "Mock pipeline.notify.smtplib.SMTP or set SMTP_* config explicitly."
        )

    monkeypatch.setattr("pipeline.notify.smtplib.SMTP", _refuse)
