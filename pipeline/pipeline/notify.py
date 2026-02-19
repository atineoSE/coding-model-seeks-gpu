"""Email notifications for pipeline events.

Uses stdlib smtplib + email.message — no extra dependencies.
All notifications are best-effort: failures are logged but never crash the pipeline.
"""

import logging
import smtplib
from email.message import EmailMessage

from pipeline.config import NOTIFY_TO, SMTP_PASSWORD, SMTP_USER

logger = logging.getLogger(__name__)

SUBJECT_PREFIX = "[coding-model-seeks-gpu]"


def is_enabled() -> bool:
    """Return True when all required email config vars are set."""
    return all([SMTP_USER, SMTP_PASSWORD, NOTIFY_TO])


def send_email(subject: str, body: str) -> None:
    """Send an email via Gmail SMTP.

    Subject is automatically prefixed with ``[coding-model-seeks-gpu]``.
    Swallows all exceptions so notification failure never crashes the pipeline.
    """
    if not is_enabled():
        return

    full_subject = f"{SUBJECT_PREFIX} {subject}"

    msg = EmailMessage()
    msg["Subject"] = full_subject
    msg["From"] = SMTP_USER
    msg["To"] = NOTIFY_TO
    msg.set_content(body)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        logger.info("Sent email: %s", full_subject)
    except Exception:
        logger.warning("Failed to send email: %s", full_subject, exc_info=True)


def notify_unsupported_architecture(model_name: str, model_type: str, hf_id: str) -> None:
    """Alert that a model's HF architecture is not yet supported."""
    send_email(
        subject=f"Unsupported architecture for {model_name}",
        body=(
            f"The model '{model_name}' ({hf_id}) has model_type='{model_type}' "
            f"which is not in KNOWN_ARCHITECTURES.\n\n"
            f"The model was skipped during enrichment. Please add support for "
            f"this architecture in param_counter.py so the pipeline can include it."
        ),
    )


def notify_missing_mapping(model_name: str) -> None:
    """Alert that a benchmark model has no HuggingFace repo mapping."""
    send_email(
        subject=f"Missing HuggingFace mapping for {model_name}",
        body=(
            f"The model '{model_name}' appeared in the OpenHands Index benchmarks "
            f"but has no entry in MODEL_NAME_TO_HF_ID.\n\n"
            f"Please add a mapping so the pipeline can fetch its HuggingFace config."
        ),
    )


def notify_failure(error: Exception, traceback_str: str) -> None:
    """Alert that the pipeline failed after all retries."""
    send_email(
        subject="Pipeline failed",
        body=(
            f"The pipeline failed after all retry attempts.\n\n"
            f"Error: {error}\n\n"
            f"Traceback:\n{traceback_str}"
        ),
    )


def notify_breaking_format_change(source: str, details: str) -> None:
    """Alert that an upstream data source has a breaking format change."""
    send_email(
        subject="Data format breaking change",
        body=(
            f"A breaking change was detected in the data format from '{source}'.\n\n"
            f"Details:\n{details}\n\n"
            f"The pipeline cannot process data from this source until the code is "
            f"updated to handle the new format."
        ),
    )


def notify_data_updated(updates: list[str]) -> None:
    """Summarize what changed in a successful pipeline run."""
    body = "The pipeline completed successfully. Updates:\n\n"
    body += "\n".join(f"  - {u}" for u in updates)
    send_email(subject="Source data updated", body=body)
