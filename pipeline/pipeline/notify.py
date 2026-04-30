"""Email notifications for pipeline events.

Uses stdlib smtplib + email.message — no extra dependencies.
All notifications are best-effort: failures are logged but never crash the pipeline.
"""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from typing import TYPE_CHECKING

from pipeline.config import NOTIFY_TO, SMTP_PASSWORD, SMTP_USER

if TYPE_CHECKING:
    from pipeline.snapshots.exporter import NewSnapshotInfo

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


def notify_missing_api_pricing(model_name: str) -> None:
    """Alert that a model's LiteLLM pricing key could not be found."""
    send_email(
        subject=f"Missing LiteLLM pricing entry for {model_name}",
        body=(
            f"The model '{model_name}' is the current best-in-lab but has no matching "
            f"entry in the LiteLLM model_prices_and_context_window.json.\n\n"
            f"To fix this:\n"
            f"1. Find the model's key in https://raw.githubusercontent.com/BerriAI/litellm/"
            f"main/model_prices_and_context_window.json (look for direct-access, "
            f"non-cloud-routed entries).\n"
            f"2. Add/update LITELLM_ID_MAP in pipeline/sources/litellm_source.py.\n"
            f"3. See UPDATE-MODEL.md → 'Updating API Pricing Mapping' for details."
        ),
    )


def notify_missing_required_api_pricing(missing: list[tuple[str, str]]) -> None:
    """Alert that one or more required labs are missing from API pricing. Pipeline will fail."""
    lines = "\n".join(f"  - {lab}: {model_name}" for lab, model_name in missing)
    send_email(
        subject="Pipeline failed: missing required API pricing",
        body=(
            f"The following required labs could not be resolved to a LiteLLM pricing entry.\n"
            f"The pipeline has been aborted and api_pricing.json was NOT updated.\n\n"
            f"Missing:\n{lines}\n\n"
            f"To fix this:\n"
            f"1. Find the model's key in https://raw.githubusercontent.com/BerriAI/litellm/"
            f"main/model_prices_and_context_window.json (look for direct-access, "
            f"non-cloud-routed entries).\n"
            f"2. Add/update LITELLM_ID_MAP in pipeline/sources/litellm_source.py.\n"
            f"3. See UPDATE-MODEL.md → 'Updating API Pricing Mapping' for details."
        ),
    )


def notify_data_updated(
    updates: list[str],
    snapshot_infos: list[NewSnapshotInfo] | None = None,
) -> None:
    """Summarize what changed in a successful pipeline run."""
    body = "The pipeline completed successfully. Updates:\n\n"
    body += "\n".join(f"  - {u}" for u in updates)
    if snapshot_infos:
        coverage = format_snapshot_coverage(snapshot_infos)
        if coverage:
            body += f"\n\n{coverage}"
    send_email(subject="Source data updated", body=body)


def format_snapshot_coverage(infos: list[NewSnapshotInfo]) -> str:
    """Format only what changed in each snapshot: new models and coverage gains.

    Skips snapshots with no new models and no gained categories.
    When new_models is None (not computed), all models are shown as a fallback.
    """
    from pipeline.snapshots.constants import DISPLAY_NAMES

    def _fmt_cats(cats: list[str]) -> str:
        return ", ".join(DISPLAY_NAMES.get(c, c) for c in cats)

    sections: list[str] = []
    for info in infos:
        lines: list[str] = []

        if info.new_models is None:
            # Fallback: coverage diff not computed, show everything
            for model in sorted(info.model_coverage):
                cats = _fmt_cats(info.model_coverage[model])
                missing = info.model_missing.get(model, [])
                suffix = f"(missing: {_fmt_cats(missing)})" if missing else "(complete)"
                lines.append(f"  {model}: {cats} {suffix}")
        else:
            for model in sorted(info.new_models):
                cats = _fmt_cats(info.model_coverage[model])
                missing = info.model_missing.get(model, [])
                suffix = f"(missing: {_fmt_cats(missing)})" if missing else "(complete)"
                lines.append(f"  NEW {model}: {cats} {suffix}")
            for model in sorted(info.gained_categories):
                gained = _fmt_cats(info.gained_categories[model])
                lines.append(f"  +{model}: gained {gained}")

        if not lines:
            continue

        header = f"Snapshot {info.snapshot_date.isoformat()}:"
        sections.append("\n".join([header] + lines))

    return "\n\n".join(sections)
