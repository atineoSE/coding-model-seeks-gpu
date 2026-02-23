"""CLI entry point for the GPU Cost Explorer data pipeline."""

import argparse
import json
import logging
import sys
import time
import traceback

from pipeline.config import SNAPSHOTS_DIR, SUBMODULE_PATH
from pipeline.errors import FormatBreakingChange
from pipeline.exporters.json_export import export_all
from pipeline.notify import (
    is_enabled,
    notify_breaking_format_change,
    notify_data_updated,
    notify_failure,
    notify_missing_mapping,
    notify_unsupported_architecture,
)
from pipeline.snapshots.exporter import load_index, run_snapshot_export
from pipeline.sources.dbgpu_source import fetch_gpu_specs
from pipeline.sources.gpuhunt_source import fetch_gpu_prices
from pipeline.sources.huggingface import MODEL_NAME_TO_HF_ID, fetch_all_models

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

CLOSED_SOURCE_PREFIXES = {"claude-", "gpt-", "gemini-", "GPT-", "Gemini-"}


def run_gpu_pipeline() -> tuple[list[dict], dict]:
    """Fetch GPU prices from gpuhunt and return (offerings, source_metadata)."""
    logger.info("=== GPU Pipeline ===")
    offerings, source_metadata = fetch_gpu_prices()
    logger.info("GPU pipeline complete: %d offerings", len(offerings))
    return offerings, source_metadata


def run_gpu_specs_pipeline() -> list[dict]:
    """Fetch GPU hardware specs from dbgpu and return spec dicts."""
    logger.info("=== GPU Specs Pipeline ===")
    gpu_specs = fetch_gpu_specs()
    logger.info("GPU specs pipeline complete: %d GPUs", len(gpu_specs))
    return gpu_specs


def run_snapshot_pipeline(*, force: bool = False) -> int:
    """Generate historical snapshots from the git submodule."""
    logger.info("=== Snapshot Pipeline ===")
    count = run_snapshot_export(SUBMODULE_PATH, SNAPSHOTS_DIR, force=force)
    logger.info("Snapshot pipeline complete: %d new snapshots", count)
    return count


def run_model_pipeline():
    """Fetch HuggingFace model configs and return (specs, skipped).

    Models with unsupported architectures are returned in *skipped* rather
    than causing a pipeline failure.
    """
    logger.info("=== Model Pipeline ===")
    specs, skipped = fetch_all_models()
    for spec in specs:
        logger.info("  Model '%s'", spec.model_name)
    logger.info("Model pipeline complete: %d models", len(specs))
    return specs, skipped


def run_export(offerings, specs, source_metadata=None, gpu_specs=None):
    """Export GPU and model data to JSON."""
    logger.info("=== JSON Export ===")
    paths = export_all(offerings, specs, source_metadata=source_metadata, gpu_specs=gpu_specs)
    for name, path in paths.items():
        logger.info("Exported %s -> %s", name, path)
    return paths


def _is_closed_source(model_name: str) -> bool:
    """Return True if the model name matches a known closed-source prefix."""
    return any(model_name.startswith(prefix) for prefix in CLOSED_SOURCE_PREFIXES)


def check_missing_mappings(snapshots_dir=None):
    """Check for benchmark models that have no HuggingFace mapping.

    Reads the latest snapshot's benchmarks.json, extracts unique model names,
    and compares against MODEL_NAME_TO_HF_ID. Closed-source models are skipped.
    """
    if snapshots_dir is None:
        snapshots_dir = SNAPSHOTS_DIR

    index = load_index(snapshots_dir)
    if index is None or not index.get("latest"):
        logger.debug("No snapshot index found, skipping mapping check")
        return

    latest = index["latest"]
    benchmarks_path = snapshots_dir / latest / "benchmarks.json"
    if not benchmarks_path.exists():
        logger.debug("No benchmarks.json for latest snapshot %s", latest)
        return

    try:
        benchmarks = json.loads(benchmarks_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read benchmarks.json for %s", latest)
        return

    benchmark_models = {entry["model_name"] for entry in benchmarks if "model_name" in entry}
    known_models = set(MODEL_NAME_TO_HF_ID.keys())

    for model_name in sorted(benchmark_models):
        if model_name in known_models:
            continue
        if _is_closed_source(model_name):
            continue
        logger.warning("Missing HF mapping for benchmark model: %s", model_name)
        if is_enabled():
            notify_missing_mapping(model_name)


def main():
    parser = argparse.ArgumentParser(description="GPU Cost Explorer Data Pipeline")
    parser.add_argument(
        "--step",
        choices=["gpu", "snapshots", "models", "export", "all"],
        default="all",
        help="Which pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--force-snapshots",
        action="store_true",
        help="Force full snapshot regeneration (ignore existing index)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            updates = []

            offerings = []
            source_metadata = None
            gpu_specs = None
            if args.step in ("gpu", "all"):
                offerings, source_metadata = run_gpu_pipeline()
                gpu_specs = run_gpu_specs_pipeline()

            if args.step in ("snapshots", "all"):
                new_snapshots = run_snapshot_pipeline(force=args.force_snapshots)
                if new_snapshots > 0:
                    updates.append(f"New benchmark snapshots: {new_snapshots}")

            # Check for missing mappings between snapshot and model steps
            if args.step in ("all",):
                check_missing_mappings()

            specs = []
            if args.step in ("models", "all"):
                specs, skipped = run_model_pipeline()
                if specs:
                    updates.append(f"Models enriched: {len(specs)}")
                if skipped and is_enabled():
                    for err in skipped:
                        notify_unsupported_architecture(
                            err.model_name, err.model_type, err.hf_id
                        )

            if args.step in ("export", "all"):
                run_export(offerings, specs, source_metadata=source_metadata, gpu_specs=gpu_specs)

            # Success — send data update notification if anything changed
            if updates and is_enabled():
                notify_data_updated(updates)

            logger.info("Pipeline complete!")
            return  # Success, exit retry loop

        except FormatBreakingChange as e:
            # Format breaks won't fix themselves — skip retries, alert immediately
            logger.exception("Breaking format change detected: %s", e)
            if is_enabled():
                notify_breaking_format_change(e.source, e.details)
            sys.exit(1)

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt,
                    MAX_RETRIES,
                    e,
                    RETRY_DELAY,
                )
                time.sleep(RETRY_DELAY)
            else:
                logger.exception("Pipeline failed after %d attempts", MAX_RETRIES)
                if is_enabled():
                    notify_failure(last_error, traceback.format_exc())
                sys.exit(1)


if __name__ == "__main__":
    main()
