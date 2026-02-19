"""Export pipeline data directly to JSON files for the frontend."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pipeline.config import EXPORT_DIR
from pipeline.enrichment import ModelSpec

logger = logging.getLogger(__name__)


def export_gpus(
    offerings: list[dict],
    output_dir: Path | None = None,
) -> Path:
    """Export GPU offerings to gpus.json.

    Adds a computed total_vram_gb field if not already present.
    """
    if output_dir is None:
        output_dir = EXPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for o in offerings:
        row = dict(o)
        if "total_vram_gb" not in row:
            row["total_vram_gb"] = o["vram_gb"] * o["gpu_count"]
        rows.append(row)

    rows.sort(key=lambda r: (r["gpu_name"], r["gpu_count"], r["price_per_hour"]))

    path = output_dir / "gpus.json"
    path.write_text(json.dumps(rows, indent=2))
    logger.info("Exported %d GPU offerings to %s", len(rows), path)
    return path


def export_gpu_source(
    source_metadata: dict,
    output_dir: Path | None = None,
) -> Path:
    """Export GPU source metadata to gpu_source.json."""
    if output_dir is None:
        output_dir = EXPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "gpu_source.json"
    path.write_text(json.dumps(source_metadata, indent=2) + "\n")
    logger.info("Exported GPU source metadata to %s", path)
    return path


def export_models(
    specs: list[ModelSpec],
    output_dir: Path | None = None,
) -> Path:
    """Export model specs to models.json."""
    if output_dir is None:
        output_dir = EXPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [spec.model_dump() for spec in specs]
    rows.sort(key=lambda r: r["model_name"])

    path = output_dir / "models.json"
    path.write_text(json.dumps(rows, indent=2))
    logger.info("Exported %d models to %s", len(rows), path)
    return path


def export_metadata(output_dir: Path | None = None) -> Path:
    """Export pipeline run metadata to metadata.json."""
    if output_dir is None:
        output_dir = EXPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {"updated_at": datetime.now(UTC).isoformat()}
    path = output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Exported pipeline metadata to %s", path)
    return path


def export_all(
    offerings: list[dict],
    specs: list[ModelSpec],
    output_dir: Path | None = None,
    source_metadata: dict | None = None,
) -> dict[str, Path]:
    """Export GPU and model data to JSON files.

    Benchmark and SOTA data are now handled by the snapshot pipeline.
    """
    result: dict[str, Path] = {
        "gpus": export_gpus(offerings, output_dir),
        "models": export_models(specs, output_dir),
        "metadata": export_metadata(output_dir),
    }
    if source_metadata is not None:
        result["gpu_source"] = export_gpu_source(source_metadata, output_dir)
    return result
