"""Export pipeline data directly to JSON files for the frontend."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pipeline.config import EXPORT_DIR
from pipeline.enrichment import ModelSpec
from pipeline.gpu_nodes import reduce_offerings_to_nodes

logger = logging.getLogger(__name__)

PRICE_HISTORY_UNIT = "usd_per_node_hour"


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


def export_gpu_specs(
    gpu_specs: list[dict],
    output_dir: Path | None = None,
) -> Path:
    """Export GPU throughput specs to gpu_specs.json.

    Appends the hardcoded GH200 entry (not available in dbgpu).
    """
    if output_dir is None:
        output_dir = EXPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # GH200: not in dbgpu. Source: pipeline/sources/gh200_specs.md (NVIDIA datasheet)
    # HBM size: 96 GB HBM3
    # FP16 Tensor Core: 990 TFLOPS (without sparsity)
    # Memory bandwidth: Up to 4 TB/s (HBM3)
    # NVLink-C2C: 900 GB/s
    specs = list(gpu_specs)
    specs.append(
        {
            "gpu_name": "GH200",
            "memory_size_gb": 96,
            "fp16_tflops": 990,
            "memory_bandwidth_tb_s": 4.0,
            "pcie_bandwidth_gb_s": 64.0,  # PCIe 5.0 x16
            "nvlink_bandwidth_gb_s": 900,  # NVLink-C2C
            "fp8_multiplier": 2,
            "architecture": "Hopper",
            "interconnect_tier": "nvswitch",
            "memory_type": "HBM3e",
        }
    )

    specs.sort(key=lambda r: r["gpu_name"])

    path = output_dir / "gpu_specs.json"
    path.write_text(json.dumps(specs, indent=2) + "\n")
    logger.info("Exported %d GPU specs to %s", len(specs), path)
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


def export_api_pricing(
    pricing: dict[str, dict],
    output_dir: Path | None = None,
) -> Path:
    """Export API pricing data to api_pricing.json.

    Each entry preserves all LiteLLM fields plus model_name, lab, and litellm_id.
    Sorted by lab name for determinism.
    """
    if output_dir is None:
        output_dir = EXPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted(pricing.values(), key=lambda r: (r.get("lab", ""), r.get("model_name", "")))

    path = output_dir / "api_pricing.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    logger.info("Exported API pricing for %d models to %s", len(rows), path)
    return path


def export_gpu_price_history(
    offerings: list[dict],
    output_dir: Path | None = None,
) -> Path:
    """Append today's cheapest node prices to gpu_price_history.json.

    Reduces ``offerings`` to one cheapest 8x price per curated node (via the
    shared ``reduce_offerings_to_nodes``) and UPSERTS a
    ``{"date": <today UTC>, "prices": [...]}`` entry into the stored history:

      * loads the existing file, or creates the
        ``{generated_at, unit, series: []}`` skeleton if missing;
      * replaces any existing entry for today's date (no duplicates);
      * keeps ``series`` sorted by ascending date and refreshes ``generated_at``.

    Mirroring the snapshot exporter, the write is skipped when the resulting
    content is byte-identical apart from ``generated_at`` — so a same-day rerun
    with unchanged prices is a genuine no-op (no churn, no timestamp bump).
    """
    if output_dir is None:
        output_dir = EXPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    prices = reduce_offerings_to_nodes(offerings)
    today = datetime.now(UTC).date().isoformat()

    path = output_dir / "gpu_price_history.json"
    existing_text = path.read_text() if path.exists() else None
    if existing_text is not None:
        history = json.loads(existing_text)
    else:
        history = {
            "generated_at": datetime.now(UTC).isoformat(),
            "unit": PRICE_HISTORY_UNIT,
            "series": [],
        }

    series = [e for e in history.get("series", []) if e.get("date") != today]
    series.append({"date": today, "prices": prices})
    series.sort(key=lambda e: e["date"])

    history["unit"] = PRICE_HISTORY_UNIT
    history["series"] = series
    history["generated_at"] = datetime.now(UTC).isoformat()

    # Skip the write when the content is unchanged (ignoring generated_at), so a
    # same-day rerun leaves the file byte-identical rather than bumping the stamp.
    if existing_text is not None:
        old = json.loads(existing_text)
        if old.get("unit") == history["unit"] and old.get("series") == series:
            logger.info("GPU price history unchanged for %s; skipping write", today)
            return path

    path.write_text(json.dumps(history, indent=2) + "\n")
    logger.info("Exported GPU node prices for %s to %s", today, path)
    return path


def export_all(
    offerings: list[dict],
    specs: list[ModelSpec],
    output_dir: Path | None = None,
    source_metadata: dict | None = None,
    gpu_specs: list[dict] | None = None,
) -> dict[str, Path]:
    """Export GPU and model data to JSON files.

    Benchmark and SOTA data are now handled by the snapshot pipeline.
    """
    result: dict[str, Path] = {
        "gpus": export_gpus(offerings, output_dir),
        "models": export_models(specs, output_dir),
        "metadata": export_metadata(output_dir),
        "gpu_price_history": export_gpu_price_history(offerings, output_dir),
    }
    if source_metadata is not None:
        result["gpu_source"] = export_gpu_source(source_metadata, output_dir)
    if gpu_specs is not None:
        result["gpu_specs"] = export_gpu_specs(gpu_specs, output_dir)
    return result
