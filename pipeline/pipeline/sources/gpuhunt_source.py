"""Fetch GPU prices from gpuhunt and return offerings + source metadata."""

import logging
from datetime import UTC, datetime

from gpuhunt import Catalog

from pipeline.errors import FormatBreakingChange

logger = logging.getLogger(__name__)

# GPUs not suitable for LLM serving (too little VRAM, old arch, consumer cards)
EXCLUDED_GPUS = {
    "P100",
    "T4",
    "RTX2000Ada",
    "RTX3070",
    "RTX3080",
    "RTX3080Ti",
    "RTX4070Ti",
    "RTX4080",
    "RTX4080SUPER",
    "RTX5080",
}

# A10 MIG slices (4/8/12 GB) are not full GPUs — only keep 24 GB
MIN_VRAM_GB = 16


def fetch_gpu_prices() -> tuple[list[dict], dict]:
    """Fetch GPU prices from gpuhunt catalog.

    Returns (offerings, source_metadata) where offerings use the same schema
    as gpus.json and source_metadata describes the data source.
    """
    catalog = Catalog(balance_resources=False, auto_reload=True)
    logger.info("Querying gpuhunt catalog for NVIDIA on-demand offerings")

    items = catalog.query(
        gpu_vendor="nvidia",
        spot=False,
        min_gpu_count=1,
    )

    # Validate gpuhunt result format — detect breaking API changes early
    _EXPECTED_ATTRS = ("gpu_name", "gpu_memory", "gpu_count", "price", "provider",
                       "instance_name", "location")
    if items:
        first = items[0]
        missing = [a for a in _EXPECTED_ATTRS if not hasattr(first, a)]
        if missing:
            raise FormatBreakingChange(
                source="gpuhunt",
                details=(
                    f"Query results are missing expected attributes: {missing}. "
                    f"The gpuhunt Catalog API may have changed. "
                    f"Available attributes: {sorted(vars(first).keys())}"
                ),
            )

    # Group by (gpu_name, gpu_memory, gpu_count), keep cheapest
    best: dict[tuple[str, float, int], object] = {}
    for item in items:
        if item.gpu_name in EXCLUDED_GPUS:
            continue
        if item.gpu_memory < MIN_VRAM_GB:
            continue
        key = (item.gpu_name, item.gpu_memory, item.gpu_count)
        if key not in best or item.price < best[key].price:
            best[key] = item

    # Build output
    offerings = []
    for (gpu_name, gpu_memory, gpu_count), item in sorted(best.items()):
        offerings.append(
            {
                "gpu_name": gpu_name,
                "vram_gb": gpu_memory,
                "gpu_count": gpu_count,
                "total_vram_gb": round(gpu_memory * gpu_count, 1),
                "price_per_hour": item.price,
                "currency": "USD",
                "provider": item.provider,
                "instance_name": item.instance_name,
                "location": item.location,
                "interconnect": None,
            }
        )

    logger.info("Found %d GPU offerings from gpuhunt", len(offerings))

    source_metadata = {
        "service_name": "gpuhunt",
        "service_url": "https://github.com/dstackai/gpuhunt",
        "description": "All regions considered. Throughput values may be underestimated, because interconnect data is missing.",
        "currency": "USD",
        "currency_symbol": "$",
        "updated_at": datetime.now(UTC).isoformat(),
    }

    return offerings, source_metadata
