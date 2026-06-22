"""Fetch GPU prices from gpuhunt and return offerings + source metadata."""

import logging
import re

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

# Datacenter GPUs whose standard multi-GPU form factor is SXM/NVLink (or
# NVSwitch). Their PCIe variants always carry "PCIe" in the gpuhunt
# instance_name (e.g. "NVIDIA H100 PCIe"), so an unlabelled offering for one
# of these is assumed to be NVLink-connected.
NVLINK_CLASS_GPUS = {
    "V100",
    "A100",
    "H100",
    "H100NVL",
    "H200",
    "B200",
    "B300",
    "GB200",
    "GH200",
}

# Explicit interconnect signals found in gpu_name / instance_name strings.
# PCIe is checked first and always wins — PCIe SKUs are reliably labelled.
_PCIE_RE = re.compile(r"pcie", re.IGNORECASE)
_NVLINK_RE = re.compile(r"sxm|hgx|dgx|nvlink|\bnvl", re.IGNORECASE)


def classify_interconnect(gpu_name: str, instance_name: str, gpu_count: int) -> str | None:
    """Best-effort interconnect classification for a GPU offering.

    Returns "nvlink", "pcie", or None. gpuhunt exposes no dedicated
    interconnect field, so we infer it from the gpu_name / instance_name
    strings, falling back to the GPU's standard datacenter form factor.

    Priority (first match wins):
      1. Single GPU (count < 2) -> None: there is no inter-GPU link.
      2. Explicit "PCIe" in the name -> "pcie" (PCIe SKUs are always labelled).
      3. Explicit "SXM"/"HGX"/"DGX"/"NVLink"/"NVL" -> "nvlink" (confirmed).
      4. SXM-class datacenter GPU -> "nvlink" (assumed; a PCIe variant would
         have matched step 2).
      5. Otherwise (consumer/workstation/unknown) -> "pcie" (conservative).

    The downstream calculator only distinguishes NVLink from everything else,
    so the conservative default never over-estimates throughput.
    """
    if gpu_count < 2:
        return None
    text = f"{gpu_name} {instance_name}"
    if _PCIE_RE.search(text):
        return "pcie"
    if _NVLINK_RE.search(text):
        return "nvlink"
    if gpu_name in NVLINK_CLASS_GPUS:
        return "nvlink"
    return "pcie"


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
                "interconnect": classify_interconnect(
                    gpu_name, item.instance_name, gpu_count
                ),
            }
        )

    logger.info("Found %d GPU offerings from gpuhunt", len(offerings))

    source_metadata = {
        "service_name": "gpuhunt",
        "service_url": "https://github.com/dstackai/gpuhunt",
        "description": "All regions considered. Interconnect (NVLink vs PCIe) is inferred from the instance name, falling back to the GPU's standard datacenter form factor.",
        "currency": "USD",
        "currency_symbol": "$",
    }

    return offerings, source_metadata
