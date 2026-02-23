"""GPU hardware specs sourced from dbgpu (TechPowerUp database).

Provides traceable GPU specifications for LLM inference throughput calculations.
Every value cites its origin: dbgpu (TechPowerUp), Wikipedia NVLink article,
or NVIDIA datasheets stored in pipeline/sources/.
"""

import logging

from dbgpu import GPUDatabase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU name mapping: our key → dbgpu specification key (slug)
#
# These slugs correspond to db.specifications[key].
# RTXPRO6000WK maps to the same dbgpu entry as RTXPRO6000 (same 96 GB hardware).
# ---------------------------------------------------------------------------
GPUHUNT_TO_DBGPU_KEY: dict[str, str] = {
    "A10": "a10-pcie",
    "A10G": "a10g",
    "A100": "a100-pcie-40gb",
    "A100_80G": "a100-sxm4-80gb",
    "A40": "a40-pcie",
    "A4000": "rtx-a4000",
    "A4500": "rtx-a4500",
    "A5000": "rtx-a5000",
    "A6000": "rtx-a6000",
    "B200": "b200",
    "B300": "b300",
    "H100": "h100-sxm5-80gb",
    "H100NVL": "h100-nvl-94gb",
    "H200": "h200-sxm-141gb",
    "L4": "l4",
    "L40": "l40",
    "L40S": "l40s",
    "RTX3090": "geforce-rtx-3090",
    "RTX3090Ti": "geforce-rtx-3090-ti",
    "RTX4000Ada": "rtx-4000-ada-generation",
    "RTX4090": "geforce-rtx-4090",
    "RTX5000Ada": "rtx-5000-ada-generation",
    "RTX5090": "geforce-rtx-5090",
    "RTX6000": "quadro-rtx-6000",
    "RTX6000Ada": "rtx-6000-ada-generation",
    "RTXPRO4500": "rtx-pro-4500-blackwell",
    "RTXPRO6000": "rtx-pro-6000-blackwell",
    "RTXPRO6000MaxQ": "rtx-pro-6000-blackwell-max-q",
    "RTXPRO6000WK": "rtx-pro-6000-blackwell",  # same hardware as RTXPRO6000
    "V100": "tesla-v100-sxm2-16gb",
}

# ---------------------------------------------------------------------------
# Dual-die packaging: dbgpu reports per-die specs; multiply by die count.
# Source: dbgpu gpu_name field. B200/B300 use dual-die (MCM) packaging.
# ---------------------------------------------------------------------------
MULTI_DIE_CHIPS: dict[str, int] = {
    "GB100": 2,  # B200
    "GB110": 2,  # B300
}

# ---------------------------------------------------------------------------
# Architecture normalization: dbgpu architecture → our canonical names.
# Source: dbgpu architecture field variations.
# ---------------------------------------------------------------------------
ARCH_NORMALIZATION: dict[str, str] = {
    "Blackwell 2.0": "Blackwell",
    "Blackwell Ultra": "Blackwell",
}

# ---------------------------------------------------------------------------
# NVLink bandwidth per chip (GB/s).
# Source: Wikipedia NVLink article (pipeline/sources/NVLink.html),
# per-semiconductor table.
# ---------------------------------------------------------------------------
CHIP_NVLINK_BANDWIDTH: dict[str, float] = {
    "GV100": 300,     # Volta — NVLink 2.0
    "TU102": 100,     # Turing — NVLink (1 bridge)
    "GA100": 600,     # Ampere — NVLink 3.0
    "GA102": 112.5,   # Ampere — NVLink (1 bridge)
    "GH100": 900,     # Hopper — NVLink 4.0
    "GB100": 1800,    # Blackwell — NVLink 5.0
    "GB110": 1800,    # Blackwell Ultra — NVLink 5.0
}

# ---------------------------------------------------------------------------
# Products with an NVLink-capable chip but no NVLink connector on the board.
# Sources: NVIDIA datasheets (pipeline/sources/*.md), TechPowerUp product pages.
# ---------------------------------------------------------------------------
NO_NVLINK: set[str] = {
    # GA102, PCIe-only inference/rendering cards
    "A10", "A10G", "A40", "A4500",
    # AD102, datacenter inference (no NVLink bridge)
    "L40", "L40S",
    # Consumer cards, no NVLink in cloud configurations
    "RTX4090", "RTX3090", "RTX3090Ti",
    # Source: pipeline/sources/rtx5000ada_specs.md — "NVLink: No"
    "RTX5000Ada",
    # Source: pipeline/sources/rtx6000ada_specs.md — "NVLink: No"
    "RTX6000Ada",
    # Source: pipeline/sources/rtxpro6000_specs.md — "NVLink: No" (PCIe 5.0 only)
    "RTX5090", "RTXPRO6000", "RTXPRO6000MaxQ", "RTXPRO6000WK", "RTXPRO4500",
}

# Architectures that support FP8 compute (2× multiplier over FP16)
_FP8_COMPUTE_ARCHITECTURES = {"Hopper", "Blackwell"}

# Architectures that support FP8 KV cache
_FP8_KV_CACHE_ARCHITECTURES = {"Ada Lovelace", "Hopper", "Blackwell"}


def fetch_gpu_specs() -> list[dict]:
    """Fetch GPU specs from dbgpu for all GPUs in GPUHUNT_TO_DBGPU_KEY.

    Returns a list of spec dicts (one per GPU). Does NOT include GH200
    (handled separately in the exporter).

    Raises KeyError if any GPU is not found in dbgpu — no silent fallbacks.
    """
    db = GPUDatabase.default()
    specs_map = db.specifications

    results: list[dict] = []

    for gpu_name, dbgpu_key in GPUHUNT_TO_DBGPU_KEY.items():
        if dbgpu_key not in specs_map:
            raise KeyError(
                f"GPU '{gpu_name}' not found in dbgpu (key='{dbgpu_key}'). "
                f"Update GPUHUNT_TO_DBGPU_KEY or upgrade dbgpu."
            )

        gpu = specs_map[dbgpu_key]

        # Memory size (GB) — used for VRAM capacity in the frontend
        mem_gb = gpu.memory_size_gb or 0

        # Memory bandwidth (GB/s → TB/s)
        bw_gb_s = gpu.memory_bandwidth_gb_s or 0
        bw_tb_s = bw_gb_s / 1000

        # FP16 TFLOPS (GFLOP/s → TFLOPS)
        fp16_gflops = gpu.half_float_performance_gflop_s or 0
        fp16_tflops = fp16_gflops / 1000

        # Chip name and architecture
        chip = gpu.gpu_name
        arch = gpu.architecture

        # Apply multi-die correction (B200/B300 are dual-die)
        die_count = MULTI_DIE_CHIPS.get(chip, 1)
        if die_count > 1:
            mem_gb *= die_count
            bw_tb_s *= die_count
            fp16_tflops *= die_count

        # Normalize architecture name
        arch = ARCH_NORMALIZATION.get(arch, arch)

        # FP8 multiplier: Hopper/Blackwell get 2×, older architectures get 1×
        fp8_multiplier = 2 if arch in _FP8_COMPUTE_ARCHITECTURES else 1

        # NVLink bandwidth
        nvlink: float | None = None
        if gpu_name not in NO_NVLINK and chip in CHIP_NVLINK_BANDWIDTH:
            nvlink = CHIP_NVLINK_BANDWIDTH[chip]

        results.append({
            "gpu_name": gpu_name,
            "memory_size_gb": round(mem_gb, 1),
            "fp16_tflops": round(fp16_tflops, 1),
            "memory_bandwidth_tb_s": round(bw_tb_s, 3),
            "nvlink_bandwidth_gb_s": nvlink,
            "fp8_multiplier": fp8_multiplier,
            "architecture": arch,
        })

        logger.debug(
            "  %s: bw=%.3f TB/s, fp16=%.1f TFLOPS, nvlink=%s, arch=%s",
            gpu_name, bw_tb_s, fp16_tflops, nvlink, arch,
        )

    logger.info("Fetched specs for %d GPUs from dbgpu", len(results))
    return results
