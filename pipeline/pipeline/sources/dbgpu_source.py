"""GPU hardware specs sourced from dbgpu (TechPowerUp database).

Provides traceable GPU specifications for LLM inference throughput calculations.
Every value cites its origin: dbgpu (TechPowerUp), Wikipedia NVLink article,
or NVIDIA datasheets stored in pipeline/sources/.
"""

import logging
import re

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
    # A100 (bare, 40 GB) is the SXM/DGX form factor — its interconnect tier is
    # nvswitch, so map it to the SXM die for consistency. (PCIe and SXM 40 GB
    # share the same 1.56 TB/s bandwidth, but the slug must match the tier.)
    "A100": "a100-sxm4-40gb",
    "A100_80G": "a100-sxm4-80gb",
    "A40": "a40-pcie",
    "A4000": "rtx-a4000",
    "A4500": "rtx-a4500",
    "A5000": "rtx-a5000",
    "A6000": "rtx-a6000",
    "B200": "b200",
    "B300": "b300",
    "H100": "h100-sxm5-80gb",
    # H100 PCIe is a distinct SKU: HBM2e @ 2.04 TB/s (vs SXM HBM3 @ 3.36).
    # gpuhunt reports it as gpu_name "H100" with "PCIe" in the instance_name;
    # gpuhunt_source remaps those offerings to this key (see PCIE_VARIANT_GPUS).
    "H100_PCIe": "h100-pcie-80gb",
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

# ---------------------------------------------------------------------------
# Inter-GPU interconnect tier, per GPU. Authoritative source for the frontend
# topology model (calc-topology.ts): every GPU resolves to exactly one tier —
# we never emit a null/absent tier.
#
#   nvswitch       — SXM/HGX/DGX baseboard, full all-to-all NVLink fabric → TP
#                    can span the node.
#   nvlink_paired  — NVLink-bridge cards connected only in 2-way pairs → TP
#                    capped at the pair, PP across pairs.
#   none           — PCIe-only (no NVLink connector) → TP all-reduce traverses
#                    PCIe; flagged for the latency penalty.
#
# Sources: NVIDIA datasheets / TechPowerUp form factor (mirrors NO_NVLINK below
# for the "none" set; the SXM-vs-paired split follows the board's NVLink fabric).
# ---------------------------------------------------------------------------
INTERCONNECT_TIER: dict[str, str] = {
    # SXM / NVSwitch datacenter GPUs (HGX/DGX baseboards).
    "A100": "nvswitch",
    "A100_80G": "nvswitch",
    "V100": "nvswitch",
    "H100": "nvswitch",
    "H200": "nvswitch",
    "B200": "nvswitch",
    "B300": "nvswitch",
    # NVLink-bridge cards (2-way paired) and H100 PCIe (bridged in pairs).
    "A5000": "nvlink_paired",
    "A6000": "nvlink_paired",
    "RTX6000": "nvlink_paired",
    "H100NVL": "nvlink_paired",
    "H100_PCIe": "nvlink_paired",
    # PCIe-only (no NVLink connector) — must include all of NO_NVLINK plus the
    # NVLink-capable chips with no board connector (A4000, L4, RTX4000Ada).
    "A10": "none",
    "A10G": "none",
    "A40": "none",
    "A4000": "none",
    "A4500": "none",
    "L4": "none",
    "L40": "none",
    "L40S": "none",
    "RTX3090": "none",
    "RTX3090Ti": "none",
    "RTX4000Ada": "none",
    "RTX4090": "none",
    "RTX5000Ada": "none",
    "RTX5090": "none",
    "RTX6000Ada": "none",
    "RTXPRO4500": "none",
    "RTXPRO6000": "none",
    "RTXPRO6000MaxQ": "none",
    "RTXPRO6000WK": "none",
}

_VALID_TIERS = {"none", "nvlink_paired", "nvswitch"}


def _pcie_bandwidth_gb_s(bus_interface: str | None) -> float | None:
    """Per-direction PCIe x16 bandwidth (GB/s) from a dbgpu bus_interface string.

    Public PCI-SIG spec, ×16 link: Gen3 ≈ 16, Gen4 ≈ 32, Gen5 ≈ 64 GB/s.
    Returns None if the generation cannot be parsed.
    """
    if not bus_interface:
        return None
    m = re.search(r"pcie\s*([0-9]+)", bus_interface, re.IGNORECASE)
    if not m:
        return None
    return {3: 16.0, 4: 32.0, 5: 64.0}.get(int(m.group(1)))


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
        if gpu_name not in INTERCONNECT_TIER:
            raise KeyError(
                f"GPU '{gpu_name}' has no interconnect_tier. Add it to "
                f"INTERCONNECT_TIER (every GPU must resolve to a tier)."
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
            "pcie_bandwidth_gb_s": _pcie_bandwidth_gb_s(getattr(gpu, "bus_interface", None)),
            "nvlink_bandwidth_gb_s": nvlink,
            "fp8_multiplier": fp8_multiplier,
            "architecture": arch,
            "interconnect_tier": INTERCONNECT_TIER[gpu_name],
            "memory_type": getattr(gpu, "memory_type", None),
        })

        logger.debug(
            "  %s: bw=%.3f TB/s, fp16=%.1f TFLOPS, nvlink=%s, tier=%s, arch=%s",
            gpu_name, bw_tb_s, fp16_tflops, nvlink, INTERCONNECT_TIER[gpu_name], arch,
        )

    logger.info("Fetched specs for %d GPUs from dbgpu", len(results))
    return results
