"""Single source of truth for reducing GPU offerings to curated node prices.

A "node" is the 8-GPU datacenter serving tier for a curated GPU. For each
node present on a given day we keep the single cheapest ``gpu_count == 8``
offering, recording its price as ``usd_per_node_hour``. Provider is
deliberately NOT recorded: the history is a price record, not a provider
directory, and we don't want to single out any supplier.
"""

# The curated 8-GPU serving tier, in display order.
SERVING_NODES = ["B300", "B200", "H200", "H100", "A100", "RTXPRO6000"]


def reduce_offerings_to_nodes(offerings: list[dict]) -> list[dict]:
    """Reduce raw GPU offerings to one cheapest 8x node price per curated GPU.

    Rules (baked in here and ONLY here):
      * only ``gpu_count == 8`` offerings count as a node;
      * for ``A100`` keep only the 80GB variant (``vram_gb == 80``);
      * per ``gpu_name`` pick the offering with the minimum ``price_per_hour``.

    Returns a list of ``{"gpu_name", "usd_per_node_hour"}`` dicts, sorted by
    ``SERVING_NODES`` order, omitting nodes with no 8x offering. Robust to
    older-schema dicts that may lack keys other than
    ``gpu_name``/``gpu_count``/``vram_gb``/``price_per_hour``.
    """
    cheapest: dict[str, float] = {}

    for offering in offerings:
        gpu_name = offering.get("gpu_name")
        if gpu_name not in SERVING_NODES:
            continue
        if offering.get("gpu_count") != 8:
            continue
        if gpu_name == "A100" and offering.get("vram_gb") != 80:
            continue

        price = offering.get("price_per_hour")
        if price is None:
            continue

        current = cheapest.get(gpu_name)
        if current is None or price < current:
            cheapest[gpu_name] = price

    return [
        {"gpu_name": name, "usd_per_node_hour": cheapest[name]}
        for name in SERVING_NODES
        if name in cheapest
    ]
