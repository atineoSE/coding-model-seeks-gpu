"""Constants for snapshot generation."""

from datetime import date

# First date where the OpenHands Index has stable methodology:
# all 5 categories present, every model evaluated across all of them,
# and swe-bench-multimodal using the solveable_accuracy metric.
SCHEMA_CONSOLIDATION_DATE = date(2026, 1, 28)

# Map repo benchmark names → (benchmark_name, display_name)
BENCHMARK_MAP: dict[str, tuple[str, str]] = {
    "swe-bench": ("issue_resolution", "Issue Resolution"),
    "swe-bench-multimodal": ("frontend", "Frontend"),
    "commit0": ("greenfield", "Greenfield"),
    "swt-bench": ("testing", "Testing"),
    "gaia": ("information_gathering", "Information Gathering"),
}

BENCHMARK_GROUP = "openhands"
BENCHMARK_GROUP_DISPLAY = "OpenHands Index"

OVERALL_NAME = "overall"
OVERALL_DISPLAY = "Overall"

# Reverse lookup: mapped benchmark_name → display_name
DISPLAY_NAMES: dict[str, str] = {name: display for name, display in BENCHMARK_MAP.values()}
DISPLAY_NAMES[OVERALL_NAME] = OVERALL_DISPLAY
