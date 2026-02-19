"""Shared exception types for the pipeline."""


class FormatBreakingChange(Exception):
    """Raised when an upstream data source has changed its format.

    Carries *source* (e.g. "gpuhunt", "openhands-index") and a human-readable
    *details* string so the notification email can explain what broke.
    """

    def __init__(self, source: str, details: str) -> None:
        self.source = source
        self.details = details
        super().__init__(f"Breaking format change in {source}: {details}")
