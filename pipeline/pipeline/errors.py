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


class UnsupportedArchitecture(Exception):
    """Raised when a model's architecture is not yet supported.

    This is non-fatal: the pipeline should skip the model, alert, and continue.
    """

    def __init__(self, model_name: str, model_type: str, hf_id: str) -> None:
        self.model_name = model_name
        self.model_type = model_type
        self.hf_id = hf_id
        super().__init__(
            f"Unsupported architecture model_type='{model_type}' "
            f"for {model_name} ({hf_id})"
        )
