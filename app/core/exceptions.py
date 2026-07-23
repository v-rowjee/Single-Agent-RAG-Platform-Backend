"""Shared domain exceptions and stable HTTP error mappings."""


class SessionNotFoundError(Exception):
    """Raised when an analysis session does not exist or is not accessible."""


class InvalidUploadError(Exception):
    """Raised when uploaded dataset content cannot be accepted."""


class DatasetAlreadyExistsError(Exception):
    """Raised when a user attempts to create a second active workspace."""


class DataPreparationError(RuntimeError):
    """Raised when a dataset cannot be cleaned or prepared."""
