class InvalidPosition(ValueError):
    """Malformed or inconsistent chess position."""


class RecognitionError(ValueError):
    """Image cannot be decoded or does not yield a usable chessboard."""


class DependencyError(RuntimeError):
    """Model files, optional packages, or engine are unavailable."""
