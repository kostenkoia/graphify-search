"""Typed errors raised by the skill writer, the atomic write, and the search stack."""


class InputError(Exception):
    """Recoverable bad-input condition surfaced to the caller as a JSON result.

    Parameters
    ----------
    message : str
        Human-readable description of what was wrong with the input.
    hint : str, optional
        Short actionable suggestion for the caller; empty when none applies.
    """

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class EndpointUnavailableError(Exception):
    """The embeddings endpoint did not answer.

    Parameters
    ----------
    message : str
        What failed, with the URL.
    hint : str, optional
        Short actionable suggestion; empty when none applies.
    """

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint
