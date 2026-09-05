"""Replay assistant turns recorded by an earlier run."""

from __future__ import annotations


class JournalBackend:
    """Replay assistant turns recorded by an earlier run."""

    def __init__(self, exchanges: list[dict]) -> None:
        self._left = list(exchanges)

    def send(self, request: dict) -> dict:
        """Return the next recorded turn, ignoring `request`.

        Parameters
        ----------
        request : dict
            The request the live backend would have been sent.

        Returns
        -------
        dict
            The recorded assistant turn.

        Raises
        ------
        DriveError
            When the recording holds no further turn.
        """
        del request
        # why: a local import, not a module-level one -- drive.py imports this module for the
        # class itself, and a top-level import back would make the two modules load each other
        from benchmark.harness.drive import DriveError

        if not self._left:
            raise DriveError("no recorded turn is left to replay")
        return self._left.pop(0)
