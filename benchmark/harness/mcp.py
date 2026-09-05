"""Start, query and stop the MCP server over stdio."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from benchmark.harness import execute

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


class ToolError(Exception):
    """Raised when a tool reply reports an error."""


def _try_listing(ctx: execute.Context) -> tuple[dict[str, tuple[int, int]] | None, NotADirectoryError | None]:
    try:
        return execute.watched_listing(ctx), None
    except NotADirectoryError as exc:
        return None, exc


def _changed_or_empty(
    ctx: execute.Context, before: dict[str, tuple[int, int]] | None, after: dict[str, tuple[int, int]] | None,
) -> list[dict]:
    # why: a listing that failed on either side leaves no pair to diff; the failure itself
    # is what the caller records as the entry's `error`, not a files list
    if before is None or after is None:
        return []
    return execute.changed(ctx, before, after)


class Server:
    """One MCP server per run, started over stdio and stopped by the harness."""

    def __init__(self, ctx: execute.Context, launcher: Path, tools: list[str]) -> None:
        self.ctx = ctx
        self.launcher = launcher
        self.tools = tools
        self._client: Client | None = None
        self._loop = asyncio.new_event_loop()
        # inv: stdio only; --http would open a port the audit cannot see
        self.argv = [str(launcher), "serve", "--repo", str(ctx.sandbox), "--tools", ",".join(tools)]

    def __enter__(self) -> Server:
        before, before_exc = _try_listing(self.ctx)
        env = {**self.ctx.environment, "HOME": str(self.ctx.home)}
        # why: default keep_alive=True leaves the subprocess running past __aexit__ for reuse;
        # one Server opens exactly one connection, so the process must die when this one closes
        transport = StdioTransport(
            command=self.argv[0], args=self.argv[1:], env=env, cwd=str(self.ctx.sandbox), keep_alive=False,
        )
        # inv: the handshake is bounded by the same timeout as a tool call, or a vendor that
        # starts and never speaks holds this thread and the sandbox lock with no bound at all
        self._client = Client(transport, init_timeout=execute.VENDOR_TIMEOUT_S)
        entry: dict = {
            "n": execute.next_n(self.ctx), "kind": "server", "event": "start", "by": "harness", "argv": self.argv,
        }
        handshake_exc: Exception | None = None
        try:
            self._loop.run_until_complete(self._client.__aenter__())
        except Exception as exc:
            handshake_exc = exc
        after, after_exc = _try_listing(self.ctx)
        # why: this vendor writes .gitignore and the graph.db-wal/-shm pair inside a tool call,
        # so both lifecycle diffs come back empty; a non-empty one means its write timing moved
        entry["files"] = _changed_or_empty(self.ctx, before, after)
        failures = [e for e in (handshake_exc, before_exc, after_exc) if e is not None]
        if failures:
            entry["error"] = "; ".join(str(e) for e in failures)
        # why: an attempt that reached the vendor must land in the journal, whether or not it
        # touched the filesystem, even when the handshake itself fails
        execute.append(self.ctx, entry)
        if handshake_exc is not None:
            self._loop.close()
            raise handshake_exc
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None,
    ) -> None:
        # why: a listing that raises here (a sandbox or home root clobbered into a file) must
        # not skip the shutdown attempt or the stop line below -- both still have to happen
        before, before_exc = _try_listing(self.ctx)
        shutdown_exc: Exception | None = None
        try:
            if self._client is not None:
                self._loop.run_until_complete(self._client.__aexit__(None, None, None))
        except Exception as caught:
            shutdown_exc = caught
        finally:
            self._loop.close()
        after, after_exc = _try_listing(self.ctx)
        entry: dict = {
            "n": execute.next_n(self.ctx), "kind": "server", "event": "stop", "by": "harness",
            "files": _changed_or_empty(self.ctx, before, after),
        }
        failures = [e for e in (shutdown_exc, before_exc, after_exc) if e is not None]
        if failures:
            entry["error"] = "; ".join(str(e) for e in failures)
        execute.append(self.ctx, entry)
        # why: a cleanup failure must never displace the body's own exception; it only surfaces
        # when the run would otherwise look like it finished normally
        if failures and exc_type is None:
            raise failures[0]

    def _client_or_raise(self) -> Client:
        if self._client is None:
            raise RuntimeError("server not started")
        return self._client

    def list_tools(self) -> list[dict]:
        """Return the server's tool definitions as plain dicts."""
        tools = self._loop.run_until_complete(self._client_or_raise().list_tools())
        return [{"name": t.name, "description": t.description or "", "inputSchema": t.inputSchema} for t in tools]

    def call(self, tool: str, args: dict) -> str:
        """Call `tool` with `args` plus the sandbox as `repo_root`; return the reply text.

        Parameters
        ----------
        tool : str
            Name of the tool to call.
        args : dict
            Tool arguments; `repo_root` is overwritten with the sandbox path.

        Returns
        -------
        str
            The joined `content[].text` of the reply.

        Raises
        ------
        ToolError
            If the reply reports an error.
        McpError
            If no reply arrives within `execute.VENDOR_TIMEOUT_S` seconds.
        """
        payload = {**args, "repo_root": str(self.ctx.sandbox)}
        # inv: this timeout is the only bound on a tool call, since execute.py's bounds
        # subprocess.run and a tool call spawns no child of its own
        result = self._loop.run_until_complete(self._client_or_raise().call_tool(
            tool, payload, raise_on_error=False, timeout=execute.VENDOR_TIMEOUT_S))
        text = "".join(getattr(c, "text", "") for c in result.content)
        if result.is_error:
            raise ToolError(text)
        return text
