"""Sandbox backend protocol and execution result.

A sandbox backend is the only component allowed to start a shell command. The
agent and tools depend on this small protocol instead of the concrete
subprocess or Docker implementation, so a stronger isolation backend can be
selected without changing tool code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SandboxError(RuntimeError):
    """Raised when a command cannot be executed safely.

    Expected configuration and validation failures use this exception so the
    calling tool can turn them into a structured ``ToolResult``. The message
    must never contain the command itself, because a command can carry secrets.
    """


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one sandboxed command.

    ``stdout`` and ``stderr`` are already decoded as UTF-8 with replacement for
    invalid bytes. ``stdout_truncated`` and ``stderr_truncated`` report whether
    the corresponding stream exceeded the requested output limit.
    """

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class SandboxBackend(Protocol):
    """Execution boundary used by shell-capable tools."""

    async def execute(
        self,
        command: str,
        *,
        working_directory: str | None = None,
        timeout_seconds: int = 30,
        max_output_bytes: int = 524_288,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Run *command* and return its bounded result.

        Implementations must validate the working directory against the
        configured allowed roots, enforce the timeout, cap captured output, and
        avoid inheriting arbitrary host environment variables.
        """

    async def is_available(self) -> bool:
        """Return whether this backend can run commands on this host."""


__all__ = ["ExecutionResult", "SandboxBackend", "SandboxError"]
