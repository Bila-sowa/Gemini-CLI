"""Internal helpers shared by sandbox backends.

These helpers keep the security-sensitive details — path confinement,
environment allow-listing, bounded pipe reads, and process-group termination —
in one place so the subprocess and Docker backends behave consistently.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Mapping, Sequence
from pathlib import Path

from gemini_cli.sandbox.base import SandboxError
from gemini_cli.tools.filesystem.path_validator import (
    PathSecurityError,
    validate_path,
)

_READ_CHUNK_BYTES = 64 * 1024
_READER_SHUTDOWN_SECONDS = 5.0
_KILL_WAIT_SECONDS = 5.0

# Only variables needed to find executables and behave predictably are passed
# to a child process. In particular, API keys and other application secrets in
# the parent environment are never inherited.
_POSIX_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "TZ",
)
_WINDOWS_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
)

# The Docker CLI is a trusted helper, not the command being sandboxed. It needs
# its own connection and configuration variables to find the daemon, but these
# values are never forwarded into the container.
_DOCKER_ENV_ALLOWLIST: tuple[str, ...] = (
    "HOME",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "XDG_RUNTIME_DIR",
)


def resolve_working_directory(
    working_directory: str | None,
    allowed_roots: Sequence[Path],
) -> Path:
    """Return a validated working directory inside *allowed_roots*.

    When *working_directory* is omitted, the first allowed root is used. An
    empty root list disables shell execution entirely rather than falling back
    to the current process directory.
    """
    if not allowed_roots:
        msg = "No allowed roots are configured; shell execution is disabled."
        raise SandboxError(msg)

    raw = working_directory if working_directory is not None else str(allowed_roots[0])
    try:
        directory = validate_path(raw, allowed_roots, must_exist=True)
    except PathSecurityError as exc:
        raise SandboxError(str(exc)) from exc

    if not directory.is_dir():
        msg = f"Working directory is not a directory: {directory}."
        raise SandboxError(msg)
    return directory


def containing_root(path: Path, allowed_roots: Sequence[Path]) -> Path:
    """Return the allowed root that contains *path*."""
    resolved_path = path.expanduser().resolve()
    for root in allowed_roots:
        resolved_root = root.expanduser().resolve()
        if resolved_path.is_relative_to(resolved_root):
            return resolved_root
    msg = f"Path {str(path)!r} is outside the allowed roots."
    raise SandboxError(msg)


def build_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build a minimal child environment without application secrets."""
    names = _WINDOWS_ENV_ALLOWLIST if os.name == "nt" else _POSIX_ENV_ALLOWLIST
    environment = {name: os.environ[name] for name in names if name in os.environ}
    if "PATH" not in environment:
        environment["PATH"] = os.defpath
    if extra:
        environment.update({str(key): str(value) for key, value in extra.items()})
    return environment


def build_docker_environment() -> dict[str, str]:
    """Build the environment for the trusted Docker CLI.

    Docker needs its daemon connection settings, but those variables must not
    leak into the command running inside the container.
    """
    environment = build_environment()
    for name in _DOCKER_ENV_ALLOWLIST:
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


async def read_limited(
    stream: asyncio.StreamReader,
    limit: int,
) -> tuple[bytes, bool]:
    """Read *stream* while keeping at most *limit* bytes.

    The rest of the stream is still drained. Stopping at the limit would let
    the child process block forever once the pipe buffer fills.
    """
    if limit <= 0:
        msg = "max_output_bytes must be positive."
        raise SandboxError(msg)

    chunks: list[bytes] = []
    size = 0
    truncated = False

    while True:
        chunk = await stream.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        if size < limit:
            remaining = limit - size
            take = chunk[:remaining]
            chunks.append(take)
            size += len(take)
            if len(chunk) > remaining:
                truncated = True
        else:
            truncated = True

    return b"".join(chunks), truncated


async def await_reader(
    task: asyncio.Task[tuple[bytes, bool]],
) -> tuple[bytes, bool]:
    """Wait for a bounded reader, giving up if the pipe never closes."""
    try:
        return await asyncio.wait_for(task, timeout=_READER_SHUTDOWN_SECONDS)
    except TimeoutError:
        task.cancel()
        return b"", True


def kill_process_group(process: asyncio.subprocess.Process) -> None:
    """Terminate the process and, on POSIX, every process in its group."""
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError):
        # The process exited between the check and the signal, or the platform
        # does not allow signalling it. Either way there is nothing to clean up.
        return


async def wait_after_kill(process: asyncio.subprocess.Process) -> None:
    """Wait briefly for a killed process so a failed kill cannot hang us."""
    try:
        await asyncio.wait_for(process.wait(), timeout=_KILL_WAIT_SECONDS)
    except TimeoutError:
        return


__all__ = [
    "await_reader",
    "build_docker_environment",
    "build_environment",
    "containing_root",
    "kill_process_group",
    "read_limited",
    "resolve_working_directory",
    "wait_after_kill",
]
