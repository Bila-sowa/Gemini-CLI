"""Subprocess sandbox backend.

This backend is a lightweight local execution boundary: it confines the
working directory to an allowed root, strips application secrets from the
environment, enforces a timeout, and caps captured output. It is *not* a
kernel-level sandbox — a command can still reach the network or read files
outside the root. The approval layer and, where available, the Docker backend
provide the stronger controls.

Use :class:`~gemini_cli.sandbox.docker_backend.DockerBackend` when stronger
filesystem and network isolation is required.
"""

from __future__ import annotations

import asyncio
import os
import time

from gemini_cli.config import Settings
from gemini_cli.sandbox._process import (
    await_reader,
    build_environment,
    kill_process_group,
    read_limited,
    resolve_working_directory,
    wait_after_kill,
)
from gemini_cli.sandbox.base import ExecutionResult, SandboxError


class SubprocessBackend:
    """Run commands through ``asyncio.create_subprocess_shell``."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def execute(
        self,
        command: str,
        *,
        working_directory: str | None = None,
        timeout_seconds: int = 30,
        max_output_bytes: int = 524_288,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute *command* with bounded time, output, and environment."""
        if not command.strip():
            msg = "Command must not be blank."
            raise SandboxError(msg)
        if timeout_seconds <= 0:
            msg = "timeout_seconds must be positive."
            raise SandboxError(msg)
        if max_output_bytes <= 0:
            msg = "max_output_bytes must be positive."
            raise SandboxError(msg)

        directory = resolve_working_directory(
            working_directory,
            self._settings.allowed_roots,
        )
        environment = build_environment(env)

        started = time.perf_counter()
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(directory),
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            msg = f"Could not start command: {exc}"
            raise SandboxError(msg) from exc

        if process.stdout is None or process.stderr is None:
            msg = "Sandbox process did not expose stdout and stderr pipes."
            raise SandboxError(msg)

        stdout_task = asyncio.create_task(
            read_limited(process.stdout, max_output_bytes)
        )
        stderr_task = asyncio.create_task(
            read_limited(process.stderr, max_output_bytes)
        )

        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
        except TimeoutError:
            timed_out = True
            kill_process_group(process)
            await wait_after_kill(process)

        stdout_bytes, stdout_truncated = await await_reader(stdout_task)
        stderr_bytes, stderr_truncated = await await_reader(stderr_task)
        duration = time.perf_counter() - started

        return ExecutionResult(
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            timed_out=timed_out,
            duration_seconds=duration,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    async def is_available(self) -> bool:
        """Subprocess execution is available wherever asyncio is."""
        return True


__all__ = ["SubprocessBackend"]
