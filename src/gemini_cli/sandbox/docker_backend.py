"""Optional Docker sandbox backend.

The backend uses the Docker CLI rather than a Python SDK, so it adds no
dependency and works with whatever Docker installation the user already has.
Containers are ephemeral, run without network access by default, drop Linux
capabilities, and mount the allowed project root read-only by default.

Docker is optional. Call :func:`gemini_cli.sandbox.select_backend` to fall back
to the subprocess backend when the Docker daemon is not available.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4

from gemini_cli.config import Settings
from gemini_cli.sandbox._process import (
    await_reader,
    build_docker_environment,
    containing_root,
    kill_process_group,
    read_limited,
    resolve_working_directory,
    wait_after_kill,
)
from gemini_cli.sandbox.base import ExecutionResult, SandboxError

_PROBE_TIMEOUT_SECONDS = 5.0
_CLEANUP_TIMEOUT_SECONDS = 5.0
_DEFAULT_IMAGE = "python:3.12-slim"
_CONTAINER_TMPFS = "/tmp:rw,noexec,nosuid,size=64m"


class DockerBackend:
    """Run commands in an ephemeral Docker container."""

    def __init__(
        self,
        settings: Settings,
        *,
        image: str = _DEFAULT_IMAGE,
        network_enabled: bool = False,
        read_only_mount: bool = True,
        memory_limit: str = "512m",
        cpu_limit: str = "1.0",
        pids_limit: int = 256,
        docker_executable: str = "docker",
    ) -> None:
        self._settings = settings
        self._image = image
        self._network_enabled = network_enabled
        self._read_only_mount = read_only_mount
        self._memory_limit = memory_limit
        self._cpu_limit = cpu_limit
        self._pids_limit = pids_limit
        self._docker_executable = docker_executable
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Return whether the Docker CLI and daemon are usable."""
        if self._available is not None:
            return self._available

        executable = shutil.which(self._docker_executable)
        if executable is None:
            self._available = False
            return False

        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                "version",
                "--format",
                "{{.Server.Version}}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=build_docker_environment(),
                start_new_session=os.name == "posix",
            )
        except OSError:
            self._available = False
            return False

        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            kill_process_group(process)
            await wait_after_kill(process)
            self._available = False
            return False

        self._available = process.returncode == 0
        return self._available

    async def execute(
        self,
        command: str,
        *,
        working_directory: str | None = None,
        timeout_seconds: int = 30,
        max_output_bytes: int = 524_288,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute *command* in an ephemeral container."""
        if not command.strip():
            msg = "Command must not be blank."
            raise SandboxError(msg)
        if timeout_seconds <= 0:
            msg = "timeout_seconds must be positive."
            raise SandboxError(msg)
        if max_output_bytes <= 0:
            msg = "max_output_bytes must be positive."
            raise SandboxError(msg)
        if not await self.is_available():
            msg = "Docker sandbox is not available."
            raise SandboxError(msg)

        executable = shutil.which(self._docker_executable)
        if executable is None:
            msg = "Docker sandbox is not available."
            raise SandboxError(msg)

        directory = resolve_working_directory(
            working_directory,
            self._settings.allowed_roots,
        )
        root = containing_root(directory, self._settings.allowed_roots)
        relative = directory.relative_to(root)
        relative_posix = relative.as_posix()
        container_workdir = (
            "/workspace" if relative_posix == "." else f"/workspace/{relative_posix}"
        )
        container_name = f"gemini-cli-sandbox-{uuid4().hex[:12]}"
        arguments = self._build_arguments(
            container_name=container_name,
            root=root,
            container_workdir=container_workdir,
            command=command,
            env=env,
        )

        started = time.perf_counter()
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=build_docker_environment(),
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            msg = f"Could not start Docker: {exc}"
            raise SandboxError(msg) from exc

        if process.stdout is None or process.stderr is None:
            msg = "Docker process did not expose stdout and stderr pipes."
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
            await self._cleanup_container(container_name)
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

    def _build_arguments(
        self,
        *,
        container_name: str,
        root: Path,
        container_workdir: str,
        command: str,
        env: dict[str, str] | None,
    ) -> list[str]:
        """Build the ``docker run`` argument list."""
        mount_mode = "ro" if self._read_only_mount else "rw"
        arguments = [
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "bridge" if self._network_enabled else "none",
            "--memory",
            self._memory_limit,
            "--cpus",
            self._cpu_limit,
            "--pids-limit",
            str(self._pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            _CONTAINER_TMPFS,
            "--pull=never",
            "-v",
            f"{root}:/workspace:{mount_mode}",
            "-w",
            container_workdir,
        ]

        if os.name == "posix":
            arguments.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

        for key, value in sorted((env or {}).items()):
            arguments.extend(["-e", f"{key}={value}"])

        arguments.extend([self._image, "sh", "-c", command])
        return arguments

    async def _cleanup_container(self, container_name: str) -> None:
        """Force-remove a container left behind by a timeout."""
        await self._run_docker_quiet("kill", container_name)
        await self._run_docker_quiet("rm", "-f", container_name)

    async def _run_docker_quiet(self, *arguments: str) -> None:
        executable = shutil.which(self._docker_executable)
        if executable is None:
            return
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                *arguments,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=build_docker_environment(),
                start_new_session=os.name == "posix",
            )
        except OSError:
            return
        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=_CLEANUP_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            kill_process_group(process)
            await wait_after_kill(process)


__all__ = ["DockerBackend"]
