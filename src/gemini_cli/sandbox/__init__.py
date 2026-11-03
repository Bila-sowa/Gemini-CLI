"""Sandbox backends and approval flow — public surface."""

from __future__ import annotations

from gemini_cli.config import Settings
from gemini_cli.sandbox.base import ExecutionResult, SandboxBackend, SandboxError
from gemini_cli.sandbox.docker_backend import DockerBackend
from gemini_cli.sandbox.subprocess_backend import SubprocessBackend


async def select_backend(
    settings: Settings,
    *,
    prefer_docker: bool = False,
    docker: DockerBackend | None = None,
) -> SandboxBackend:
    """Return Docker when requested and available, otherwise subprocess.

    The subprocess backend is always available, so this function never fails
    just because Docker is missing.
    """
    if prefer_docker:
        candidate = docker if docker is not None else DockerBackend(settings)
        if await candidate.is_available():
            return candidate
    return SubprocessBackend(settings)


__all__ = [
    "DockerBackend",
    "ExecutionResult",
    "SandboxBackend",
    "SandboxError",
    "SubprocessBackend",
    "select_backend",
]
