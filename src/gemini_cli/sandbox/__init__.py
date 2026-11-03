"""Sandbox backends and approval flow — public surface."""

from __future__ import annotations

from gemini_cli.sandbox.base import ExecutionResult, SandboxBackend, SandboxError
from gemini_cli.sandbox.subprocess_backend import SubprocessBackend

__all__ = [
    "ExecutionResult",
    "SandboxBackend",
    "SandboxError",
    "SubprocessBackend",
]
