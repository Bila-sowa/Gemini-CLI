"""End-to-end acceptance checks for Phase 6."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

import pytest

from gemini_cli.config import Settings
from gemini_cli.sandbox.approval import ApprovalDecision, ApprovalManager
from gemini_cli.sandbox.docker_backend import DockerBackend
from gemini_cli.sandbox.subprocess_backend import SubprocessBackend
from gemini_cli.tools.base import RiskLevel
from gemini_cli.tools.shell import RunCommandParams, RunCommandTool


def python_command(code: str) -> str:
    """Build a shell command that runs *code* with the current interpreter."""
    if os.name == "nt":
        return subprocess.list2cmdline([sys.executable, "-c", code])
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


class ApprovingCallback:
    """Callback that approves every destructive action."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def request_approval(
        self,
        action_description: str,
        risk_level: str,
        details: str,
    ) -> ApprovalDecision:
        self.calls.append((action_description, risk_level, details))
        return ApprovalDecision.APPROVED


async def test_subprocess_backend_executes_echo(settings: Settings) -> None:
    result = await SubprocessBackend(settings).execute("echo hello")

    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"
    assert result.timed_out is False


async def test_subprocess_backend_times_out(settings: Settings) -> None:
    result = await SubprocessBackend(settings).execute(
        python_command("import time; time.sleep(5)"),
        timeout_seconds=1,
    )

    assert result.timed_out is True


async def test_subprocess_backend_truncates_output(settings: Settings) -> None:
    result = await SubprocessBackend(settings).execute(
        python_command("print('x' * 500)"),
        max_output_bytes=50,
    )

    assert result.stdout_truncated is True
    assert len(result.stdout) <= 50


async def test_approval_auto_approves_read_and_asks_for_destructive() -> None:
    callback = ApprovingCallback()
    manager = ApprovalManager(callback)

    read_decision = await manager.check("read_file", RiskLevel.READ, {})
    destructive_decision = await manager.check(
        "run_command",
        RiskLevel.DESTRUCTIVE,
        {"command": "echo hello"},
    )

    assert read_decision is ApprovalDecision.AUTO_APPROVED
    assert destructive_decision is ApprovalDecision.APPROVED
    assert callback.calls[0][1] == "destructive"
    assert "echo hello" in callback.calls[0][2]


async def test_docker_backend_is_unavailable_without_cli(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gemini_cli.sandbox.docker_backend.shutil.which",
        lambda _name: None,
    )

    assert await DockerBackend(settings).is_available() is False


async def test_run_command_uses_real_subprocess_backend(settings: Settings) -> None:
    tool = RunCommandTool(settings, SubprocessBackend(settings))

    result = await tool.execute(RunCommandParams(command="echo hello"))

    assert result.success is True
    assert result.output.strip() == "hello"
    assert result.error is None
    assert result.metadata["exit_code"] == 0
