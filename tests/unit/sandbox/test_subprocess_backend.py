"""Tests for the subprocess sandbox backend."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from gemini_cli.config import Settings
from gemini_cli.sandbox.base import SandboxError
from gemini_cli.sandbox.subprocess_backend import SubprocessBackend


def python_command(code: str) -> str:
    """Build a shell command that runs *code* with the current interpreter."""
    if os.name == "nt":
        return subprocess.list2cmdline([sys.executable, "-c", code])
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


async def test_executes_command(settings: Settings) -> None:
    result = await SubprocessBackend(settings).execute(python_command("print('hello')"))

    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"
    assert result.stderr == ""
    assert result.timed_out is False
    assert result.duration_seconds >= 0
    assert result.stdout_truncated is False
    assert result.stderr_truncated is False


async def test_captures_stderr(settings: Settings) -> None:
    result = await SubprocessBackend(settings).execute(
        python_command("import sys; sys.stderr.write('oops')")
    )

    assert result.exit_code == 0
    assert result.stderr == "oops"


async def test_returns_non_zero_exit_code(settings: Settings) -> None:
    result = await SubprocessBackend(settings).execute(
        python_command("import sys; sys.exit(3)")
    )

    assert result.exit_code == 3
    assert result.timed_out is False


async def test_timeout_kills_process(settings: Settings) -> None:
    result = await SubprocessBackend(settings).execute(
        python_command("import time; time.sleep(5)"),
        timeout_seconds=1,
    )

    assert result.timed_out is True


async def test_truncates_output(settings: Settings) -> None:
    result = await SubprocessBackend(settings).execute(
        python_command("print('x' * 500)"),
        max_output_bytes=50,
    )

    assert result.stdout_truncated is True
    assert len(result.stdout) <= 50
    assert result.stdout == "x" * 50


async def test_truncates_stderr(settings: Settings) -> None:
    result = await SubprocessBackend(settings).execute(
        python_command("import sys; sys.stderr.write('e' * 500)"),
        max_output_bytes=40,
    )

    assert result.stderr_truncated is True
    assert len(result.stderr) <= 40


async def test_uses_requested_working_directory(
    settings: Settings,
    tmp_path: Path,
) -> None:
    child = tmp_path / "child"
    child.mkdir()

    result = await SubprocessBackend(settings).execute(
        python_command("import os; print(os.getcwd())"),
        working_directory=str(child),
    )

    assert Path(result.stdout.strip()).resolve() == child.resolve()


async def test_defaults_to_first_allowed_root(
    settings: Settings,
    tmp_path: Path,
) -> None:
    result = await SubprocessBackend(settings).execute(
        python_command("import os; print(os.getcwd())")
    )

    assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()


async def test_rejects_working_directory_outside_roots(
    settings: Settings,
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()

    with pytest.raises(SandboxError, match="outside the allowed roots"):
        await SubprocessBackend(settings).execute(
            python_command("print('hello')"),
            working_directory=str(outside),
        )


async def test_rejects_missing_working_directory(
    settings: Settings,
    tmp_path: Path,
) -> None:
    with pytest.raises(SandboxError, match="does not exist"):
        await SubprocessBackend(settings).execute(
            python_command("print('hello')"),
            working_directory=str(tmp_path / "missing"),
        )


async def test_rejects_file_as_working_directory(
    settings: Settings,
    tmp_path: Path,
) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(SandboxError, match="not a directory"):
        await SubprocessBackend(settings).execute(
            python_command("print('hello')"),
            working_directory=str(target),
        )


async def test_rejects_execution_without_allowed_roots(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, allowed_roots=[])

    with pytest.raises(SandboxError, match="No allowed roots"):
        await SubprocessBackend(settings).execute(python_command("print('hello')"))


async def test_does_not_inherit_application_secrets(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")

    result = await SubprocessBackend(settings).execute(
        python_command("import os; print(os.environ.get('GEMINI_API_KEY', 'missing'))")
    )

    assert result.stdout.strip() == "missing"


async def test_passes_explicit_environment(settings: Settings) -> None:
    result = await SubprocessBackend(settings).execute(
        python_command("import os; print(os.environ.get('CUSTOM_VALUE', 'missing'))"),
        env={"CUSTOM_VALUE": "ok"},
    )

    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize(
    "command",
    ["", "   ", "\n\t"],
)
async def test_rejects_blank_command(settings: Settings, command: str) -> None:
    with pytest.raises(SandboxError, match="must not be blank"):
        await SubprocessBackend(settings).execute(command)


async def test_rejects_invalid_limits(settings: Settings) -> None:
    backend = SubprocessBackend(settings)

    with pytest.raises(SandboxError, match="timeout_seconds"):
        await backend.execute("echo hello", timeout_seconds=0)
    with pytest.raises(SandboxError, match="max_output_bytes"):
        await backend.execute("echo hello", max_output_bytes=0)


async def test_is_available(settings: Settings) -> None:
    assert await SubprocessBackend(settings).is_available() is True
