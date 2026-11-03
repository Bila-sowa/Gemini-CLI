"""Tests for the sandbox backend contract."""

from __future__ import annotations

from gemini_cli.sandbox.base import (
    ExecutionResult,
    SandboxBackend,
    SandboxError,
)


class FakeBackend:
    """Minimal backend used to check structural protocol compatibility."""

    async def execute(
        self,
        command: str,
        *,
        working_directory: str | None = None,
        timeout_seconds: int = 30,
        max_output_bytes: int = 524_288,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        del command, working_directory, timeout_seconds, max_output_bytes, env
        return ExecutionResult(
            exit_code=0,
            stdout="ok\n",
            stderr="",
            timed_out=False,
            duration_seconds=0.01,
        )

    async def is_available(self) -> bool:
        return True


def test_execution_result_defaults() -> None:
    result = ExecutionResult(
        exit_code=0,
        stdout="hello\n",
        stderr="",
        timed_out=False,
        duration_seconds=0.5,
    )

    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    assert result.stderr == ""
    assert result.timed_out is False
    assert result.duration_seconds == 0.5
    assert result.stdout_truncated is False
    assert result.stderr_truncated is False


def test_execution_result_is_frozen() -> None:
    result = ExecutionResult(
        exit_code=0,
        stdout="",
        stderr="",
        timed_out=False,
        duration_seconds=0.0,
    )

    try:
        result.exit_code = 1  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("ExecutionResult must be immutable.")


async def test_fake_backend_matches_protocol() -> None:
    backend: SandboxBackend = FakeBackend()

    assert await backend.is_available() is True
    result = await backend.execute("echo hello")
    assert result.stdout == "ok\n"


def test_sandbox_error_is_runtime_error() -> None:
    error = SandboxError("cannot run")

    assert isinstance(error, RuntimeError)
    assert str(error) == "cannot run"
