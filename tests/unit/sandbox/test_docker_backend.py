"""Tests for the optional Docker sandbox backend."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from gemini_cli.config import Settings
from gemini_cli.sandbox import DockerBackend, SubprocessBackend, select_backend
from gemini_cli.sandbox.base import SandboxError


class FakeStream:
    """Small async stream used instead of a real subprocess pipe."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._data):
            return b""
        if size < 0:
            size = len(self._data) - self._offset
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class FakeProcess:
    """Minimal subprocess stand-in with controllable completion."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
        wait_forever: bool = False,
    ) -> None:
        self.returncode = returncode
        self.pid = 4242
        self.stdout = FakeStream(stdout)
        self.stderr = FakeStream(stderr)
        self._done = asyncio.Event()
        if not wait_forever:
            self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9
        self._done.set()


def _patch_docker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    executable: str | None = "/usr/bin/docker",
) -> list[tuple[Any, ...]]:
    calls: list[tuple[Any, ...]] = []

    async def fake_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        del kwargs
        calls.append(args)
        if len(args) > 1 and args[1] == "run":
            return FakeProcess(returncode=0, stdout=b"hello\n")
        return FakeProcess(returncode=0)

    monkeypatch.setattr(
        "gemini_cli.sandbox.docker_backend.shutil.which",
        lambda _name: executable,
    )
    monkeypatch.setattr(
        "gemini_cli.sandbox.docker_backend.asyncio.create_subprocess_exec",
        fake_exec,
    )
    return calls


async def test_unavailable_without_docker(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gemini_cli.sandbox.docker_backend.shutil.which",
        lambda _name: None,
    )

    assert await DockerBackend(settings).is_available() is False


async def test_unavailable_when_daemon_is_down(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        del args, kwargs
        return FakeProcess(returncode=1)

    monkeypatch.setattr(
        "gemini_cli.sandbox.docker_backend.shutil.which",
        lambda _name: "/usr/bin/docker",
    )
    monkeypatch.setattr(
        "gemini_cli.sandbox.docker_backend.asyncio.create_subprocess_exec",
        fake_exec,
    )

    assert await DockerBackend(settings).is_available() is False


async def test_is_available_caches_probe(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_docker(monkeypatch)
    backend = DockerBackend(settings)

    assert await backend.is_available() is True
    assert await backend.is_available() is True
    assert len(calls) == 1


async def test_execute_builds_restricted_docker_command(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_docker(monkeypatch)
    backend = DockerBackend(settings)
    backend._available = True

    result = await backend.execute("echo hello")

    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    assert result.timed_out is False

    args = calls[0]
    assert args[0] == "/usr/bin/docker"
    assert "run" in args
    assert "--rm" in args
    assert "--network" in args
    assert "none" in args
    assert "--pull=never" in args
    assert "--read-only" in args
    assert "--cap-drop" in args
    assert "ALL" in args
    assert "--security-opt" in args
    assert "no-new-privileges" in args
    assert "-v" in args
    assert f"{tmp_path.resolve()}:/workspace:ro" in args
    assert "-w" in args
    assert "/workspace" in args
    assert "python:3.12-slim" in args
    assert "sh" in args
    assert "-c" in args
    assert "echo hello" in args


async def test_execute_can_enable_network_and_writes(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_docker(monkeypatch)
    backend = DockerBackend(
        settings,
        network_enabled=True,
        read_only_mount=False,
    )
    backend._available = True

    await backend.execute("echo hello", working_directory=str(tmp_path))

    args = calls[0]
    assert "bridge" in args
    assert f"{tmp_path.resolve()}:/workspace:rw" in args


async def test_execute_uses_nested_container_workdir(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_docker(monkeypatch)
    child = tmp_path / "nested" / "dir"
    child.mkdir(parents=True)
    backend = DockerBackend(settings)
    backend._available = True

    await backend.execute("echo hello", working_directory=str(child))

    args = calls[0]
    assert "/workspace/nested/dir" in args


async def test_execute_rejects_unavailable_docker(settings: Settings) -> None:
    backend = DockerBackend(settings)
    backend._available = False

    with pytest.raises(SandboxError, match="not available"):
        await backend.execute("echo hello")


async def test_timeout_cleans_up_container(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []

    async def fake_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        del kwargs
        calls.append(args)
        if len(args) > 1 and args[1] == "run":
            return FakeProcess(wait_forever=True)
        return FakeProcess(returncode=0)

    monkeypatch.setattr(
        "gemini_cli.sandbox.docker_backend.shutil.which",
        lambda _name: "/usr/bin/docker",
    )
    monkeypatch.setattr(
        "gemini_cli.sandbox.docker_backend.asyncio.create_subprocess_exec",
        fake_exec,
    )
    monkeypatch.setattr(
        "gemini_cli.sandbox.docker_backend.kill_process_group",
        lambda process: process.kill(),
    )

    backend = DockerBackend(settings)
    backend._available = True

    result = await backend.execute("sleep 10", timeout_seconds=1)

    assert result.timed_out is True
    assert result.exit_code == -9
    assert any(len(call) > 1 and call[1] == "kill" for call in calls)
    assert any(len(call) > 1 and call[1] == "rm" for call in calls)


async def test_select_backend_prefers_available_docker(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def available(_self: DockerBackend) -> bool:
        return True

    monkeypatch.setattr(DockerBackend, "is_available", available)

    backend = await select_backend(settings, prefer_docker=True)

    assert isinstance(backend, DockerBackend)


async def test_select_backend_falls_back_to_subprocess(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(_self: DockerBackend) -> bool:
        return False

    monkeypatch.setattr(DockerBackend, "is_available", unavailable)

    backend = await select_backend(settings, prefer_docker=True)

    assert isinstance(backend, SubprocessBackend)


async def test_select_backend_defaults_to_subprocess(settings: Settings) -> None:
    backend = await select_backend(settings)

    assert isinstance(backend, SubprocessBackend)
