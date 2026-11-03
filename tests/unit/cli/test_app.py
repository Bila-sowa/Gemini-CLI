"""Tests for the CLI application entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gemini_cli import __version__
from gemini_cli.auth.api_key import ResolvedApiKey
from gemini_cli.cli.app import app
from gemini_cli.cli.runtime import build_runtime
from gemini_cli.config import Settings
from gemini_cli.llm import UnknownModelError

runner = CliRunner()

BUILTIN_TOOLS = {
    "code_search",
    "edit_file",
    "fetch_url",
    "find_files",
    "grep",
    "read_file",
    "run_command",
    "write_file",
}


def test_version_flag_prints_version() -> None:
    """``--version`` prints the package version and exits successfully."""
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"v{__version__}"


def test_module_entry_point_prints_version() -> None:
    """``python -m gemini_cli --version`` works without extra setup."""
    result = subprocess.run(
        [sys.executable, "-m", "gemini_cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"v{__version__}"


def test_help_lists_top_level_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("chat", "run", "login", "logout", "auth", "config"):
        assert command in result.stdout


def test_auth_help_lists_status() -> None:
    result = runner.invoke(app, ["auth", "--help"])

    assert result.exit_code == 0
    assert "status" in result.stdout


def test_config_help_lists_show_and_set() -> None:
    result = runner.invoke(app, ["config", "--help"])

    assert result.exit_code == 0
    assert "show" in result.stdout
    assert "set" in result.stdout


def test_unknown_command_fails() -> None:
    result = runner.invoke(app, ["does-not-exist"])

    assert result.exit_code != 0


def test_help_does_not_resolve_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Settings] = []

    def fake_resolve(settings: Settings) -> ResolvedApiKey:
        calls.append(settings)
        raise AssertionError("credentials must not be resolved for --help")

    monkeypatch.setattr("gemini_cli.cli.runtime.resolve_api_key", fake_resolve)

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert calls == []


def test_config_show_stub_does_not_resolve_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Settings] = []

    def fake_resolve(settings: Settings) -> ResolvedApiKey:
        calls.append(settings)
        raise AssertionError("credentials must not be resolved for config show")

    monkeypatch.setattr("gemini_cli.cli.runtime.resolve_api_key", fake_resolve)

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 1
    assert calls == []


async def test_build_runtime_registers_builtin_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "gemini_cli.cli.runtime.resolve_api_key",
        lambda settings: ResolvedApiKey(key="test-key", source="environment"),
    )
    monkeypatch.setattr(
        "gemini_cli.core.quota_tracker._default_storage_path",
        lambda: tmp_path / "quota.json",
    )

    settings = Settings(_env_file=None, allowed_roots=[])
    runtime = await build_runtime(settings)
    try:
        assert {tool.name for tool in runtime.registry.list_tools()} == BUILTIN_TOOLS
        assert runtime.settings.allowed_roots == [tmp_path]
        assert runtime.system_prompt
        assert runtime.agent is not None
    finally:
        runtime.close()


async def test_build_runtime_applies_model_and_role_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "gemini_cli.cli.runtime.resolve_api_key",
        lambda settings: ResolvedApiKey(key="test-key", source="environment"),
    )
    monkeypatch.setattr(
        "gemini_cli.core.quota_tracker._default_storage_path",
        lambda: tmp_path / "quota.json",
    )

    settings = Settings(_env_file=None, allowed_roots=[tmp_path])
    runtime = await build_runtime(settings, model="gemini-2.5-pro", role="code")
    try:
        assert runtime.settings.model == "gemini-2.5-pro"
        assert runtime.settings.role == "code"
        assert runtime.settings.allowed_roots == [tmp_path]
    finally:
        runtime.close()


async def test_build_runtime_rejects_unknown_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "gemini_cli.cli.runtime.resolve_api_key",
        lambda settings: ResolvedApiKey(key="test-key", source="environment"),
    )
    monkeypatch.setattr(
        "gemini_cli.core.quota_tracker._default_storage_path",
        lambda: tmp_path / "quota.json",
    )

    settings = Settings(_env_file=None, allowed_roots=[tmp_path])

    with pytest.raises(UnknownModelError):
        await build_runtime(settings, model="gemini-does-not-exist")
