"""Tests for the CLI application entry point."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gemini_cli import __version__
from gemini_cli.auth import AuthenticationError
from gemini_cli.auth.api_key import ResolvedApiKey
from gemini_cli.cli.app import app
from gemini_cli.cli.runtime import build_runtime
from gemini_cli.cli.tui.repl import ConsoleApprovalCallback
from gemini_cli.config import Settings
from gemini_cli.llm import LLMError, UnknownModelError

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


def test_config_show_does_not_resolve_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Settings] = []

    def fake_resolve(settings: Settings) -> ResolvedApiKey:
        calls.append(settings)
        raise AssertionError("credentials must not be resolved for config show")

    monkeypatch.setattr("gemini_cli.cli.runtime.resolve_api_key", fake_resolve)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
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


def test_chat_wires_runtime_and_repl(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []
    called: list[object] = []
    captured: dict[str, object] = {}

    class FakeRuntime:
        def close(self) -> None:
            closed.append(True)

    async def fake_build(
        settings: Settings,
        *,
        model: str | None = None,
        role: str | None = None,
        approval_callback: object = None,
    ) -> FakeRuntime:
        del settings
        captured["model"] = model
        captured["role"] = role
        captured["approval"] = approval_callback
        return FakeRuntime()

    async def fake_repl(runtime: object) -> None:
        called.append(runtime)

    monkeypatch.setattr(
        "gemini_cli.cli.commands.chat.build_runtime",
        fake_build,
    )
    monkeypatch.setattr("gemini_cli.cli.commands.chat.run_repl", fake_repl)

    result = runner.invoke(
        app,
        ["chat", "--model", "gemini-2.5-pro", "--role", "code"],
    )

    assert result.exit_code == 0
    assert len(called) == 1
    assert closed == [True]
    assert captured["model"] == "gemini-2.5-pro"
    assert captured["role"] == "code"
    assert isinstance(captured["approval"], ConsoleApprovalCallback)


def test_chat_reports_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_build(
        settings: Settings,
        **kwargs: object,
    ) -> None:
        del settings, kwargs
        raise AuthenticationError("no key configured")

    monkeypatch.setattr(
        "gemini_cli.cli.commands.chat.build_runtime",
        failing_build,
    )

    result = runner.invoke(app, ["chat"])

    assert result.exit_code == 1


def test_login_stores_key_without_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: list[str] = []
    monkeypatch.setattr(
        "gemini_cli.cli.commands.auth.store_api_key",
        stored.append,
    )

    result = runner.invoke(
        app,
        ["login", "--api-key", "secret-key", "--no-verify"],
    )

    assert result.exit_code == 0
    assert stored == ["secret-key"]
    assert "secret-key" not in result.stdout


def test_login_prompts_for_key(monkeypatch: pytest.MonkeyPatch) -> None:
    stored: list[str] = []
    monkeypatch.setattr(
        "gemini_cli.cli.commands.auth.store_api_key",
        stored.append,
    )

    result = runner.invoke(
        app,
        ["login", "--no-verify"],
        input="prompted-key\n",
    )

    assert result.exit_code == 0
    assert stored == ["prompted-key"]


def test_login_verifies_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    verified: list[str] = []
    stored: list[str] = []

    async def fake_verify(key: str) -> None:
        verified.append(key)

    monkeypatch.setattr("gemini_cli.cli.commands.auth._verify_key", fake_verify)
    monkeypatch.setattr(
        "gemini_cli.cli.commands.auth.store_api_key",
        stored.append,
    )

    result = runner.invoke(app, ["login", "--api-key", "checked-key"])

    assert result.exit_code == 0
    assert verified == ["checked-key"]
    assert stored == ["checked-key"]


def test_login_rejects_failed_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: list[str] = []

    async def failing_verify(key: str) -> None:
        del key
        raise LLMError("bad key")

    monkeypatch.setattr("gemini_cli.cli.commands.auth._verify_key", failing_verify)
    monkeypatch.setattr(
        "gemini_cli.cli.commands.auth.store_api_key",
        stored.append,
    )

    result = runner.invoke(app, ["login", "--api-key", "bad-key"])

    assert result.exit_code == 1
    assert stored == []


def test_login_rejects_empty_key(monkeypatch: pytest.MonkeyPatch) -> None:
    stored: list[str] = []
    monkeypatch.setattr(
        "gemini_cli.cli.commands.auth.store_api_key",
        stored.append,
    )

    result = runner.invoke(
        app,
        ["login", "--no-verify", "--api-key", " "],
        input="\n",
    )

    assert result.exit_code == 1
    assert stored == []


def test_logout_deletes_stored_key(monkeypatch: pytest.MonkeyPatch) -> None:
    deleted: list[bool] = []
    monkeypatch.setattr(
        "gemini_cli.cli.commands.auth.delete_api_key",
        lambda: deleted.append(True),
    )

    result = runner.invoke(app, ["logout"])

    assert result.exit_code == 0
    assert deleted == [True]


def test_status_reports_keyring_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gemini_cli.auth.api_key.load_api_key",
        lambda: "stored-secret",
    )

    result = runner.invoke(app, ["auth", "status"])

    assert result.exit_code == 0
    assert "keyring" in result.stdout
    assert "stored-secret" not in result.stdout


def test_status_reports_environment_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-secret")

    result = runner.invoke(app, ["auth", "status"])

    assert result.exit_code == 0
    assert "environment" in result.stdout
    assert "env-secret" not in result.stdout


def test_status_reports_missing_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gemini_cli.auth.api_key.load_api_key",
        lambda: None,
    )

    result = runner.invoke(app, ["auth", "status"])

    assert result.exit_code == 1
    assert "not configured" in result.stdout


def test_config_show_redacts_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "top-secret")

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "api_key: set" in result.stdout
    assert "top-secret" not in result.stdout
    assert "model: gemini-2.5-flash" in result.stdout


def test_config_show_reports_missing_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "api_key: not set" in result.stdout


def test_config_set_writes_project_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "set", "model", "gemini-2.5-pro"])

    assert result.exit_code == 0
    document = tomllib.loads((tmp_path / ".gemini-cli.toml").read_text())
    assert document["model"] == "gemini-2.5-pro"


def test_config_set_preserves_existing_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["config", "set", "model", "gemini-2.5-pro"])
    result = runner.invoke(
        app,
        ["config", "set", "telemetry_enabled", "true"],
    )

    assert result.exit_code == 0
    document = tomllib.loads((tmp_path / ".gemini-cli.toml").read_text())
    assert document["model"] == "gemini-2.5-pro"
    assert document["telemetry_enabled"] is True


def test_config_set_writes_global_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["config", "set", "role", "code", "--global"],
    )

    assert result.exit_code == 0
    path = tmp_path / ".gemini-cli" / "config.toml"
    document = tomllib.loads(path.read_text())
    assert document["role"] == "code"


def test_config_set_rejects_blocked_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "set", "api_key", "secret"])

    assert result.exit_code == 1


def test_config_set_rejects_unknown_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "set", "not_a_setting", "x"])

    assert result.exit_code == 1


def test_config_set_rejects_invalid_integer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "set", "retry_attempts", "many"])

    assert result.exit_code == 1


def test_config_set_rejects_invalid_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "set", "role", "wizard"])

    assert result.exit_code == 1


def test_config_set_rejects_invalid_boolean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["config", "set", "telemetry_enabled", "maybe"],
    )

    assert result.exit_code == 1
