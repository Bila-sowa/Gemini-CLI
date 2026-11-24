"""Tests for slash command parsing and line-mode dispatch."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from datetime import timedelta

import pytest

from gemini_cli.cli.tui.repl import (
    SlashCommand,
    parse_slash_command,
    run_repl,
)
from gemini_cli.config import Settings
from gemini_cli.core.quota_tracker import QuotaWindowStatus
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEvent


class FakeQuota:
    def get_5h_status(self) -> QuotaWindowStatus:
        return QuotaWindowStatus(100_000, 200_000, 50.0, timedelta(hours=2))

    def get_weekly_status(self) -> QuotaWindowStatus:
        return QuotaWindowStatus(310_000, 1_000_000, 31.0, timedelta(days=3))


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[Session, str]] = []

    async def run(
        self,
        session: Session,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        self.calls.append((session, user_message))
        return
        yield  # pragma: no cover


class FakeRuntime:
    def __init__(
        self,
        *,
        model: str = "gemini-3.8-flash",
        role: str = "basic",
    ) -> None:
        self.settings = Settings(
            _env_file=None,
            model=model,
            role=role,
        )
        self.agent = FakeAgent()
        self.quota = FakeQuota()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def scripted_input(lines: Sequence[str]) -> Callable[[str], str]:
    iterator = iter(lines)

    def read(_prompt: str) -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError from None

    return read


def collector() -> tuple[Callable[[str], None], list[str]]:
    parts: list[str] = []

    def write(text: str) -> None:
        parts.append(text)

    return write, parts


async def run_commands(commands: Sequence[str]) -> tuple[str, FakeRuntime]:
    runtime = FakeRuntime()
    write, parts = collector()
    await run_repl(
        runtime,
        input_func=scripted_input([*commands, "/exit"]),
        output=write,
    )
    return "".join(parts), runtime


def test_parse_plain_text_is_not_a_command() -> None:
    assert parse_slash_command("hello world") is None


def test_parse_slash_command_without_argument() -> None:
    assert parse_slash_command("  /help ") == SlashCommand("/help")


def test_parse_slash_command_with_argument() -> None:
    assert parse_slash_command("/ROLE code") == SlashCommand("/role", "code")


def test_parse_unknown_slash_command_still_parses() -> None:
    assert parse_slash_command("/nope") == SlashCommand("/nope")


async def test_help_lists_commands() -> None:
    text, _runtime = await run_commands(["/help"])

    assert "Commands:" in text
    assert "/quota" in text


async def test_clear_resets_the_conversation() -> None:
    text, _runtime = await run_commands(["/clear"])

    assert "Conversation cleared" in text


async def test_unknown_command_is_reported() -> None:
    text, _runtime = await run_commands(["/nope"])

    assert "Unknown command /nope" in text


async def test_menu_commands_point_to_the_tui() -> None:
    text, _runtime = await run_commands(["/role", "/model", "/theme"])

    assert text.count("needs the full-screen UI") == 3


async def test_theme_argument_is_tui_only_in_line_mode() -> None:
    text, _runtime = await run_commands(["/theme light"])

    assert "needs the full-screen UI" in text


async def test_plural_and_short_aliases_use_the_same_dispatch() -> None:
    text, _runtime = await run_commands(["/models", "/roles", "/themes", "/m"])

    assert text.count("needs the full-screen UI") == 4


async def test_unique_prefix_uses_the_same_dispatch() -> None:
    text, _runtime = await run_commands(["/mod"])

    assert "/model needs the full-screen UI" in text


@pytest.mark.parametrize("command", ["/q", "/quit", "/bye"])
async def test_exit_aliases_do_not_contact_the_agent(command: str) -> None:
    _text, runtime = await run_commands([command])

    assert runtime.agent.calls == []


async def test_model_argument_rebuilds_and_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[FakeRuntime] = []
    saved: list[tuple[str, str]] = []

    async def fake_build(
        settings: Settings,
        *,
        model: str | None = None,
        role: str | None = None,
        approval_callback: object = None,
    ) -> FakeRuntime:
        del approval_callback
        replacement = FakeRuntime(
            model=model or settings.model,
            role=role or settings.role,
        )
        built.append(replacement)
        return replacement

    def fake_set_env(_path: object, key: str, value: str) -> None:
        saved.append((key, value))

    monkeypatch.setattr("gemini_cli.cli.tui.repl.build_runtime", fake_build)
    monkeypatch.setattr("gemini_cli.cli.tui.repl.set_env_value", fake_set_env)

    write, parts = collector()
    runtime = FakeRuntime()
    await run_repl(
        runtime,
        input_func=scripted_input(["/model gemini-4-argon", "hello", "/exit"]),
        output=write,
    )

    assert built
    assert built[0].settings.model == "gemini-4-argon"
    assert saved == [("GEMINI_MODEL", "gemini-4-argon")]
    assert built[0].agent.calls[0][0].model == "gemini-4-argon"
    assert "Model: gemini-4-argon" in "".join(parts)


async def test_role_argument_rebuilds_and_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[FakeRuntime] = []
    saved: list[tuple[str, str]] = []

    async def fake_build(
        settings: Settings,
        *,
        model: str | None = None,
        role: str | None = None,
        approval_callback: object = None,
    ) -> FakeRuntime:
        del approval_callback
        replacement = FakeRuntime(
            model=model or settings.model,
            role=role or settings.role,
        )
        built.append(replacement)
        return replacement

    def fake_set_env(_path: object, key: str, value: str) -> None:
        saved.append((key, value))

    monkeypatch.setattr("gemini_cli.cli.tui.repl.build_runtime", fake_build)
    monkeypatch.setattr("gemini_cli.cli.tui.repl.set_env_value", fake_set_env)

    write, parts = collector()
    runtime = FakeRuntime()
    await run_repl(
        runtime,
        input_func=scripted_input(["/role code", "hello", "/exit"]),
        output=write,
    )

    assert built
    assert built[0].settings.role == "code"
    assert saved == [("GEMINI_ROLE", "code")]
    assert built[0].agent.calls[0][0].role == "code"
    assert "Role: code" in "".join(parts)


async def test_invalid_model_argument_suggests_a_close_match() -> None:
    text, _runtime = await run_commands(["/model gemini-3.5-flash-lit"])

    assert "Error: Unknown model" in text
    assert "Did you mean 'gemini-3.5-flash-lite'?" in text


async def test_help_command_shows_argument_details() -> None:
    text, _runtime = await run_commands(["/help model"])

    assert "/model [name]" in text
    assert "Aliases: /models, /m" in text


async def test_command_without_arguments_rejects_extra_text() -> None:
    text, _runtime = await run_commands(["/clear extra"])

    assert "/clear does not take arguments." in text
    assert "Conversation cleared" not in text


async def test_unknown_command_suggests_a_close_match() -> None:
    text, _runtime = await run_commands(["/modl"])

    assert "Did you mean /model?" in text


async def test_ambiguous_prefix_is_reported() -> None:
    text, _runtime = await run_commands(["/qu"])

    assert "ambiguous" in text
    assert "/quota" in text
    assert "/exit" in text


async def test_help_lists_command_aliases() -> None:
    text, _runtime = await run_commands(["/help"])

    assert "/models" in text
    assert "/quit" in text
    assert "/usage" in text


async def test_reset_removes_the_stored_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[bool] = []
    monkeypatch.setattr(
        "gemini_cli.cli.tui.repl.delete_api_key",
        lambda: removed.append(True),
    )

    text, _runtime = await run_commands(["/reset"])

    assert removed == [True]
    assert "Stored API key removed" in text


async def test_exit_does_not_contact_the_agent() -> None:
    _text, runtime = await run_commands([])

    assert runtime.agent.calls == []
