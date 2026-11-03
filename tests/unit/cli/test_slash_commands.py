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
    def __init__(self) -> None:
        self.settings = Settings(
            _env_file=None,
            model="gemini-3.8-flash",
            role="basic",
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
