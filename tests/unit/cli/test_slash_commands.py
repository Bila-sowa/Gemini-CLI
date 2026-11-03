"""Tests for slash command parsing and dispatch."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from datetime import timedelta

import pytest

from gemini_cli.cli.tui.repl import (
    MenuOption,
    SlashCommand,
    completion_words,
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
    def __init__(self, events: Sequence[AgentEvent] = ()) -> None:
        self.events = list(events)
        self.calls: list[tuple[Session, str]] = []

    async def run(
        self,
        session: Session,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        self.calls.append((session, user_message))
        for event in self.events:
            yield event


class FakeRuntime:
    def __init__(
        self,
        events: Sequence[AgentEvent] = (),
        *,
        model: str = "gemini-2.5-flash",
        role: str = "basic",
    ) -> None:
        self.settings = Settings(_env_file=None, model=model, role=role)
        self.agent = FakeAgent(events)
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


async def run_commands(
    commands: Sequence[str],
    runtime: FakeRuntime,
    *,
    menu: Callable[..., object] | None = None,
    rebuild: Callable[..., object] | None = None,
) -> tuple[str, list[str]]:
    write, parts = collector()
    kwargs: dict[str, object] = {}
    if menu is not None:
        kwargs["menu_func"] = menu
    if rebuild is not None:
        kwargs["rebuild"] = rebuild
    await run_repl(
        runtime,
        input_func=scripted_input([*commands, "/exit"]),
        output=write,
        **kwargs,
    )
    return "".join(parts), parts


def test_parse_plain_text_is_not_a_command() -> None:
    assert parse_slash_command("hello world") is None


def test_parse_slash_command_without_argument() -> None:
    assert parse_slash_command("  /help ") == SlashCommand("/help")


def test_parse_slash_command_with_argument() -> None:
    assert parse_slash_command("/ROLE code") == SlashCommand("/role", "code")


def test_parse_unknown_slash_command_still_parses() -> None:
    assert parse_slash_command("/nope") == SlashCommand("/nope")


def test_completion_words_match_slash_commands() -> None:
    words = completion_words()

    assert words == sorted(words)
    assert "/role" in words
    assert "/exit" in words


async def test_help_lists_commands() -> None:
    text, _parts = await run_commands(["/help"], FakeRuntime())

    assert "Commands:" in text
    assert "/quota" in text


async def test_quota_panel_is_rendered() -> None:
    text, _parts = await run_commands(["/quota"], FakeRuntime())

    assert "QUOTA" in text
    assert "50.0%" in text


async def test_clear_resets_the_conversation() -> None:
    text, _parts = await run_commands(["/clear"], FakeRuntime())

    assert "cleared" in text


async def test_unknown_command_is_reported() -> None:
    text, _parts = await run_commands(["/nope"], FakeRuntime())

    assert "Unknown command /nope" in text


async def test_role_selection_rebuilds_the_runtime() -> None:
    calls: list[tuple[str | None, str | None]] = []

    async def menu(
        options: Sequence[MenuOption],
        _title: str,
    ) -> MenuOption | None:
        return next(option for option in options if option.value == "code")

    async def rebuild(
        _runtime: FakeRuntime,
        *,
        model: str | None = None,
        role: str | None = None,
    ) -> FakeRuntime:
        calls.append((model, role))
        return FakeRuntime(model=model or "gemini-2.5-flash", role=role or "basic")

    text, _parts = await run_commands(
        ["/role"],
        FakeRuntime(),
        menu=menu,
        rebuild=rebuild,
    )

    assert calls == [(None, "code")]
    assert "model: gemini-2.5-flash" in text


async def test_model_selection_rebuilds_the_runtime() -> None:
    calls: list[tuple[str | None, str | None]] = []

    async def menu(
        options: Sequence[MenuOption],
        _title: str,
    ) -> MenuOption | None:
        return next(option for option in options if option.value == "gemini-2.5-pro")

    async def rebuild(
        _runtime: FakeRuntime,
        *,
        model: str | None = None,
        role: str | None = None,
    ) -> FakeRuntime:
        calls.append((model, role))
        return FakeRuntime(model=model or "gemini-2.5-flash", role=role or "basic")

    await run_commands(
        ["/model"],
        FakeRuntime(),
        menu=menu,
        rebuild=rebuild,
    )

    assert calls == [("gemini-2.5-pro", None)]


async def test_cancelled_selection_changes_nothing() -> None:
    async def menu(
        _options: Sequence[MenuOption],
        _title: str,
    ) -> MenuOption | None:
        return None

    text, _parts = await run_commands(["/role"], FakeRuntime(), menu=menu)

    assert "Selection cancelled" in text


async def test_auth_menu_points_to_login() -> None:
    async def menu(
        options: Sequence[MenuOption],
        _title: str,
    ) -> MenuOption | None:
        return next(option for option in options if option.value == "api_key")

    text, _parts = await run_commands(["/auth"], FakeRuntime(), menu=menu)

    assert "gemini-cli-py login" in text


async def test_auth_menu_reports_google_unavailable() -> None:
    async def menu(
        options: Sequence[MenuOption],
        _title: str,
    ) -> MenuOption | None:
        return next(option for option in options if option.value == "google")

    text, _parts = await run_commands(["/auth"], FakeRuntime(), menu=menu)

    assert "not available yet" in text


async def test_reset_removes_the_stored_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[bool] = []
    monkeypatch.setattr(
        "gemini_cli.cli.tui.repl.delete_api_key",
        lambda: removed.append(True),
    )

    text, _parts = await run_commands(["/reset"], FakeRuntime())

    assert removed == [True]
    assert "Stored API key removed" in text


async def test_reset_reports_keyring_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_cli.auth import CredentialStoreError

    def failing_delete() -> None:
        raise CredentialStoreError("keyring unavailable")

    monkeypatch.setattr(
        "gemini_cli.cli.tui.repl.delete_api_key",
        failing_delete,
    )

    text, _parts = await run_commands(["/reset"], FakeRuntime())

    assert "keyring unavailable" in text


async def test_exit_does_not_contact_the_agent() -> None:
    runtime = FakeRuntime()

    await run_commands([], runtime)

    assert runtime.agent.calls == []
