"""Tests for REPL rendering, menus, and streaming behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from datetime import timedelta

import pytest

from gemini_cli import __version__
from gemini_cli.cli.tui.repl import (
    ConsoleApprovalCallback,
    MenuOption,
    MenuState,
    completion_words,
    handle_menu_key,
    model_options,
    render_bar,
    render_header,
    render_menu,
    render_quota_panel,
    role_options,
    run_repl,
    select_from_menu,
)
from gemini_cli.config import Settings
from gemini_cli.core.quota_tracker import QuotaWindowStatus
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEvent, AgentEventType
from gemini_cli.sandbox.approval import ApprovalDecision


class FakeQuota:
    """Deterministic quota statuses."""

    def get_5h_status(self) -> QuotaWindowStatus:
        return QuotaWindowStatus(
            used_tokens=100_000,
            limit_tokens=200_000,
            percentage=50.0,
            time_remaining=timedelta(hours=2),
        )

    def get_weekly_status(self) -> QuotaWindowStatus:
        return QuotaWindowStatus(
            used_tokens=310_000,
            limit_tokens=1_000_000,
            percentage=31.0,
            time_remaining=timedelta(days=3),
        )


class FakeAgent:
    """Yield a scripted event list for every turn."""

    def __init__(self, events: Sequence[AgentEvent]) -> None:
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
    """Minimal runtime surface used by the REPL."""

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


async def test_repl_streams_text_turn() -> None:
    runtime = FakeRuntime(
        [
            AgentEvent(AgentEventType.TEXT_CHUNK, "Hello"),
            AgentEvent(AgentEventType.DONE),
        ]
    )
    write, parts = collector()

    await run_repl(runtime, input_func=scripted_input(["hi"]), output=write)

    text = "".join(parts)
    assert "Hi" not in text
    assert "Hello" in text
    assert "GEMINI CLI" in text
    assert "User: hi" in text
    assert len(runtime.agent.calls) == 1
    assert runtime.agent.calls[0][1] == "hi"


async def test_repl_prints_errors() -> None:
    runtime = FakeRuntime(
        [
            AgentEvent(AgentEventType.ERROR, "boom"),
            AgentEvent(AgentEventType.DONE),
        ]
    )
    write, parts = collector()

    await run_repl(runtime, input_func=scripted_input(["hi"]), output=write)

    assert "Error: boom" in "".join(parts)


async def test_repl_exits_on_exit_command() -> None:
    runtime = FakeRuntime()
    write, _parts = collector()

    await run_repl(runtime, input_func=scripted_input(["exit"]), output=write)

    assert runtime.agent.calls == []


async def test_repl_exits_on_eof() -> None:
    runtime = FakeRuntime()
    write, _parts = collector()

    await run_repl(runtime, input_func=scripted_input([]), output=write)

    assert runtime.agent.calls == []


async def test_approval_accepts_yes() -> None:
    callback = ConsoleApprovalCallback(input_func=lambda _prompt: "y")

    decision = await callback.request_approval("run a tool", "write", "path: a")

    assert decision is ApprovalDecision.APPROVED


@pytest.mark.parametrize("answer", ["n", "", "no"])
async def test_approval_denies_by_default(answer: str) -> None:
    callback = ConsoleApprovalCallback(input_func=lambda _prompt: answer)

    decision = await callback.request_approval("run a tool", "write", "path: a")

    assert decision is ApprovalDecision.DENIED


async def test_approval_denies_on_eof() -> None:
    def failing_input(_prompt: str) -> str:
        raise EOFError

    callback = ConsoleApprovalCallback(input_func=failing_input)

    decision = await callback.request_approval("run a tool", "write", "path: a")

    assert decision is ApprovalDecision.DENIED


def test_render_header_includes_version_and_model() -> None:
    header = render_header(FakeRuntime())

    assert f"v{__version__}" in header
    assert "gemini-2.5-flash" in header
    assert "100k/200k tokens" in header


def test_render_bar_bounds() -> None:
    assert render_bar(0) == "░" * 15
    assert render_bar(100) == "█" * 15
    assert render_bar(-10) == "░" * 15
    assert render_bar(150) == "█" * 15
    assert render_bar(50).count("█") == 8


def test_render_quota_panel_contains_percentages_and_time() -> None:
    panel = render_quota_panel(
        FakeQuota().get_5h_status(),
        FakeQuota().get_weekly_status(),
    )

    assert "QUOTA" in panel
    assert "50.0%" in panel
    assert "2h 00m remaining" in panel
    assert "31.0%" in panel


def test_menu_navigation_wraps_within_bounds() -> None:
    state = MenuState(role_options())

    state, decision = handle_menu_key(state, "up")
    assert state.index == 0
    assert decision is None

    state, _ = handle_menu_key(state, "down")
    assert state.index == 1

    state = MenuState(role_options(), index=len(role_options()) - 1)
    state, _ = handle_menu_key(state, "down")
    assert state.index == len(role_options()) - 1


def test_menu_enter_selects_and_escape_cancels() -> None:
    state = MenuState(role_options(), index=2)

    _state, decision = handle_menu_key(state, "enter")
    assert decision is not None
    assert decision.index == 2

    _state, decision = handle_menu_key(state, "escape")
    assert decision is not None
    assert decision.index is None


def test_menu_ignores_unknown_keys() -> None:
    state = MenuState(role_options())

    unchanged, decision = handle_menu_key(state, "x")

    assert unchanged == state
    assert decision is None


def test_render_menu_marks_selected_option() -> None:
    state = MenuState(role_options(), index=1)

    rendered = render_menu(state, "Roles:")

    assert "Roles:" in rendered
    assert "> Gemini Code" in rendered
    assert "  Gemini Basic" in rendered


def test_role_and_model_options_cover_the_registry() -> None:
    assert {option.value for option in role_options()} == {
        "basic",
        "code",
        "creator",
        "analyser",
        "poet",
    }
    assert {option.value for option in model_options()} == {
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    }


def test_completion_words_include_every_slash_command() -> None:
    words = completion_words()

    assert "/help" in words
    assert "/quota" in words
    assert words == sorted(words)


async def test_select_from_menu_returns_none_without_options() -> None:
    assert await select_from_menu([], "Empty") is None


async def test_select_from_menu_uses_application() -> None:
    options = (MenuOption("a", "A"), MenuOption("b", "B"))

    class FakeApplication:
        async def run_async(self) -> int:
            return 1

    from gemini_cli.cli.tui import repl

    original = repl._build_menu_application
    repl._build_menu_application = lambda _options, _title: (
        FakeApplication(),
        MenuState(options),
    )
    try:
        selected = await select_from_menu(options, "Pick")
    finally:
        repl._build_menu_application = original

    assert selected == options[1]


def test_default_theme_has_expected_styles() -> None:
    from gemini_cli.cli.tui.themes import DEFAULT_THEME

    assert DEFAULT_THEME.name == "dark"
    assert DEFAULT_THEME.assistant_color == "bright_cyan"
    assert DEFAULT_THEME.error_color == "red"


def test_get_theme_is_case_insensitive() -> None:
    from gemini_cli.cli.tui.themes import LIGHT_THEME, get_theme

    assert get_theme("LIGHT") is LIGHT_THEME


def test_get_theme_rejects_unknown_names() -> None:
    from gemini_cli.cli.tui.themes import UnknownThemeError, get_theme

    with pytest.raises(UnknownThemeError, match="solarized"):
        get_theme("solarized")


async def test_repl_accepts_a_theme() -> None:
    from gemini_cli.cli.tui.themes import LIGHT_THEME

    runtime = FakeRuntime([AgentEvent(AgentEventType.DONE)])
    write, parts = collector()

    await run_repl(
        runtime,
        input_func=scripted_input(["hi"]),
        output=write,
        theme=LIGHT_THEME,
    )

    assert "GEMINI CLI" in "".join(parts)
