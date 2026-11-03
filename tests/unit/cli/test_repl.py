"""Tests for the terminal UI and the plain line fallback."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from datetime import timedelta

import pytest

from gemini_cli.cli.tui.repl import (
    ConsoleApprovalCallback,
    MenuOption,
    model_options,
    role_options,
    run_repl,
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
    """Minimal runtime surface used by both interfaces."""

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


# --- Plain line fallback -------------------------------------------------


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


async def test_repl_skips_blank_lines() -> None:
    runtime = FakeRuntime()
    write, _parts = collector()

    await run_repl(runtime, input_func=scripted_input(["", "   "]), output=write)

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


# --- Themes --------------------------------------------------------------


def test_default_theme_has_expected_styles() -> None:
    from gemini_cli.cli.tui.themes import DEFAULT_THEME

    assert DEFAULT_THEME.name == "dark"
    assert DEFAULT_THEME.assistant_color == "#5B9BD5"
    assert DEFAULT_THEME.error_color == "#E06C75"


def test_get_theme_is_case_insensitive() -> None:
    from gemini_cli.cli.tui.themes import LIGHT_THEME, get_theme

    assert get_theme("LIGHT") is LIGHT_THEME


def test_get_theme_rejects_unknown_names() -> None:
    from gemini_cli.cli.tui.themes import UnknownThemeError, get_theme

    with pytest.raises(UnknownThemeError, match="solarized"):
        get_theme("solarized")


def test_light_theme_uses_a_light_palette() -> None:
    from gemini_cli.cli.tui.themes import LIGHT_THEME

    assert LIGHT_THEME.background == "#F5F6F7"
    assert LIGHT_THEME.foreground == "#1E2023"
    assert LIGHT_THEME.name == "light"


# --- Textual application -------------------------------------------------


async def test_tui_shell_renders_status_and_footer() -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from gemini_cli.cli.tui.widgets import RoleBar, StatusBar

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert "version" in app.query_one(StatusBar).plain
        assert "gemini-2.5-flash" in app.query_one(StatusBar).plain
        assert "Gemini Basic" in app.query_one(RoleBar).plain
        assert any("Can I help you today?" in line for line in app.transcript)


async def test_tui_accepts_prompt_input() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("h", "i", "enter")
        await pilot.pause()
        assert "User: hi" in app.transcript


async def test_tui_streams_agent_turn() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(
        FakeRuntime(
            [
                AgentEvent(AgentEventType.TEXT_CHUNK, "Hel"),
                AgentEvent(AgentEventType.TEXT_CHUNK, "lo"),
                AgentEvent(AgentEventType.DONE),
            ]
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("h", "i", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert "User: hi" in app.transcript
    assert any(line == "✦: Hello" for line in app.transcript)


async def test_tui_renders_agent_error() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(
        FakeRuntime(
            [
                AgentEvent(AgentEventType.ERROR, "boom"),
                AgentEvent(AgentEventType.DONE),
            ]
        )
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("h", "i", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any("Error: boom" in line for line in app.transcript)


async def test_approval_screen_approves_with_y() -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from gemini_cli.cli.tui.widgets import ApprovalScreen

    app = GeminiApp(FakeRuntime())
    results: list[bool | None] = []

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(
            ApprovalScreen("run a tool", "write", "path: a"),
            callback=results.append,
        )
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()

    assert results == [True]


async def test_approval_screen_denies_with_escape() -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from gemini_cli.cli.tui.widgets import ApprovalScreen

    app = GeminiApp(FakeRuntime())
    results: list[bool | None] = []

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(
            ApprovalScreen("run a tool", "write", "path: a"),
            callback=results.append,
        )
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert results == [False]


async def test_approval_bridge_denies_without_app() -> None:
    from gemini_cli.cli.tui.app import TuiApprovalBridge

    bridge = TuiApprovalBridge()

    decision = await bridge.request_approval("run", "write", "path: a")

    assert decision is ApprovalDecision.DENIED


async def test_approval_bridge_routes_to_the_app() -> None:
    from gemini_cli.cli.tui.app import TuiApprovalBridge

    class FakeApp:
        def __init__(self, approved: bool) -> None:
            self.approved = approved

        async def request_approval(self, *args: object) -> bool:
            del args
            return self.approved

    bridge = TuiApprovalBridge()
    bridge.attach(FakeApp(True))  # type: ignore[arg-type]

    assert (
        await bridge.request_approval("run", "write", "path: a")
        is ApprovalDecision.APPROVED
    )


async def test_menu_screen_selects_with_enter() -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from gemini_cli.cli.tui.widgets import MenuScreen

    app = GeminiApp(FakeRuntime())
    results: list[object] = []

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(
            MenuScreen(role_options(), "Roles:"),
            callback=results.append,
        )
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()

    assert results
    assert getattr(results[0], "value", None) == "code"


async def test_menu_screen_cancels_with_escape() -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from gemini_cli.cli.tui.widgets import MenuScreen

    app = GeminiApp(FakeRuntime())
    results: list[object] = []

    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(
            MenuScreen(role_options(), "Roles:"),
            callback=results.append,
        )
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert results == [None]


async def test_tui_help_lists_commands() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "h", "e", "l", "p", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any("Commands:" in line for line in app.transcript)
    assert any("/quota" in line for line in app.transcript)


async def test_tui_unknown_command_is_reported() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "n", "o", "p", "e", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any("Unknown command /nope" in line for line in app.transcript)


async def test_tui_role_menu_calls_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())
    calls: list[tuple[str | None, str | None]] = []

    async def fake_rebuild(
        *,
        model: str | None = None,
        role: str | None = None,
    ) -> None:
        calls.append((model, role))

    monkeypatch.setattr(app, "_rebuild", fake_rebuild)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "r", "o", "l", "e", "enter")
        await pilot.pause()
        await pilot.press("down", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert calls == [(None, "code")]


async def test_tui_clear_resets_conversation() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "c", "l", "e", "a", "r", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any("Conversation cleared" in line for line in app.transcript)


async def test_tui_auth_menu_points_to_login() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "a", "u", "t", "h", "enter")
        await pilot.pause()
        await pilot.press("down", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any("gemini-cli-py login" in line for line in app.transcript)


async def test_tui_reset_removes_the_stored_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    removed: list[bool] = []
    monkeypatch.setattr(
        "gemini_cli.cli.tui.app.delete_api_key",
        lambda: removed.append(True),
    )

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "r", "e", "s", "e", "t", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert removed == [True]
    assert any("Stored API key removed" in line for line in app.transcript)


def test_render_quota_text_contains_bars_and_percentages() -> None:
    from gemini_cli.cli.tui.widgets import render_quota_text

    rendered = render_quota_text(
        FakeQuota().get_5h_status(),
        FakeQuota().get_weekly_status(),
    )

    assert "QUOTA" in rendered.plain
    assert "50.0%" in rendered.plain
    assert "31.0%" in rendered.plain
    assert "2h 00m remaining" in rendered.plain
    assert "█" in rendered.plain
    assert "░" in rendered.plain


async def test_tui_quota_panel_toggles_with_slash_and_escape() -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from gemini_cli.cli.tui.widgets import QuotaPanel

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "q", "u", "o", "t", "a", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        panel = app.query_one(QuotaPanel)
        assert panel.display is True
        assert "QUOTA" in panel.plain

        await pilot.press("escape")
        await pilot.pause()
        assert panel.display is False


async def test_tui_theme_toggles_with_slash() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        assert app._theme.name == "dark"
        await pilot.press("slash", "t", "h", "e", "m", "e", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._theme.name == "light"
        assert any("Theme: light" in line for line in app.transcript)

        await pilot.press("slash", "t", "h", "e", "m", "e", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._theme.name == "dark"


def test_menu_option_defaults() -> None:
    option = MenuOption("value", "Label")

    assert option.description == ""
