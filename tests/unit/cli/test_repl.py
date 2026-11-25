"""Tests for the terminal UI and the plain line fallback."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path

import pytest

from gemini_cli.cli.tui import repl as repl_module
from gemini_cli.cli.tui.repl import (
    ConsoleApprovalCallback,
    MenuOption,
    model_options,
    role_options,
    run_repl,
    theme_options,
)
from gemini_cli.config import Settings
from gemini_cli.config.env_file import EnvFileError
from gemini_cli.core.context_tracker import ContextWindowStatus
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEvent, AgentEventType
from gemini_cli.sandbox.approval import ApprovalDecision


class FakeContext:
    """Deterministic context-window tracker."""

    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def status(self) -> ContextWindowStatus:
        return ContextWindowStatus(
            used_tokens=130_000,
            window_tokens=1_048_576,
            budget_tokens=1_040_384,
            reserved_output_tokens=8_192,
            percentage=12.5,
            remaining_tokens=910_384,
        )


class FakeAgent:
    """Yield a scripted event list for every turn."""

    def __init__(self, events: Sequence[AgentEvent] = ()) -> None:
        self.events = list(events)
        self.calls: list[tuple[Session, str]] = []

    def refresh_context(self, session: Session) -> None:
        del session

    async def run(
        self,
        session: Session,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        self.calls.append((session, user_message))
        for event in self.events:
            yield event


class GatedAgent:
    """Yield scripted events, pausing on ``asyncio.Event`` gates."""

    def __init__(self, steps: Sequence[AgentEvent | asyncio.Event]) -> None:
        self.steps = list(steps)

    def refresh_context(self, session: Session) -> None:
        del session

    async def run(
        self,
        session: Session,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        del session, user_message
        for step in self.steps:
            if isinstance(step, asyncio.Event):
                await step.wait()
            else:
                yield step


class FakeIndicator:
    """Record indicator changes without drawing anything."""

    def __init__(self, **_kwargs: object) -> None:
        self.states: list[bool] = []
        self.closed = False

    def set_waiting(self, waiting: bool) -> None:
        self.states.append(waiting)

    def close(self) -> None:
        self.closed = True


class FakeRuntime:
    """Minimal runtime surface used by both interfaces."""

    def __init__(
        self,
        events: Sequence[AgentEvent] = (),
        *,
        model: str = "gemini-3.8-flash",
        role: str = "basic",
    ) -> None:
        self.settings = Settings(_env_file=None, model=model, role=role)
        self.agent = FakeAgent(events)
        self.context_window = FakeContext()
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


async def test_repl_turn_toggles_the_indicator_around_model_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(
        [
            AgentEvent(AgentEventType.TOOL_CALL_START),
            AgentEvent(AgentEventType.TOOL_CALL_END),
            AgentEvent(AgentEventType.TEXT_CHUNK, "Done"),
            AgentEvent(AgentEventType.DONE),
        ]
    )
    indicator = FakeIndicator()
    monkeypatch.setattr(
        repl_module,
        "RichWaitIndicator",
        lambda **_kwargs: indicator,
    )
    writes: list[tuple[str, list[bool]]] = []

    def write(text: str) -> None:
        writes.append((text, list(indicator.states)))

    await repl_module._run_turn(
        runtime,
        Session(),
        "go",
        write,
        show_indicator=False,
    )

    assert indicator.states == [True, False, False, True, False, False]
    assert indicator.closed is True
    assert "Done" in "".join(text for text, _states in writes)
    done_states = next(states for text, states in writes if text == "Done")
    assert done_states[-1] is False


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


def test_model_options_include_an_unregistered_current_model() -> None:
    options = model_options("my-custom-model")

    assert options[-1].value == "my-custom-model"
    assert "current custom model" in options[-1].label


def test_theme_options_cover_available_themes() -> None:
    options = theme_options()

    assert {option.value for option in options} == {"dark", "light"}
    assert all(option.description for option in options)


def test_role_and_model_options_cover_the_registry() -> None:
    assert {option.value for option in role_options()} == {
        "basic",
        "code",
        "creator",
        "analyser",
        "poet",
    }
    assert {option.value for option in model_options()} == {
        "gemini-4-argon",
        "gemini-4-carbon",
        "gemini-4-neon",
        "gemini-4-hydrogen",
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
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
        assert "gemini-3.8-flash" in app.query_one(StatusBar).plain
        assert "context: 130k/1.0M (12.5%)" in app.query_one(StatusBar).plain
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


async def test_tui_waiting_indicator_tracks_only_model_requests() -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from gemini_cli.cli.tui.widgets import WaitingIndicator

    initial = asyncio.Event()
    tool_running = asyncio.Event()
    next_model = asyncio.Event()
    runtime = FakeRuntime()
    runtime.agent = GatedAgent(  # type: ignore[assignment]
        [
            initial,
            AgentEvent(AgentEventType.TEXT_CHUNK, "Hi"),
            AgentEvent(AgentEventType.TOOL_CALL_START),
            tool_running,
            AgentEvent(AgentEventType.TOOL_CALL_END),
            next_model,
            AgentEvent(AgentEventType.TEXT_CHUNK, "!"),
            AgentEvent(AgentEventType.DONE),
        ]
    )
    app = GeminiApp(runtime, waiting_delay=0)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("h", "i", "enter")
        await pilot.pause(0.1)

        indicator = app.query_one(WaitingIndicator)
        assert indicator.display is True

        initial.set()
        await pilot.pause(0.1)
        indicator = app.query_one(WaitingIndicator)
        assert indicator.display is False

        tool_running.set()
        await pilot.pause(0.1)
        indicator = app.query_one(WaitingIndicator)
        assert indicator.display is True

        next_model.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert list(app.query(WaitingIndicator)) == []
    assert any(line == "✦: Hi!" for line in app.transcript)


async def test_tui_escape_stops_an_active_model_request() -> None:
    from textual.widgets import Input

    from gemini_cli.cli.tui.app import (
        DEFAULT_PROMPT_PLACEHOLDER,
        STOP_PROMPT_PLACEHOLDER,
        GeminiApp,
    )

    gate = asyncio.Event()
    runtime = FakeRuntime()
    runtime.agent = GatedAgent(  # type: ignore[assignment]
        [
            gate,
            AgentEvent(AgentEventType.TEXT_CHUNK, "partial"),
            AgentEvent(AgentEventType.DONE),
        ]
    )
    app = GeminiApp(runtime, waiting_delay=0)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("h", "i", "enter")
        await pilot.pause(0.1)
        prompt = app.query_one("#prompt", Input)
        assert prompt.placeholder == STOP_PROMPT_PLACEHOLDER

        await pilot.press("escape")
        await pilot.pause(0.1)
        await app.workers.wait_for_complete()

        assert prompt.disabled is False
        assert prompt.placeholder == DEFAULT_PROMPT_PLACEHOLDER
        assert any("Generation stopped." in line for line in app.transcript)


async def test_tui_escape_during_tools_stops_after_the_batch() -> None:
    from textual.widgets import Input

    from gemini_cli.cli.tui.app import (
        DEFAULT_PROMPT_PLACEHOLDER,
        STOPPING_TOOLS_PROMPT_PLACEHOLDER,
        GeminiApp,
    )

    tools_done = asyncio.Event()
    next_request = asyncio.Event()
    runtime = FakeRuntime()
    runtime.agent = GatedAgent(  # type: ignore[assignment]
        [
            AgentEvent(AgentEventType.TOOL_CALL_START),
            tools_done,
            AgentEvent(AgentEventType.TOOL_CALL_END),
            next_request,
            AgentEvent(AgentEventType.DONE),
        ]
    )
    app = GeminiApp(runtime, waiting_delay=0)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("h", "i", "enter")
        await pilot.pause(0.1)

        await pilot.press("escape")
        await pilot.pause(0.1)
        prompt = app.query_one("#prompt", Input)
        assert prompt.placeholder == STOPPING_TOOLS_PROMPT_PLACEHOLDER
        assert prompt.disabled is True

        tools_done.set()
        await pilot.pause(0.1)
        await app.workers.wait_for_complete()

        assert prompt.disabled is False
        assert prompt.placeholder == DEFAULT_PROMPT_PLACEHOLDER
        assert any("Generation stopped." in line for line in app.transcript)


async def test_tui_waiting_indicator_animates_frames() -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from gemini_cli.cli.tui.widgets import WaitingIndicator

    app = GeminiApp(FakeRuntime(), waiting_delay=0)

    async with app.run_test(size=(120, 40)) as pilot:
        indicator = WaitingIndicator(show_delay=0)
        await app.query_one("#conversation").mount(indicator)
        indicator.show()
        await pilot.pause()

        first = indicator.plain
        await pilot.pause(0.25)

        assert indicator.plain != first

        indicator.hide()


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
    assert any("/context" in line for line in app.transcript)
    assert any("/models" in line for line in app.transcript)


async def test_tui_unknown_command_is_reported() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "n", "o", "p", "e", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any("Unknown command /nope" in line for line in app.transcript)


async def test_tui_unknown_command_suggests_a_close_match() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "m", "o", "d", "l", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any("Did you mean /model?" in line for line in app.transcript)


async def test_tui_ambiguous_command_is_reported() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "c", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any(
        "ambiguous" in line and "/clear" in line and "/context" in line
        for line in app.transcript
    )


@pytest.mark.parametrize("command", ["models", "roles", "authentication"])
async def test_tui_alias_opens_a_menu(command: str) -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from gemini_cli.cli.tui.widgets import MenuScreen

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", *command, "enter")
        await pilot.pause()
        assert isinstance(app.screen, MenuScreen)
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.parametrize("command", ["q", "quit", "bye"])
async def test_tui_exit_alias_quits(command: str) -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", *command, "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert app.return_code == 0


async def test_tui_is_slash_only_for_exit() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    runtime = FakeRuntime()
    app = GeminiApp(runtime)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("e", "x", "i", "t", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert [message for _session, message in runtime.agent.calls] == ["exit"]


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


async def test_tui_model_menu_persists_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    app = GeminiApp(FakeRuntime())
    saved: list[tuple[Path, str, str]] = []

    async def fake_rebuild(
        *,
        model: str | None = None,
        role: str | None = None,
    ) -> bool:
        del model, role
        return True

    def fake_set_env(path: Path, key: str, value: str) -> None:
        saved.append((path, key, value))

    monkeypatch.setattr(app, "_rebuild", fake_rebuild)
    monkeypatch.setattr("gemini_cli.cli.tui.app.set_env_value", fake_set_env)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "m", "o", "d", "e", "l", "enter")
        await pilot.pause()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert saved == [(tmp_path / ".env", "GEMINI_MODEL", "gemini-4-argon")]
    assert any("Saved GEMINI_MODEL" in line for line in app.transcript)


async def test_tui_role_menu_persists_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_ROLE", raising=False)
    app = GeminiApp(FakeRuntime())
    saved: list[tuple[Path, str, str]] = []

    async def fake_rebuild(
        *,
        model: str | None = None,
        role: str | None = None,
    ) -> bool:
        del model, role
        return True

    def fake_set_env(path: Path, key: str, value: str) -> None:
        saved.append((path, key, value))

    monkeypatch.setattr(app, "_rebuild", fake_rebuild)
    monkeypatch.setattr("gemini_cli.cli.tui.app.set_env_value", fake_set_env)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "r", "o", "l", "e", "enter")
        await pilot.pause()
        await pilot.press("down", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert saved == [(tmp_path / ".env", "GEMINI_ROLE", "code")]
    assert any("Saved GEMINI_ROLE" in line for line in app.transcript)


async def test_tui_model_argument_rebuilds_and_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    app = GeminiApp(FakeRuntime())
    calls: list[tuple[str | None, str | None]] = []

    async def fake_rebuild(
        *,
        model: str | None = None,
        role: str | None = None,
    ) -> bool:
        calls.append((model, role))
        return True

    monkeypatch.setattr(app, "_rebuild", fake_rebuild)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", *"models", "space", *"gemini-4-argon", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert calls == [("gemini-4-argon", None)]
    assert any("Model: gemini-4-argon" in line for line in app.transcript)
    assert "GEMINI_MODEL=gemini-4-argon" in (tmp_path / ".env").read_text()


async def test_tui_model_argument_suggests_a_close_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())
    calls: list[tuple[str | None, str | None]] = []

    async def fake_rebuild(
        *,
        model: str | None = None,
        role: str | None = None,
    ) -> bool:
        calls.append((model, role))
        return True

    monkeypatch.setattr(app, "_rebuild", fake_rebuild)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", *"model", "space", *"gemini-3.5-flash-lit", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert calls == []
    assert any(
        "Did you mean 'gemini-3.5-flash-lite'?" in line for line in app.transcript
    )


async def test_tui_role_argument_rebuilds_and_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_ROLE", raising=False)
    app = GeminiApp(FakeRuntime())
    calls: list[tuple[str | None, str | None]] = []

    async def fake_rebuild(
        *,
        model: str | None = None,
        role: str | None = None,
    ) -> bool:
        calls.append((model, role))
        return True

    monkeypatch.setattr(app, "_rebuild", fake_rebuild)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", *"role", "space", *"code", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert calls == [(None, "code")]
    assert any("Role: code" in line for line in app.transcript)
    assert "GEMINI_ROLE=code" in (tmp_path / ".env").read_text()


async def test_tui_help_command_shows_argument_details() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", *"help", "space", *"model", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    text = "\n".join(app.transcript)
    assert "/model [name]" in text
    assert "Aliases: /models, /m" in text


async def test_tui_command_without_arguments_rejects_extra_text() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", *"clear", "space", *"extra", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any("/clear does not take arguments." in line for line in app.transcript)
    assert not any("Conversation cleared." in line for line in app.transcript)


async def test_tui_model_menu_reports_env_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    app = GeminiApp(FakeRuntime())

    async def fake_rebuild(
        *,
        model: str | None = None,
        role: str | None = None,
    ) -> bool:
        del model, role
        return True

    def failing_set_env(path: Path, key: str, value: str) -> None:
        del path, key, value
        raise EnvFileError("read-only file")

    monkeypatch.setattr(app, "_rebuild", fake_rebuild)
    monkeypatch.setattr("gemini_cli.cli.tui.app.set_env_value", failing_set_env)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "m", "o", "d", "e", "l", "enter")
        await pilot.pause()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any("Could not save GEMINI_MODEL" in line for line in app.transcript)


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


def test_render_context_text_contains_bars_and_percentages() -> None:
    from gemini_cli.cli.tui.widgets import render_context_text

    rendered = render_context_text(FakeContext().status(), "gemini-3.8-flash")

    assert "CONTEXT" in rendered.plain
    assert "gemini-3.8-flash" in rendered.plain
    assert "12.5%" in rendered.plain
    assert "1.0M tokens" in rendered.plain
    assert "█" in rendered.plain
    assert "░" in rendered.plain


def test_render_context_text_uses_theme_for_empty_bar() -> None:
    from gemini_cli.cli.tui.themes import LIGHT_THEME
    from gemini_cli.cli.tui.widgets import render_context_text

    rendered = render_context_text(
        FakeContext().status(),
        "gemini-3.8-flash",
        LIGHT_THEME,
    )

    assert LIGHT_THEME.border in {str(span.style) for span in rendered.spans}


async def test_tui_context_panel_toggles_with_slash_and_escape() -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from gemini_cli.cli.tui.widgets import ContextPanel

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "c", "o", "n", "t", "e", "x", "t", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        panel = app.query_one(ContextPanel)
        assert panel.display is True
        assert "CONTEXT" in panel.plain

        await pilot.press("escape")
        await pilot.pause()
        assert panel.display is False


async def test_tui_theme_menu_selects_light_and_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_THEME", raising=False)
    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        assert app._theme.name == "dark"
        await pilot.press("slash", "t", "h", "e", "m", "e", "enter")
        await pilot.pause()
        await pilot.press("down", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert app._theme.name == "light"
    assert any("Theme: light" in line for line in app.transcript)
    assert "GEMINI_THEME=light" in (tmp_path / ".env").read_text()


async def test_tui_theme_menu_cancel_keeps_theme() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", "t", "h", "e", "m", "e", "enter")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert app._theme.name == "dark"


async def test_tui_theme_argument_sets_theme_and_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_THEME", raising=False)
    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", *"theme", "space", *"light", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert app._theme.name == "light"
    assert any("Theme: light" in line for line in app.transcript)
    assert "GEMINI_THEME=light" in (tmp_path / ".env").read_text()


async def test_tui_theme_argument_suggests_a_close_match() -> None:
    from gemini_cli.cli.tui.app import GeminiApp

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash", *"theme", "space", *"lite", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert any("Did you mean 'light'?" in line for line in app.transcript)
    assert app._theme.name == "dark"


async def test_tui_theme_switch_restyles_existing_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from gemini_cli.cli.tui.markdown import MarkdownMessageView
    from gemini_cli.cli.tui.themes import DEFAULT_THEME, LIGHT_THEME
    from gemini_cli.cli.tui.widgets import MessageView

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_THEME", raising=False)
    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("h", "i", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        app._write_info("old info")
        user_view = next(
            view for view in app.query(MarkdownMessageView) if view._prefix == "User: "
        )
        info_view = list(app.query(MessageView))[-1]
        assert user_view._prefix_style == f"bold {DEFAULT_THEME.user_color}"
        assert info_view._text.style == DEFAULT_THEME.muted

        await app._set_theme("light")
        await pilot.pause()

        assert user_view._prefix_style == f"bold {LIGHT_THEME.user_color}"
        assert info_view._text.style == LIGHT_THEME.muted


async def test_tui_light_theme_styles_modals() -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from gemini_cli.cli.tui.repl import theme_options
    from gemini_cli.cli.tui.themes import LIGHT_THEME
    from gemini_cli.cli.tui.widgets import ApprovalScreen, MenuScreen

    app = GeminiApp(FakeRuntime())

    async with app.run_test(size=(120, 40)) as pilot:
        app._theme = LIGHT_THEME
        app._apply_theme()
        await pilot.pause()

        app.push_screen(MenuScreen(theme_options(), "Themes:", app._theme))
        await pilot.pause()
        assert (
            app.screen.styles.background.hex.lower() == LIGHT_THEME.background.lower()
        )
        menu = app.screen.query_one("#menu")
        assert menu.styles.background.hex.lower() == LIGHT_THEME.surface.lower()
        await pilot.press("escape")
        await pilot.pause()

        app.push_screen(
            ApprovalScreen("run a tool", "write", "path: a", theme=app._theme)
        )
        await pilot.pause()
        assert (
            app.screen.styles.background.hex.lower() == LIGHT_THEME.background.lower()
        )
        approval = app.screen.query_one("#approval")
        assert approval.styles.background.hex.lower() == LIGHT_THEME.surface.lower()
        await pilot.press("escape")


async def test_run_tui_uses_the_configured_theme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gemini_cli.cli.tui import app as app_module

    captured: dict[str, object] = {}

    class FakeApp:
        def __init__(
            self,
            runtime: object,
            *,
            bridge: object = None,
            theme: object = None,
        ) -> None:
            captured["runtime"] = runtime
            captured["theme"] = theme

        async def run_async(self) -> None:
            return None

    monkeypatch.setattr(app_module, "GeminiApp", FakeApp)
    runtime = FakeRuntime()
    runtime.settings = Settings(_env_file=None, theme="light")

    await app_module.run_tui(runtime)

    theme = captured["theme"]
    assert getattr(theme, "name", None) == "light"


def test_menu_option_defaults() -> None:
    option = MenuOption("value", "Label")

    assert option.description == ""
