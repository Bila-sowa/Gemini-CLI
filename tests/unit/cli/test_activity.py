"""Tests for the inline activity log in assistant messages."""

from __future__ import annotations

from textual.app import App, ComposeResult

from gemini_cli.cli.tui.activity import ActivityLine, AgentMessageView
from gemini_cli.cli.tui.markdown import MarkdownMessageView
from gemini_cli.cli.tui.themes import DEFAULT_THEME, THEMES, to_textual_theme
from gemini_cli.core.streaming import AgentEvent, AgentEventType
from gemini_cli.llm import ToolCallRequest
from gemini_cli.tools.base import ToolResult


class ActivityApp(App[None]):
    """Minimal app that hosts one activity widget."""

    def __init__(self, widget: ActivityLine | AgentMessageView) -> None:
        self._widget = widget
        super().__init__()
        for theme in THEMES.values():
            self.register_theme(to_textual_theme(theme))
        self.theme = f"gemini-{DEFAULT_THEME.name}"

    def compose(self) -> ComposeResult:
        yield self._widget


async def test_thought_line_shows_duration_and_details() -> None:
    line = ActivityLine(summary="thought", kind="thought", icon="🧠")
    app = ActivityApp(line)

    async with app.run_test(size=(80, 24)) as pilot:
        line.append_text("first")
        line.append_text(" second")
        await pilot.pause()

        assert line.collapsed is True
        assert line.title.startswith("🧠 thought • ")
        assert "⏳" in line.title
        assert "first second" in line.plain

        line.mark_done()
        assert line.title.endswith("✅")
        assert line.status == "done"

        line.collapsed = False
        await pilot.pause()
        assert line.collapsed is False


async def test_tool_line_shows_status_and_output() -> None:
    line = ActivityLine(
        summary="create notes.md",
        kind="tool",
        icon="📝",
        details="notes.md",
    )
    app = ActivityApp(line)

    async with app.run_test(size=(80, 24)) as pilot:
        line.finish_tool(ToolResult(success=True, output="created"), "success")
        await pilot.pause()

        assert line.title == "📝 create notes.md ✅"
        assert not line.has_class("-error")
        assert "success" in line.plain
        assert "created" in line.plain


async def test_tool_error_is_visible_in_the_title() -> None:
    line = ActivityLine(summary="create notes.md", kind="tool", icon="📝")
    app = ActivityApp(line)

    async with app.run_test(size=(80, 24)) as pilot:
        line.finish_tool(
            ToolResult(success=False, output="", error="permission denied"),
            "error",
        )
        await pilot.pause()

        assert line.title == "📝 create notes.md ❌"
        assert line.has_class("-error")
        assert "permission denied" in line.plain


async def test_denied_tool_is_red_and_marked() -> None:
    line = ActivityLine(summary="run: rm -rf /", kind="tool", icon="💻")
    app = ActivityApp(line)

    async with app.run_test(size=(80, 24)) as pilot:
        line.finish_tool(
            ToolResult(success=False, output="", error="denied"),
            "denied",
        )
        await pilot.pause()

        assert line.title == "💻 run: rm -rf / 🚫"
        assert line.has_class("-error")


async def test_agent_message_interleaves_text_and_activity() -> None:
    view = AgentMessageView("✦: ", DEFAULT_THEME.assistant_color, theme=DEFAULT_THEME)
    app = ActivityApp(view)
    call = ToolCallRequest(
        id="call_1",
        name="fetch_url",
        arguments={"url": "https://example.com"},
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await view.append_text("before ")
        await view.append_thought("thinking")
        await view.append_text("after ")
        await view.start_tool(call, {"tool": "fetch_url", "call_id": "call_1"})
        view.finish_tool(
            ToolResult(success=True, output="result"),
            {"call_id": "call_1", "status": "success"},
        )
        await view.finish()
        await pilot.pause()

        children = list(view._body.children)
        assert [type(child).__name__ for child in children] == [
            "MarkdownMessageView",
            "ActivityLine",
            "MarkdownMessageView",
            "ActivityLine",
        ]
        assert "before" in view.plain
        assert "thinking" in view.plain
        assert "after" in view.plain
        assert "search: https://example.com" in view.plain


async def test_tool_summaries_are_human_readable() -> None:
    view = AgentMessageView("✦: ", DEFAULT_THEME.assistant_color, theme=DEFAULT_THEME)
    app = ActivityApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        await view.start_tool(
            ToolCallRequest(
                id="write",
                name="write_file",
                arguments={"path": "wheater.md"},
            ),
            {"tool": "write_file", "call_id": "write"},
        )
        await view.start_tool(
            ToolCallRequest(
                id="run",
                name="run_command",
                arguments={"command": "ls -la"},
            ),
            {"tool": "run_command", "call_id": "run"},
        )
        await pilot.pause()

        lines = list(view.query(ActivityLine))
        assert [line.summary for line in lines] == [
            "create wheater.md",
            "run: ls -la",
        ]
        assert [line.title for line in lines] == [
            "📝 create wheater.md ⏳",
            "💻 run: ls -la ⏳",
        ]


async def test_parallel_tools_are_matched_by_call_id() -> None:
    view = AgentMessageView("✦: ", DEFAULT_THEME.assistant_color, theme=DEFAULT_THEME)
    app = ActivityApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        await view.start_tool(
            ToolCallRequest(id="first", name="read_file", arguments={"path": "a.txt"}),
            {"tool": "read_file", "call_id": "first"},
        )
        await view.start_tool(
            ToolCallRequest(
                id="second", name="fetch_url", arguments={"url": "https://x"}
            ),
            {"tool": "fetch_url", "call_id": "second"},
        )
        view.finish_tool(
            ToolResult(success=True, output="A"),
            {"call_id": "second", "status": "success"},
        )
        view.finish_tool(
            ToolResult(success=False, output="", error="missing"),
            {"call_id": "first", "status": "error"},
        )
        await pilot.pause()

        lines = list(view.query(ActivityLine))
        assert [line.status for line in lines] == ["error", "success"]
        assert "missing" in lines[0].plain
        assert "https://x" in lines[1].plain


async def test_long_output_is_truncated() -> None:
    view = AgentMessageView("✦: ", DEFAULT_THEME.assistant_color, theme=DEFAULT_THEME)
    app = ActivityApp(view)
    output = "\n".join(f"line {index}" for index in range(100))

    async with app.run_test(size=(80, 24)) as pilot:
        await view.start_tool(
            ToolCallRequest(id="run", name="run_command", arguments={"command": "x"}),
            {"tool": "run_command", "call_id": "run"},
        )
        view.finish_tool(
            ToolResult(success=True, output=output),
            {"call_id": "run", "status": "success"},
        )
        await pilot.pause()

        line = view.query_one(ActivityLine)
        assert "line 0" in line.plain
        assert "line 99" not in line.plain
        assert "… 70 more lines" in line.plain


async def test_stopped_activity_is_marked() -> None:
    view = AgentMessageView("✦: ", DEFAULT_THEME.assistant_color, theme=DEFAULT_THEME)
    app = ActivityApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        await view.start_tool(
            ToolCallRequest(id="run", name="run_command", arguments={"command": "x"}),
            {"tool": "run_command", "call_id": "run"},
        )
        view.mark_stopped()
        await pilot.pause()

        line = view.query_one(ActivityLine)
        assert line.status == "stopped"
        assert line.title == "💻 run: x ⏹️"
        assert not line.has_class("-error")


async def test_tui_renders_activity_inside_the_assistant_message() -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from tests.unit.cli.test_repl import FakeRuntime

    events = [
        AgentEvent(AgentEventType.THINKING, "planning"),
        AgentEvent(
            AgentEventType.TOOL_CALL_START,
            ToolCallRequest(
                id="call_1",
                name="fetch_url",
                arguments={"url": "https://example.com"},
            ),
            {"tool": "fetch_url", "call_id": "call_1"},
        ),
        AgentEvent(
            AgentEventType.TOOL_CALL_END,
            ToolResult(success=True, output="result"),
            {"tool": "fetch_url", "call_id": "call_1", "status": "success"},
        ),
        AgentEvent(AgentEventType.TEXT_CHUNK, "Here is the result."),
        AgentEvent(AgentEventType.DONE),
    ]
    app = GeminiApp(FakeRuntime(events))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("h", "i", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        view = app.query_one(AgentMessageView)
        children = list(view._body.children)
        assert [type(child).__name__ for child in children] == [
            "ActivityLine",
            "ActivityLine",
            "MarkdownMessageView",
        ]
        assert "planning" in view.plain
        assert "search: https://example.com" in view.plain
        assert "Here is the result." in app.transcript[-1]


async def test_text_only_turn_has_no_activity_lines() -> None:
    view = AgentMessageView("✦: ", DEFAULT_THEME.assistant_color, theme=DEFAULT_THEME)
    app = ActivityApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        await view.append_text("hello")
        await view.finish()
        await pilot.pause()

        assert list(view.query(ActivityLine)) == []
        assert len(list(view.query(MarkdownMessageView))) == 1
