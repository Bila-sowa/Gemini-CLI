"""Inline activity log for the Textual terminal UI.

The activity is rendered inside the assistant message, interleaved with the
text the model streams. Each thought or tool call is a compact collapsible
line; expanding it reveals the details and bounded output.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Collapsible, Static

from gemini_cli.cli.tui.markdown import MarkdownMessageView, sanitize_markdown
from gemini_cli.cli.tui.themes import DEFAULT_THEME, Theme
from gemini_cli.llm import ToolCallRequest
from gemini_cli.tools.base import ToolResult

_PREVIEW_LINES = 30
_PREVIEW_CHARS = 4_000
_SUMMARY_CHARS = 90

_TOOL_ICONS: dict[str, str] = {
    "fetch_url": "🌐",
    "write_file": "📝",
    "edit_file": "✏️",
    "read_file": "📄",
    "run_command": "💻",
    "grep": "🔎",
    "find_files": "🔎",
    "code_search": "🔎",
}
_STATUS_ICONS: dict[str, str] = {
    "running": "⏳",
    "success": "✅",
    "done": "✅",
    "error": "❌",
    "denied": "🚫",
    "stopped": "⏹️",
}


def _tool_icon(name: str) -> str:
    return _TOOL_ICONS.get(name, "🔧")


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _preview(text: str) -> tuple[str, int]:
    """Return a bounded, sanitized preview and the number of hidden lines."""
    cleaned = sanitize_markdown(text).rstrip()
    if not cleaned:
        return "", 0
    lines = cleaned.splitlines()
    omitted = max(0, len(lines) - _PREVIEW_LINES)
    preview = "\n".join(lines[:_PREVIEW_LINES])
    if len(preview) > _PREVIEW_CHARS:
        preview = f"{preview[: _PREVIEW_CHARS - 1]}…"
    return preview, omitted


def _arguments_text(name: str, arguments: Mapping[str, Any]) -> str:
    if not arguments:
        return ""
    if name == "run_command":
        command = str(arguments.get("command", "")).strip()
        return f"$ {command}" if command else ""
    if name == "fetch_url":
        return str(arguments.get("url", ""))
    if name in {"read_file", "write_file", "edit_file"}:
        return str(arguments.get("path", ""))
    if name in {"grep", "find_files", "code_search"}:
        return str(arguments.get("pattern") or arguments.get("query") or "")
    try:
        rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        rendered = str(arguments)
    return _truncate(rendered, 300)


def _tool_summary(name: str, arguments: Mapping[str, Any]) -> str:
    if name == "fetch_url":
        summary = f"search: {arguments.get('url', '')}"
    elif name == "write_file":
        summary = f"create {arguments.get('path', '')}"
    elif name == "edit_file":
        summary = f"edit {arguments.get('path', '')}"
    elif name == "read_file":
        summary = f"read {arguments.get('path', '')}"
    elif name == "run_command":
        summary = f"run: {arguments.get('command', '')}"
    elif name in {"grep", "find_files", "code_search"}:
        pattern = arguments.get("pattern") or arguments.get("query") or ""
        summary = f"search: {pattern}"
    else:
        summary = f"tool: {name}"
    return _truncate(summary.strip(), _SUMMARY_CHARS)


class ActivityLine(Collapsible):
    """One collapsible thought, tool call, or error inside a message."""

    DEFAULT_CSS = """
    ActivityLine {
        width: 100%;
        height: auto;
        background: transparent;
        border: none;
        padding: 0;
        margin: 0 0 1 0;
    }
    ActivityLine CollapsibleTitle {
        background: transparent;
        color: $activity-muted;
        padding: 0;
    }
    ActivityLine.-error CollapsibleTitle {
        color: $activity-error;
        text-style: bold;
    }
    ActivityLine .activity-details {
        width: 100%;
        height: auto;
        max-height: 16;
        overflow-y: auto;
        padding: 0 0 0 2;
    }
    """

    def __init__(
        self,
        *,
        summary: str,
        theme: Theme = DEFAULT_THEME,
        kind: str = "tool",
        icon: str = "🔧",
        details: str = "",
        status: str = "running",
    ) -> None:
        self._theme = theme
        self._kind = kind
        self._icon = icon
        self._status = status
        self._summary = summary
        self._detail_text = sanitize_markdown(details).strip() if details else ""
        self._output = ""
        self._error = ""
        self._duration: float | None = None
        self._started_at = time.monotonic()
        self._details_widget = Static("", classes="activity-details")
        super().__init__(
            self._details_widget,
            title=summary,
            collapsed=True,
            collapsed_symbol=">",
            expanded_symbol="v",
        )
        self._refresh_title()

    def on_mount(self) -> None:
        self._update_error_class()
        self._refresh_details()

    @property
    def status(self) -> str:
        return self._status

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def plain(self) -> str:
        content = self._details_widget.content
        details = content.plain if isinstance(content, Text) else str(content)
        return f"{self.title}\n{details}"

    def append_text(self, text: str) -> None:
        """Append streamed thought text and update the live duration."""
        cleaned = sanitize_markdown(text)
        if not cleaned:
            return
        self._detail_text = (
            f"{self._detail_text}{cleaned}" if self._detail_text else cleaned
        )
        self._refresh_title()
        self._refresh_details()

    def mark_done(self) -> None:
        """Mark a thought block as complete."""
        if self._status != "running":
            return
        self._status = "done"
        self._duration = self._elapsed()
        self._refresh_title()
        self._refresh_details()

    def finish_tool(self, result: ToolResult | None, status: str) -> None:
        """Attach a tool result and final status."""
        self._status = status
        self._duration = self._elapsed()
        if result is None:
            result = ToolResult(success=True, output="")
        if result.success:
            self._output = result.output
        else:
            self._error = result.error or result.output
        self._update_error_class()
        self._refresh_title()
        self._refresh_details()

    def mark_stopped(self) -> None:
        """Mark a still-running line as interrupted."""
        if self._status != "running":
            return
        self._status = "stopped"
        self._duration = self._elapsed()
        self._update_error_class()
        self._refresh_title()
        self._refresh_details()

    def apply_theme(self, theme: Theme) -> None:
        """Update the detail colors after a theme change."""
        self._theme = theme
        self._refresh_details()

    def _elapsed(self) -> float:
        return time.monotonic() - self._started_at

    def _refresh_title(self) -> None:
        if self._kind == "error":
            self.title = f"{self._icon} error"
            return
        status_icon = _STATUS_ICONS.get(self._status, "")
        if self._kind == "thought":
            duration = self._elapsed()
            self.title = f"{self._icon} thought • {duration:.1f}s {status_icon}".strip()
            return
        self.title = f"{self._icon} {self._summary} {status_icon}".strip()

    def _update_error_class(self) -> None:
        self.set_class(self._status in {"error", "denied"}, "-error")

    def _refresh_details(self) -> None:
        text = Text()
        if self._kind == "thought":
            if self._detail_text:
                text.append(self._detail_text, style=self._theme.muted)
        else:
            header = self._status
            if self._duration is not None:
                header = f"{header} • {self._duration:.1f}s"
            header_style = (
                self._theme.error_color
                if self._status in {"error", "denied"}
                else self._theme.muted
            )
            text.append(header, style=header_style)
            if self._detail_text:
                text.append("\n")
                text.append(self._detail_text, style=self._theme.muted)
            if self._output:
                preview, omitted = _preview(self._output)
                text.append("\n")
                text.append(preview, style=self._theme.foreground)
                if omitted:
                    text.append(
                        f"\n… {omitted} more lines",
                        style=self._theme.muted,
                    )
            if self._error:
                text.append("\n")
                text.append(self._error, style=self._theme.error_color)
        self._details_widget.update(text)


class AgentMessageView(Vertical):
    """Assistant turn with interleaved text and activity lines."""

    DEFAULT_CSS = """
    AgentMessageView {
        width: 100%;
        height: auto;
    }
    AgentMessageView .agent-message-row {
        width: 100%;
        height: auto;
    }
    AgentMessageView .agent-message-prefix {
        width: auto;
        height: 1;
        padding-right: 1;
    }
    AgentMessageView .agent-body {
        width: 1fr;
        height: auto;
    }
    """

    def __init__(
        self,
        prefix: str,
        prefix_style: str,
        *,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        super().__init__(classes="message agent-message")
        self._prefix = prefix
        self._prefix_style = prefix_style
        self._theme = theme
        self._body = Vertical(classes="agent-body")
        self._segments: list[MarkdownMessageView] = []
        self._lines: list[ActivityLine] = []
        self._tools: dict[str, ActivityLine] = {}
        self._current_text: MarkdownMessageView | None = None
        self._current_thought: ActivityLine | None = None
        self._last_call_id: str | None = None
        self._errors: list[str] = []
        self._finished = False

    def compose(self) -> ComposeResult:
        with Horizontal(classes="agent-message-row"):
            yield Static(
                Text(self._prefix, style=self._prefix_style),
                classes="agent-message-prefix",
            )
            yield self._body

    async def append_text(self, chunk: str) -> None:
        """Append model text, starting a new Markdown segment if needed."""
        if not chunk:
            return
        self._finish_thought()
        if self._current_text is None:
            segment = MarkdownMessageView(
                "",
                "",
                streaming=True,
                role="assistant",
            )
            await self._body.mount(segment)
            self._segments.append(segment)
            self._current_text = segment
        await self._current_text.append(chunk)

    async def append_thought(self, text: str) -> None:
        """Append a thought, finishing any open text segment first."""
        if not text:
            return
        await self._finish_text()
        if self._current_thought is None:
            line = ActivityLine(
                summary="thought",
                theme=self._theme,
                kind="thought",
                icon="🧠",
            )
            await self._body.mount(line)
            self._lines.append(line)
            self._current_thought = line
        self._current_thought.append_text(text)

    async def start_tool(
        self,
        call: ToolCallRequest | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Add a running tool line."""
        await self._finish_text()
        self._finish_thought()
        meta = metadata or {}
        call_id = str(
            meta.get("call_id") or (call.id if call else f"call-{len(self._tools) + 1}")
        )
        name = str(meta.get("tool") or (call.name if call else "tool"))
        arguments = call.arguments if call else {}
        line = ActivityLine(
            summary=_tool_summary(name, arguments),
            theme=self._theme,
            kind="tool",
            icon=_tool_icon(name),
            details=_arguments_text(name, arguments),
        )
        await self._body.mount(line)
        self._lines.append(line)
        self._tools[call_id] = line
        self._last_call_id = call_id

    def finish_tool(
        self,
        result: ToolResult | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach a result to the matching tool line."""
        meta = metadata or {}
        call_id = str(meta.get("call_id") or self._last_call_id or "")
        line = self._tools.get(call_id)
        if line is None and self._last_call_id is not None:
            line = self._tools.get(self._last_call_id)
        if line is None:
            return
        if result is None:
            result = ToolResult(success=True, output="")
        status = str(meta.get("status") or ("success" if result.success else "error"))
        line.finish_tool(result, status)

    async def add_error(self, message: str) -> None:
        """Add an error line and keep the transcript text compatible."""
        await self._finish_text()
        self._finish_thought()
        self._errors.append(message)
        line = ActivityLine(
            summary="error",
            theme=self._theme,
            kind="error",
            icon="❌",
            details=message,
            status="error",
        )
        await self._body.mount(line)
        self._lines.append(line)

    async def finish(self) -> None:
        """Flush the current text segment and close the open thought."""
        if self._finished:
            return
        self._finished = True
        await self._finish_text()
        self._finish_thought()

    def mark_stopped(self) -> None:
        """Mark every running line as stopped."""
        for line in self._lines:
            line.mark_stopped()

    def apply_theme(self, theme: Theme) -> None:
        """Re-style the prefix, text segments, and activity lines."""
        self._theme = theme
        self._prefix_style = theme.assistant_color
        try:
            prefix = self.query_one(".agent-message-prefix", Static)
        except NoMatches:
            prefix = None
        if prefix is not None:
            prefix.update(Text(self._prefix, style=self._prefix_style))
        for segment in self._segments:
            segment.apply_theme(theme)
        for line in self._lines:
            line.apply_theme(theme)

    async def rerender(self) -> None:
        """Re-render text segments after a theme change."""
        for segment in self._segments:
            await segment.rerender()

    @property
    def plain(self) -> str:
        parts = [self._prefix]
        parts.extend(segment.plain for segment in self._segments)
        parts.extend(line.plain for line in self._lines)
        parts.extend(f"\nError: {error}" for error in self._errors)
        return "".join(parts)

    async def _finish_text(self) -> None:
        if self._current_text is None:
            return
        await self._current_text.finish()
        self._current_text = None

    def _finish_thought(self) -> None:
        if self._current_thought is None:
            return
        self._current_thought.mark_done()
        self._current_thought = None


__all__ = ["ActivityLine", "AgentMessageView"]
