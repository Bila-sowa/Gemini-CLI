"""Markdown rendering for messages in the Textual terminal UI.

Textual already ships a GFM parser and a streaming API. This module adds the
small amount of behavior the product needs on top of it:

* control-character and ANSI sanitization for untrusted model output;
* inline ``<u>`` support, which GFM does not define;
* a message widget that keeps the speaker prefix and the rendered body
  separate, so a user message is no longer bold as a whole.

The custom block mixin follows Textual's inline token conversion. It is kept
deliberately small because the upstream implementation is private API; the
tests exercise every supported inline construct.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content, Span
from textual.css.query import NoMatches
from textual.style import Style
from textual.widgets import Button, Label, Markdown, Static
from textual.widgets.markdown import MarkdownBlock, MarkdownFence, MarkdownStream

from gemini_cli.cli.tui.clipboard import copy_with_feedback
from gemini_cli.cli.tui.themes import Theme

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)?)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]")


def sanitize_markdown(text: str) -> str:
    """Normalize line endings and remove terminal control sequences.

    Markdown rendering must never receive raw ANSI escapes from model output.
    Tabs and newlines are preserved because they are meaningful, if harmless.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    without_ansi = _ANSI_ESCAPE_RE.sub("", normalized)
    return _CONTROL_RE.sub("", without_ansi)


class _UnderlineMarkdownBlock:
    """Inline token conversion with ``<u>`` support.

    This mixin is combined with every Textual Markdown block class. It mirrors
    ``MarkdownBlock._token_to_content`` and adds only the underline branch.
    """

    def _token_to_content(self, token: Any) -> Content:
        if token.children is None:
            return Content("")

        parts: list[str] = []
        spans: list[Span] = []
        style_stack: list[tuple[Style | str, int]] = []
        underline_stack: list[int] = []
        position = 0

        def add_content(value: str) -> None:
            nonlocal position
            parts.append(value)
            position += len(value)

        def add_style(style: Style | str) -> None:
            style_stack.append((style, position))

        def close_tag() -> None:
            if not style_stack:
                return
            style, start = style_stack.pop()
            if start < position:
                spans.append(Span(start, position, style))

        def close_underline() -> None:
            if not underline_stack:
                return
            start = underline_stack.pop()
            if start < position:
                spans.append(Span(start, position, Style(underline=True)))

        for child in token.children:
            child_type = child.type
            if child_type == "text":
                add_content(re.sub(r"\s+", " ", child.content))
            elif child_type == "hardbreak":
                add_content("\n")
            elif child_type == "softbreak":
                add_content(" ")
            elif child_type == "code_inline":
                add_style(".code_inline")
                add_content(child.content)
                close_tag()
            elif child_type == "em_open":
                add_style(".em")
            elif child_type == "strong_open":
                add_style(".strong")
            elif child_type == "s_open":
                add_style(".s")
            elif child_type == "html_inline":
                tag = child.content.strip().lower()
                if tag == "<u>" or tag.startswith("<u "):
                    underline_stack.append(position)
                elif tag == "</u>":
                    close_underline()
            elif child_type == "link_open":
                href = child.attrs.get("href", "")
                action = f"link({href!r})"
                add_style(Style.from_meta({"@click": action}))
            elif child_type == "image":
                href = child.attrs.get("src", "")
                alt = child.attrs.get("alt", "")
                action = f"link({href!r})"
                add_style(Style.from_meta({"@click": action}))
                add_content("🖼  ")
                if alt:
                    add_content(f"({alt})")
                if child.children is not None:
                    for grandchild in child.children:
                        add_content(grandchild.content)
                close_tag()
            elif child_type.endswith("_close"):
                close_tag()

        return Content("".join(parts), spans=spans)


_UNDERLINE_BLOCK_CACHE: dict[type[MarkdownBlock], type[MarkdownBlock]] = {}


def _underline_block(base: type[MarkdownBlock]) -> type[MarkdownBlock]:
    """Return a block class with underline-aware inline conversion."""
    cached = _UNDERLINE_BLOCK_CACHE.get(base)
    if cached is not None:
        return cached
    variant = type(
        f"Underline{base.__name__}",
        (_UnderlineMarkdownBlock, base),
        {"__module__": __name__},
    )
    _UNDERLINE_BLOCK_CACHE[base] = variant
    return variant


class CopyableMarkdownFence(MarkdownFence):
    """A fenced code block with a toolbar that copies the whole block.

    The toolbar and the code area are separate widgets: the code keeps its
    horizontal scrolling, while the copy button stays reachable for long lines.
    """

    def compose(self) -> ComposeResult:
        with Horizontal(classes="fence-toolbar"):
            yield Static(self.lexer or "text", classes="fence-language")
            yield Button(
                "copy",
                classes="fence-copy",
                compact=True,
                tooltip="Copy code block",
            )
        yield VerticalScroll(
            Label(self._highlighted_code, id="code-content", expand=True),
            classes="fence-body",
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if not event.button.has_class("fence-copy"):
            return
        event.stop()
        await self._copy_code()

    async def _copy_code(self) -> None:
        await copy_with_feedback(self.app, self.code, label="Code block")


class GeminiMarkdown(Markdown):
    """Textual Markdown widget with inline underline support."""

    def get_block_class(self, block_name: str) -> type[MarkdownBlock]:
        if block_name in {"fence", "code_block"}:
            return CopyableMarkdownFence
        return _underline_block(super().get_block_class(block_name))


class MarkdownMessageView(Vertical):
    """A speaker prefix followed by a Markdown-rendered message body."""

    DEFAULT_CSS = """
    MarkdownMessageView {
        width: 100%;
        height: auto;
    }
    MarkdownMessageView .markdown-message-row {
        width: 100%;
        height: auto;
    }
    MarkdownMessageView .message-prefix {
        width: auto;
        height: 1;
        padding-right: 1;
    }
    MarkdownMessageView .message-markdown {
        width: 1fr;
        height: auto;
        padding: 0 2 0 0;
        background: transparent;
    }
    MarkdownMessageView .message-error {
        width: 100%;
        height: auto;
        margin: 0 0 0 3;
    }
    MarkdownMessageView MarkdownH1,
    MarkdownMessageView MarkdownH2,
    MarkdownMessageView MarkdownH3,
    MarkdownMessageView MarkdownH4,
    MarkdownMessageView MarkdownH5,
    MarkdownMessageView MarkdownH6 {
        content-align: left middle;
        margin: 0 0 1 0;
    }
    MarkdownMessageView MarkdownParagraph {
        margin: 0 0 1 0;
    }
    MarkdownMessageView MarkdownBulletList,
    MarkdownMessageView MarkdownOrderedList {
        margin: 0 0 1 0;
    }
    MarkdownMessageView MarkdownList MarkdownList {
        margin: 0;
    }
    MarkdownMessageView MarkdownOrderedList Vertical MarkdownParagraph,
    MarkdownMessageView MarkdownBulletList Vertical MarkdownParagraph,
    MarkdownMessageView MarkdownBlockQuote MarkdownParagraph,
    MarkdownMessageView MarkdownBlockQuote MarkdownList {
        margin: 0;
    }
    MarkdownMessageView MarkdownBlockQuote {
        background: $boost;
        border-left: outer $primary 50%;
        margin: 1 0;
        padding: 1 2;
    }
    MarkdownMessageView MarkdownFence {
        layout: vertical;
        background: $surface;
        border: round $panel;
        margin: 1 0;
        padding: 0;
        overflow: hidden;
    }
    MarkdownMessageView MarkdownFence .fence-toolbar {
        height: 1;
        width: 100%;
        background: $panel;
    }
    MarkdownMessageView MarkdownFence .fence-language {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: $text 60%;
    }
    MarkdownMessageView MarkdownFence .fence-copy {
        height: 1;
        min-width: 8;
        border: none;
        padding: 0 1;
        background: $panel;
        color: $primary;
    }
    MarkdownMessageView MarkdownFence .fence-copy:hover {
        background: $primary 30%;
        color: $text;
    }
    MarkdownMessageView MarkdownFence .fence-body {
        height: auto;
        max-height: 20;
        background: transparent;
        overflow-x: auto;
        overflow-y: auto;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
    }
    MarkdownMessageView MarkdownFence Label {
        padding: 0 2;
    }
    MarkdownMessageView MarkdownHorizontalRule {
        color: $primary 60%;
        margin: 1 0;
    }
    MarkdownMessageView MarkdownTableContent {
        keyline: thin $primary 20%;
    }
    """

    def __init__(
        self,
        prefix: str,
        prefix_style: str,
        *,
        initial: str = "",
        error_style: str = "",
        streaming: bool = False,
        role: Literal["user", "assistant"] = "assistant",
    ) -> None:
        super().__init__(classes="message markdown-message")
        self._prefix = prefix
        self._prefix_style = prefix_style
        self._error_style = error_style
        self._role = role
        self._source: list[str] = []
        self._errors: list[str] = []
        self._markdown: GeminiMarkdown | None = None
        self._error_widget: Static | None = None
        self._stream: MarkdownStream | None = None
        self._finished = False
        self._streaming = streaming
        if initial:
            self._source.append(sanitize_markdown(initial))

    def compose(self) -> ComposeResult:
        with Horizontal(classes="markdown-message-row"):
            yield Static(
                Text(self._prefix, style=self._prefix_style),
                classes="message-prefix",
            )
            yield GeminiMarkdown(
                classes="message-markdown",
                open_links=False,
            )
        yield Static("", classes="message-error")

    async def on_mount(self) -> None:
        try:
            self._markdown = self.query_one(GeminiMarkdown)
            self._error_widget = self.query_one(".message-error", Static)
        except NoMatches:
            # The conversation may be cleared between mounting the message and
            # its children processing the mount event.
            return
        if self._streaming and not self._finished:
            self._stream = Markdown.get_stream(self._markdown)
            if self._source:
                await self._stream.write(self.source)
        else:
            # Complete the widget's own empty-document mount before rendering.
            await self._markdown.update(self.source)
        self._refresh_error()

    async def on_unmount(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None

    @property
    def source(self) -> str:
        """The sanitized Markdown source received so far."""
        return "".join(self._source)

    @property
    def plain(self) -> str:
        """Prefix plus raw source and errors, used by the TUI transcript."""
        errors = "".join(f"\nError: {error}" for error in self._errors)
        return f"{self._prefix}{self.source}{errors}"

    async def append(self, chunk: str) -> None:
        """Append a streamed Markdown fragment."""
        cleaned = sanitize_markdown(chunk)
        if not cleaned:
            return
        self._source.append(cleaned)
        if self._finished:
            if self._markdown is not None and self._markdown.is_mounted:
                await self._markdown.update(self.source)
            return
        if self._markdown is None or not self._markdown.is_mounted:
            return
        if self._stream is None:
            self._stream = Markdown.get_stream(self._markdown)
        await self._stream.write(cleaned)

    async def finish(self) -> None:
        """Flush buffered fragments and render the final document once."""
        if self._finished:
            return
        self._finished = True
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
        if self._markdown is not None and self._markdown.is_mounted:
            await self._markdown.update(self.source)

    def add_error(self, message: str) -> None:
        """Attach an error that must bypass Markdown parsing."""
        self._errors.append(message)
        self._refresh_error()

    def apply_theme(self, theme: Theme) -> None:
        """Update the speaker prefix and error colors for a new theme."""
        if self._role == "user":
            self._prefix_style = f"bold {theme.user_color}"
        else:
            self._prefix_style = theme.assistant_color
        self._error_style = theme.error_color

        try:
            prefix = self.query_one(".message-prefix", Static)
        except NoMatches:
            pass
        else:
            prefix.update(Text(self._prefix, style=self._prefix_style))

        self._refresh_error()

    async def rerender(self) -> None:
        """Re-render the current source, for example after a theme change."""
        if self._stream is not None:
            # An active stream re-renders when finish() runs.
            return
        if self._markdown is not None and self._markdown.is_mounted:
            await self._markdown.update(self.source)

    def _refresh_error(self) -> None:
        if self._error_widget is None:
            return
        if not self._errors:
            self._error_widget.update("")
            self._error_widget.display = False
            return
        text = "\n".join(f"Error: {error}" for error in self._errors)
        self._error_widget.update(Text(text, style=self._error_style))
        self._error_widget.display = True


__all__ = [
    "GeminiMarkdown",
    "MarkdownMessageView",
    "sanitize_markdown",
]
