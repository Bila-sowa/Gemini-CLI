"""Tests for Markdown rendering in the Textual terminal UI."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from gemini_cli.cli.tui.markdown import (
    GeminiMarkdown,
    MarkdownMessageView,
    sanitize_markdown,
)
from gemini_cli.cli.tui.themes import LIGHT_THEME


class MarkdownApp(App[None]):
    """Minimal host for a single Markdown message."""

    def __init__(self, initial: str = "") -> None:
        super().__init__()
        self._initial = initial

    def compose(self) -> ComposeResult:
        yield MarkdownMessageView("A: ", "bold #5B9BD5", initial=self._initial)


def _block_content(block: object) -> object:
    return getattr(block, "_content", None)


def _block_plain(block: object) -> str:
    content = _block_content(block)
    return str(getattr(content, "plain", ""))


def _block_styles(block: object) -> list[str]:
    content = _block_content(block)
    spans = getattr(content, "spans", ())
    return [str(span.style) for span in spans]


def test_sanitize_markdown_removes_terminal_sequences() -> None:
    raw = "\x1b[31mred\x1b[0m\r\nline\x00\tend\x1b]0;title\x07"
    assert sanitize_markdown(raw) == "red\nline\tend"


def test_sanitize_markdown_normalizes_carriage_returns() -> None:
    assert sanitize_markdown("a\rb\r\nc") == "a\nb\nc"


def test_markdown_message_view_applies_theme_to_prefix_and_errors() -> None:
    user_view = MarkdownMessageView("User: ", "bold #E6E6E6", role="user")
    assistant_view = MarkdownMessageView("✦: ", "#5B9BD5")

    user_view.apply_theme(LIGHT_THEME)
    assistant_view.apply_theme(LIGHT_THEME)

    assert user_view._prefix_style == f"bold {LIGHT_THEME.user_color}"
    assert assistant_view._prefix_style == LIGHT_THEME.assistant_color
    assert user_view._error_style == LIGHT_THEME.error_color


async def test_markdown_message_renders_supported_blocks_and_inline_styles() -> None:
    initial = (
        "# Heading one\n\n"
        "## Heading two\n\n"
        "**bold** *italic* ~~strike~~ <u>under</u> `code`\n\n"
        "- item\n\n"
        "> quote\n\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |\n\n"
        "```python\n"
        "print(1)\n"
        "```\n"
    )
    app = MarkdownApp(initial)

    async with app.run_test(size=(100, 50)) as pilot:
        await pilot.pause()
        view = app.query_one(MarkdownMessageView)
        markdown = view.query_one(GeminiMarkdown)
        blocks = list(markdown.children)
        names = [block.name for block in blocks]
        plain_blocks = [_block_plain(block) for block in blocks]

        assert "Heading one" in plain_blocks
        assert "Heading two" in plain_blocks
        assert "bullet_list_open" in names
        assert "blockquote_open" in names
        assert "table_open" in names
        assert "fence" in names

        paragraph = next(
            block for block in blocks if _block_plain(block).startswith("bold")
        )
        assert _block_plain(paragraph) == "bold italic strike under code"
        styles = _block_styles(paragraph)
        assert ".strong" in styles
        assert ".em" in styles
        assert ".s" in styles
        assert "underline" in styles
        assert ".code_inline" in styles
        assert view.plain.startswith("A: # Heading one")


async def test_underline_is_not_applied_inside_inline_code() -> None:
    app = MarkdownApp("`<u>literal</u>`")

    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        view = app.query_one(MarkdownMessageView)
        paragraph = next(iter(view.query_one(GeminiMarkdown).children))

        assert _block_plain(paragraph) == "<u>literal</u>"
        assert _block_styles(paragraph) == [".code_inline"]


async def test_inline_code_keeps_its_tight_background() -> None:
    app = MarkdownApp("alpha `code` omega")

    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        view = app.query_one(MarkdownMessageView)
        paragraph = next(iter(view.query_one(GeminiMarkdown).children))
        content = _block_content(paragraph)
        spans = list(getattr(content, "spans", ()))

        assert _block_plain(paragraph) == "alpha code omega"
        assert len(spans) == 1
        assert spans[0].style == ".code_inline"
        assert (spans[0].start, spans[0].end) == (6, 10)


async def test_code_block_copy_button_copies_the_whole_block(monkeypatch) -> None:
    from textual.widgets import Button

    from gemini_cli.cli.tui import markdown as markdown_module
    from gemini_cli.cli.tui.clipboard import ClipboardResult
    from gemini_cli.cli.tui.markdown import CopyableMarkdownFence

    copied: list[tuple[str, str]] = []

    async def fake_copy(
        _app: object,
        text: str,
        *,
        label: str = "Text",
        fallback_file_name: str = "last-code-block.txt",
    ) -> ClipboardResult:
        copied.append((text, label))
        return ClipboardResult(ok=True, method="test-clipboard")

    monkeypatch.setattr(markdown_module, "copy_with_feedback", fake_copy)
    app = MarkdownApp("```python\nprint(1)\nprint(2)\n```")

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        view = app.query_one(MarkdownMessageView)
        fence = view.query_one(CopyableMarkdownFence)
        button = fence.query_one(".fence-copy", Button)

        assert fence.query_one(".fence-language", Static).content == "python"
        assert button.label.plain == "copy"

        await pilot.click(".fence-copy")
        await pilot.pause()

    assert copied == [("print(1)\nprint(2)", "Code block")]


async def test_markdown_spacing_keeps_quotes_centered_and_lists_compact() -> None:
    app = MarkdownApp("> quoted\n\n1. first\n2. second\n\n- bullet\n")

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        view = app.query_one(MarkdownMessageView)
        markdown = view.query_one(GeminiMarkdown)
        blockquote = markdown.query_one("MarkdownBlockQuote")
        quoted = blockquote.query_one("MarkdownParagraph")

        assert quoted.region.y == blockquote.region.y + 1
        assert blockquote.region.height == quoted.region.height + 2

        for item_row in markdown.query("MarkdownList Horizontal"):
            assert item_row.region.height == 1

        assert markdown.styles.padding.right == 2


async def test_streamed_markdown_is_reparsed_until_the_final_render() -> None:
    app = MarkdownApp()

    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        view = app.query_one(MarkdownMessageView)

        await view.append("```python\nprint(1)")
        await pilot.pause(0.1)
        await view.append("\n```\n\n**bo")
        await view.append("ld**\n")
        await view.finish()
        await pilot.pause()

        markdown = view.query_one(GeminiMarkdown)
        names = [block.name for block in markdown.children]
        assert "fence" in names
        paragraph = next(
            block for block in markdown.children if _block_plain(block) == "bold"
        )
        assert ".strong" in _block_styles(paragraph)
        assert view.source.endswith("**bold**\n")


async def test_errors_are_rendered_outside_markdown() -> None:
    app = MarkdownApp("ok")

    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        view = app.query_one(MarkdownMessageView)
        view.add_error("**not markdown**")
        await pilot.pause()

        error = view.query_one(".message-error", Static)
        assert error.display is True
        assert "**not markdown**" in view.plain
        assert "<u>" not in view.plain
        assert "Error: **not markdown**" in view.plain


async def test_user_and_assistant_prefixes_are_separate_from_the_body() -> None:
    app = MarkdownApp("**body**")

    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        view = app.query_one(MarkdownMessageView)
        prefix = view.query_one(".message-prefix", Static)
        markdown = view.query_one(GeminiMarkdown)

        assert str(prefix.content) == "A: "
        assert markdown.source == "**body**"
        assert view.plain == "A: **body**"


def test_themes_style_markdown_headings_and_links() -> None:
    from gemini_cli.cli.tui.themes import DEFAULT_THEME, LIGHT_THEME, to_textual_theme

    for theme in (DEFAULT_THEME, LIGHT_THEME):
        textual_theme = to_textual_theme(theme)
        variables = textual_theme.variables

        assert variables["markdown-h1-color"] == theme.accent
        assert variables["markdown-h2-text-style"] == "bold"
        assert "underline" not in variables["markdown-h2-text-style"]
        assert variables["link-color"] == theme.accent
        assert variables["link-color"] != theme.foreground
        assert textual_theme.name == f"gemini-{theme.name}"


async def test_links_are_styled_but_do_not_open_automatically() -> None:
    app = MarkdownApp("[docs](https://example.com)")

    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        view = app.query_one(MarkdownMessageView)
        markdown = view.query_one(GeminiMarkdown)
        paragraph = next(iter(markdown.children))

        assert markdown._open_links is False
        assert _block_plain(paragraph) == "docs"
        assert any("link(" in style for style in _block_styles(paragraph))
