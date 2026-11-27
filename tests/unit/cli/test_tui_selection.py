"""Tests for automatic clipboard copying of mouse selections."""

from __future__ import annotations

from typing import Any

from textual import events

from gemini_cli.cli.tui.clipboard import SELECTION_FALLBACK_FILE_NAME, ClipboardResult


def _record_copy(
    monkeypatch: Any,
    calls: list[tuple[str, str, str]],
) -> None:
    from gemini_cli.cli.tui import app as app_module

    async def fake_copy(
        _app: object,
        text: str,
        *,
        label: str = "Text",
        fallback_file_name: str = "last-code-block.txt",
    ) -> ClipboardResult:
        calls.append((text, label, fallback_file_name))
        return ClipboardResult(ok=True, method="fake")

    monkeypatch.setattr(app_module, "copy_with_feedback", fake_copy)


async def test_mouse_selection_is_copied_and_labelled(monkeypatch: Any) -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from tests.unit.cli.test_repl import FakeRuntime

    calls: list[tuple[str, str, str]] = []
    _record_copy(monkeypatch, calls)

    app = GeminiApp(FakeRuntime())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        markdown = app.query_one("MarkdownMessageView").query_one("GeminiMarkdown")
        region = markdown.region
        await pilot.mouse_down(offset=(region.x + 1, region.y))
        await pilot.mouse_up(offset=(region.x + 20, region.y))
        await pilot.pause(0.1)

    assert len(calls) == 1
    text, label, fallback_file_name = calls[0]
    assert "help" in text
    assert label == "Selection"
    assert fallback_file_name == SELECTION_FALLBACK_FILE_NAME


async def test_plain_click_does_not_copy(monkeypatch: Any) -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from tests.unit.cli.test_repl import FakeRuntime

    calls: list[tuple[str, str, str]] = []
    _record_copy(monkeypatch, calls)

    app = GeminiApp(FakeRuntime())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        markdown = app.query_one("MarkdownMessageView").query_one("GeminiMarkdown")
        region = markdown.region
        await pilot.click(offset=(region.x + 3, region.y))
        await pilot.pause(0.1)

    assert calls == []


async def test_whitespace_selection_is_ignored(monkeypatch: Any) -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from tests.unit.cli.test_repl import FakeRuntime

    calls: list[tuple[str, str, str]] = []
    _record_copy(monkeypatch, calls)

    app = GeminiApp(FakeRuntime())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(
            type(app.screen),
            "get_selected_text",
            lambda _self: "   \n",
        )
        app.screen.post_message(events.TextSelected())
        await pilot.pause(0.1)

    assert calls == []


async def test_empty_selection_is_ignored(monkeypatch: Any) -> None:
    from gemini_cli.cli.tui.app import GeminiApp
    from tests.unit.cli.test_repl import FakeRuntime

    calls: list[tuple[str, str, str]] = []
    _record_copy(monkeypatch, calls)

    app = GeminiApp(FakeRuntime())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(
            type(app.screen),
            "get_selected_text",
            lambda _self: None,
        )
        app.screen.post_message(events.TextSelected())
        await pilot.pause(0.1)

    assert calls == []
