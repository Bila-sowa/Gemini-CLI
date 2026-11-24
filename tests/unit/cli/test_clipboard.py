"""Tests for desktop clipboard integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.containers import VerticalScroll

from gemini_cli.cli.tui import clipboard
from gemini_cli.cli.tui.markdown import CopyableMarkdownFence, MarkdownMessageView


class FakeClipboardHost:
    """Minimal app surface used by ``copy_with_feedback``."""

    def __init__(self) -> None:
        self.notifications: list[dict[str, Any]] = []
        self.copied: list[str] = []

    def notify(
        self,
        message: str,
        *,
        title: str = "",
        severity: str = "information",
        timeout: float | None = None,
        markup: bool = True,
    ) -> None:
        self.notifications.append(
            {
                "message": message,
                "title": title,
                "severity": severity,
                "timeout": timeout,
                "markup": markup,
            }
        )

    def copy_to_clipboard(self, text: str) -> None:
        self.copied.append(text)


def test_wayland_prefers_wl_copy(monkeypatch) -> None:
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"wl-copy", "xclip"} else None,
    )

    assert clipboard.clipboard_commands() == [("wl-copy", ["wl-copy"])]


def test_x11_uses_xclip(monkeypatch) -> None:
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"xclip", "xsel"} else None,
    )

    assert clipboard.clipboard_commands() == [
        ("xclip", ["xclip", "-selection", "clipboard"]),
        ("xsel", ["xsel", "--clipboard", "--input"]),
    ]


def test_no_helpers_is_reported(monkeypatch) -> None:
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr(clipboard.shutil, "which", lambda _name: None)

    assert clipboard.clipboard_commands() == []


def test_vte_is_known_to_ignore_osc52(monkeypatch) -> None:
    monkeypatch.setenv("VTE_VERSION", "8400")
    assert clipboard.terminal_ignores_osc52() is True

    monkeypatch.delenv("VTE_VERSION")
    assert clipboard.terminal_ignores_osc52() is False


async def test_native_helper_receives_the_text(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "clipboard.txt"
    monkeypatch.setattr(
        clipboard,
        "clipboard_commands",
        lambda: [("fake", ["/bin/sh", "-c", f"cat > {target}"])],
    )

    result = await clipboard.copy_to_system_clipboard("hello\nworld")

    assert result.ok is True
    assert result.method == "fake"
    assert target.read_text(encoding="utf-8") == "hello\nworld"


async def test_failed_helper_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(
        clipboard,
        "clipboard_commands",
        lambda: [("broken", ["/bin/sh", "-c", "exit 1"])],
    )

    result = await clipboard.copy_to_system_clipboard("hello")

    assert result.ok is False


def test_fallback_file_is_private(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        clipboard.Path,
        "home",
        classmethod(lambda _cls: tmp_path),
    )

    path = clipboard.save_clipboard_fallback("secret code")

    assert path == tmp_path / ".gemini-cli" / "clipboard" / "last-code-block.txt"
    assert path is not None
    assert path.read_text(encoding="utf-8") == "secret code"
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


async def test_copy_with_feedback_prefers_a_native_helper(monkeypatch) -> None:
    async def native_copy(_text: str) -> clipboard.ClipboardResult:
        return clipboard.ClipboardResult(ok=True, method="wl-copy")

    monkeypatch.setattr(clipboard, "copy_to_system_clipboard", native_copy)
    host = FakeClipboardHost()

    result = await clipboard.copy_with_feedback(host, "hello", label="Selection")

    assert result.ok is True
    assert host.copied == []
    assert host.notifications == [
        {
            "message": "Selection copied (wl-copy).",
            "title": "",
            "severity": "information",
            "timeout": 2,
            "markup": True,
        }
    ]


async def test_copy_with_feedback_uses_osc52_when_no_helper_exists(
    monkeypatch,
) -> None:
    async def no_clipboard(_text: str) -> clipboard.ClipboardResult:
        return clipboard.ClipboardResult(ok=False)

    monkeypatch.setattr(clipboard, "copy_to_system_clipboard", no_clipboard)
    monkeypatch.setattr(clipboard, "terminal_ignores_osc52", lambda: False)
    host = FakeClipboardHost()

    result = await clipboard.copy_with_feedback(host, "hello", label="Selection")

    assert result.ok is True
    assert host.copied == ["hello"]
    assert host.notifications[0]["message"] == (
        "Selection sent through the terminal clipboard."
    )


async def test_copy_with_feedback_saves_a_selection_file_on_vte(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def no_clipboard(_text: str) -> clipboard.ClipboardResult:
        return clipboard.ClipboardResult(ok=False)

    saved: list[tuple[str, str]] = []
    fallback_path = tmp_path / "last-selection.txt"

    def fake_save(text: str, *, file_name: str = "") -> Path:
        saved.append((text, file_name))
        return fallback_path

    monkeypatch.setattr(clipboard, "copy_to_system_clipboard", no_clipboard)
    monkeypatch.setattr(clipboard, "terminal_ignores_osc52", lambda: True)
    monkeypatch.setattr(clipboard, "save_clipboard_fallback", fake_save)
    host = FakeClipboardHost()

    result = await clipboard.copy_with_feedback(
        host,
        "hello",
        label="Selection",
        fallback_file_name=clipboard.SELECTION_FALLBACK_FILE_NAME,
    )

    assert saved == [("hello", "last-selection.txt")]
    assert result.ok is True
    assert result.fallback_path == fallback_path
    notification = host.notifications[0]
    assert notification["severity"] == "warning"
    assert notification["timeout"] == 8
    assert str(fallback_path) in str(notification["message"])


async def test_copy_with_feedback_reports_a_failed_fallback(monkeypatch) -> None:
    async def no_clipboard(_text: str) -> clipboard.ClipboardResult:
        return clipboard.ClipboardResult(ok=False)

    monkeypatch.setattr(clipboard, "copy_to_system_clipboard", no_clipboard)
    monkeypatch.setattr(clipboard, "terminal_ignores_osc52", lambda: True)
    monkeypatch.setattr(clipboard, "save_clipboard_fallback", lambda *_, **__: None)
    host = FakeClipboardHost()

    result = await clipboard.copy_with_feedback(host, "hello", label="Selection")

    assert result.ok is False
    notification = host.notifications[0]
    assert notification["severity"] == "warning"
    assert "Could not save the text" in str(notification["message"])


def test_fallback_file_accepts_a_custom_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        clipboard.Path,
        "home",
        classmethod(lambda _cls: tmp_path),
    )

    path = clipboard.save_clipboard_fallback(
        "selected text",
        file_name=clipboard.SELECTION_FALLBACK_FILE_NAME,
    )

    assert path == tmp_path / ".gemini-cli" / "clipboard" / "last-selection.txt"
    assert path is not None
    assert path.read_text(encoding="utf-8") == "selected text"
    assert path.stat().st_mode & 0o777 == 0o600


async def test_code_block_copy_uses_the_shared_helper(monkeypatch) -> None:
    from gemini_cli.cli.tui import markdown as markdown_module
    from gemini_cli.cli.tui.clipboard import ClipboardResult
    from tests.unit.cli.test_markdown import MarkdownApp

    calls: list[tuple[str, str]] = []

    async def fake_copy(
        _app: object,
        text: str,
        *,
        label: str = "Text",
        fallback_file_name: str = clipboard.FALLBACK_FILE_NAME,
    ) -> ClipboardResult:
        calls.append((text, label))
        return ClipboardResult(ok=True, method="fake")

    monkeypatch.setattr(markdown_module, "copy_with_feedback", fake_copy)

    app = MarkdownApp("```text\nsecret\n```")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.click(".fence-copy")
        await pilot.pause()

    assert calls == [("secret", "Code block")]


async def test_code_toolbar_stays_visible_while_code_scrolls() -> None:
    from tests.unit.cli.test_markdown import MarkdownApp

    code = "\n".join(f"line {index}" for index in range(40))
    app = MarkdownApp(f"```text\n{code}\n```")

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        view = app.query_one(MarkdownMessageView)
        fence = view.query_one(CopyableMarkdownFence)
        body = fence.query_one(".fence-body", VerticalScroll)
        button = fence.query_one(".fence-copy")
        before = button.region

        assert body.max_scroll_y > 0
        body.scroll_to(y=10, animate=False)
        await pilot.pause()

        assert button.region == before
