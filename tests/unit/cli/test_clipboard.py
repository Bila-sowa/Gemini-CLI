"""Tests for desktop clipboard integration."""

from __future__ import annotations

from pathlib import Path

from textual.containers import VerticalScroll

from gemini_cli.cli.tui import clipboard
from gemini_cli.cli.tui.markdown import CopyableMarkdownFence, MarkdownMessageView


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


async def test_copy_falls_back_to_a_private_file_when_clipboard_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from gemini_cli.cli.tui import markdown as markdown_module
    from gemini_cli.cli.tui.clipboard import ClipboardResult
    from tests.unit.cli.test_markdown import MarkdownApp

    saved: list[str] = []

    async def no_clipboard(_text: str) -> ClipboardResult:
        return ClipboardResult(ok=False)

    def fake_save(text: str) -> Path:
        saved.append(text)
        return tmp_path / "last-code-block.txt"

    monkeypatch.setattr(markdown_module, "copy_to_system_clipboard", no_clipboard)
    monkeypatch.setattr(markdown_module, "terminal_ignores_osc52", lambda: True)
    monkeypatch.setattr(markdown_module, "save_clipboard_fallback", fake_save)

    app = MarkdownApp("```text\nsecret\n```")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.click(".fence-copy")
        await pilot.pause()

    assert saved == ["secret"]


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
