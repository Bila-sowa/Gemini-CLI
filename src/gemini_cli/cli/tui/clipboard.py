"""Copy text to the desktop clipboard from the terminal UI.

Terminal emulators that implement OSC 52 (kitty, WezTerm, foot, and similar)
accept Textual's own clipboard write. VTE-based terminals such as GNOME
Terminal, Tilix, and Terminator silently drop OSC 52, so installed native
helpers are tried first. When no helper is available the caller can fall back
to a private file and tell the user where the text was saved instead of
claiming a copy that never happened.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gemini_cli.config.defaults import GLOBAL_CONFIG_DIR_NAME

if TYPE_CHECKING:
    from textual.app import App

logger = logging.getLogger(__name__)

COPY_TIMEOUT_SECONDS = 3.0
FALLBACK_FILE_NAME = "last-code-block.txt"
SELECTION_FALLBACK_FILE_NAME = "last-selection.txt"


@dataclass(frozen=True)
class ClipboardResult:
    """Outcome of a clipboard attempt."""

    ok: bool
    method: str = ""
    fallback_path: Path | None = None


def _on_wayland() -> bool:
    return bool(os.environ.get("WAYLAND_DISPLAY"))


def clipboard_commands() -> list[tuple[str, list[str]]]:
    """Return the native clipboard helpers available on this system.

    Wayland and X11 selections are not shared on every desktop, so helpers are
    only offered for the active session type. This keeps a copy from silently
    landing in a selection that the desktop never reads.
    """
    commands: list[tuple[str, list[str]]] = []

    if _on_wayland() and shutil.which("wl-copy"):
        commands.append(("wl-copy", ["wl-copy"]))
    if shutil.which("pbcopy"):
        commands.append(("pbcopy", ["pbcopy"]))
    if shutil.which("clip.exe"):
        commands.append(("clip.exe", ["clip.exe"]))
    if not _on_wayland() and os.environ.get("DISPLAY"):
        if shutil.which("xclip"):
            commands.append(
                ("xclip", ["xclip", "-selection", "clipboard"]),
            )
        if shutil.which("xsel"):
            commands.append(
                ("xsel", ["xsel", "--clipboard", "--input"]),
            )
    return commands


def terminal_ignores_osc52() -> bool:
    """Whether the current terminal is known to drop OSC 52 writes.

    VTE (and therefore GNOME Terminal, Tilix, Terminator, and Ptyxis) does not
    implement OSC 52 clipboard writes. Reporting success there would be a lie.
    """
    return bool(os.environ.get("VTE_VERSION"))


async def _run_helper(command: list[str], text: str) -> bool:
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(
            process.communicate(text.encode("utf-8")),
            timeout=COPY_TIMEOUT_SECONDS,
        )
    except (TimeoutError, OSError):
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        return False
    return process.returncode == 0


async def copy_to_system_clipboard(text: str) -> ClipboardResult:
    """Copy *text* with the first native helper that succeeds."""
    for name, command in clipboard_commands():
        if await _run_helper(command, text):
            return ClipboardResult(ok=True, method=name)
    return ClipboardResult(ok=False)


async def copy_with_feedback(
    app: App[Any],
    text: str,
    *,
    label: str = "Text",
    fallback_file_name: str = FALLBACK_FILE_NAME,
) -> ClipboardResult:
    """Copy *text* and report the outcome with an app notification.

    Native helpers are tried first because VTE terminals drop OSC 52. When the
    terminal supports OSC 52 the copy is sent through it, and otherwise the
    text is saved to a private fallback file instead of claiming a copy that
    never reached the desktop clipboard.
    """
    result = await copy_to_system_clipboard(text)
    if result.ok:
        app.notify(f"{label} copied ({result.method}).", timeout=2)
        return result

    app.copy_to_clipboard(text)
    if not terminal_ignores_osc52():
        app.notify(f"{label} sent through the terminal clipboard.", timeout=2)
        return ClipboardResult(ok=True, method="OSC 52")

    message = (
        "System clipboard unavailable. Install wl-clipboard (Wayland) "
        "or xclip/xsel (X11)."
    )
    fallback = save_clipboard_fallback(text, file_name=fallback_file_name)
    if fallback is None:
        app.notify(
            f"{message} Could not save the text to a fallback file.",
            severity="warning",
            timeout=8,
        )
        return ClipboardResult(ok=False, method="file")

    app.notify(
        f"{message} {label} saved to {fallback}.",
        severity="warning",
        timeout=8,
    )
    return ClipboardResult(ok=True, method="file", fallback_path=fallback)


def save_clipboard_fallback(
    text: str,
    *,
    file_name: str = FALLBACK_FILE_NAME,
) -> Path | None:
    """Save *text* to a private file when the system clipboard is unavailable."""
    directory = Path.home() / GLOBAL_CONFIG_DIR_NAME / "clipboard"
    path = directory / file_name
    temporary: Path | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)
        descriptor, raw_name = tempfile.mkstemp(
            dir=str(directory),
            prefix=".code-block-",
            suffix=".tmp",
        )
        temporary = Path(raw_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
        path.chmod(0o600)
    except OSError:
        logger.warning("Could not save clipboard fallback to %s.", path)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        return None
    return path


__all__ = [
    "FALLBACK_FILE_NAME",
    "SELECTION_FALLBACK_FILE_NAME",
    "ClipboardResult",
    "clipboard_commands",
    "copy_to_system_clipboard",
    "copy_with_feedback",
    "save_clipboard_fallback",
    "terminal_ignores_osc52",
]
