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

from gemini_cli.config.defaults import GLOBAL_CONFIG_DIR_NAME

logger = logging.getLogger(__name__)

COPY_TIMEOUT_SECONDS = 3.0
FALLBACK_FILE_NAME = "last-code-block.txt"


@dataclass(frozen=True)
class ClipboardResult:
    """Outcome of a native clipboard attempt."""

    ok: bool
    method: str = ""


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


def save_clipboard_fallback(text: str) -> Path | None:
    """Save *text* to a private file when the system clipboard is unavailable."""
    directory = Path.home() / GLOBAL_CONFIG_DIR_NAME / "clipboard"
    path = directory / FALLBACK_FILE_NAME
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
    "ClipboardResult",
    "clipboard_commands",
    "copy_to_system_clipboard",
    "save_clipboard_fallback",
    "terminal_ignores_osc52",
]
