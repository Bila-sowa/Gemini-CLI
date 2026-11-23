"""Safe, atomic updates to a dotenv file.

The TUI persists the selected model and role in ``.env``. This module updates
exactly one key, preserves comments and unrelated lines, and never writes the
file through a symlink. It also avoids logging the file contents, because a
dotenv file may contain an API key.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LINE_PATTERN = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?)(?P<key>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*=\s*(?P<value>.*?)(?P<comment>\s+#.*)?$"
)
_ENDING_PATTERN = re.compile(r"(\r\n|\r|\n)$")


class EnvFileError(ValueError):
    """Raised when a dotenv file cannot be updated safely."""


def _split_ending(line: str) -> tuple[str, str]:
    """Return *line* without its terminator and the terminator itself."""
    match = _ENDING_PATTERN.search(line)
    if match is None:
        return line, ""
    return line[: match.start()], match.group(1)


def _detect_ending(text: str) -> str:
    """Return the first line terminator used by *text*."""
    for line in text.splitlines(keepends=True):
        _body, ending = _split_ending(line)
        if ending:
            return ending
    return "\n"


def _rewrite_line(line: str, key: str, value: str) -> str | None:
    """Return *line* with *key* set to *value*, or ``None`` when absent."""
    body, ending = _split_ending(line)
    match = _LINE_PATTERN.match(body)
    if match is None or match.group("key").upper() != key.upper():
        return None
    comment = match.group("comment") or ""
    prefix = match.group("prefix")
    return f"{prefix}{key}={value}{comment}{ending}"


def _validate_key(key: str) -> None:
    if _KEY_PATTERN.fullmatch(key) is None:
        msg = f"Invalid dotenv key {key!r}."
        raise EnvFileError(msg)


def _validate_value(value: str) -> None:
    if not value:
        msg = "Dotenv values cannot be empty."
        raise EnvFileError(msg)
    if "\0" in value or any(character.isspace() for character in value):
        msg = "Dotenv values cannot contain whitespace or NUL bytes."
        raise EnvFileError(msg)


def _read_existing(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except UnicodeDecodeError as exc:
        msg = f"{path} is not valid UTF-8."
        raise EnvFileError(msg) from exc


def _write_atomically(path: Path, content: str, mode: int) -> None:
    """Replace *path* with *content* using a temporary file in the same directory."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def set_env_value(path: Path, key: str, value: str) -> None:
    """Set *key* to *value* in the dotenv file at *path*.

    The update is atomic and preserves comments, unrelated keys, and the
    original line endings. New files are created with private permissions.

    Raises:
        EnvFileError: if the key, value, or target path is unsafe.
        OSError: if the file cannot be read or replaced.
    """
    _validate_key(key)
    _validate_value(value)

    target = Path(path)
    if target.is_symlink():
        msg = f"Refusing to write through symlink {target}."
        raise EnvFileError(msg)
    if target.exists() and not target.is_file():
        msg = f"Refusing to replace non-file path {target}."
        raise EnvFileError(msg)

    if target.exists():
        original = _read_existing(target)
        mode = target.stat().st_mode & 0o777
    else:
        original = ""
        mode = 0o600

    ending = _detect_ending(original)
    lines = original.splitlines(keepends=True)
    output: list[str] = []
    updated = False
    for line in lines:
        replacement = _rewrite_line(line, key, value)
        if replacement is None:
            output.append(line)
        else:
            output.append(replacement)
            updated = True

    if not updated:
        if output and not output[-1].endswith(("\n", "\r")):
            output[-1] = f"{output[-1]}{ending}"
        output.append(f"{key}={value}{ending}")

    _write_atomically(target, "".join(output), mode)


__all__ = ["EnvFileError", "set_env_value"]
