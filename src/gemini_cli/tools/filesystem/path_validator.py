"""Resolve and confine filesystem paths to approved roots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class PathSecurityError(ValueError):
    """Raised when a path is outside the approved roots or otherwise unsafe."""


def validate_path(
    path: str | Path,
    allowed_roots: Sequence[Path],
    *,
    must_exist: bool = False,
    must_be_file: bool = False,
) -> Path:
    """Resolve *path* and verify it stays inside one of *allowed_roots*.

    The path is resolved before the containment check, so traversal components
    and symlinks pointing outside the roots are rejected. The resolved path is
    returned and should be used for the actual filesystem operation.

    An empty *allowed_roots* sequence denies every path: filesystem access is
    closed unless it is explicitly granted.

    Raises:
        PathSecurityError: if the path is empty, contains a NUL byte, cannot be
            resolved, lies outside every root, or fails the requested checks.
    """
    raw = str(path)
    if "\x00" in raw:
        msg = "Path must not contain a NUL byte."
        raise PathSecurityError(msg)
    if not raw.strip():
        msg = "Path must not be empty."
        raise PathSecurityError(msg)
    if not allowed_roots:
        msg = "No allowed roots are configured; filesystem access is disabled."
        raise PathSecurityError(msg)

    roots = tuple(root.expanduser().resolve() for root in allowed_roots)
    try:
        resolved = Path(raw).expanduser().resolve()
    except OSError as exc:
        msg = f"Cannot resolve path {raw!r}: {exc}"
        raise PathSecurityError(msg) from exc

    if not any(resolved.is_relative_to(root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        msg = f"Path {raw!r} is outside the allowed roots: {allowed}."
        raise PathSecurityError(msg)

    if must_exist and not resolved.exists():
        msg = f"Path does not exist: {raw!r}."
        raise PathSecurityError(msg)
    if must_be_file and not resolved.is_file():
        msg = f"Path is not a regular file: {raw!r}."
        raise PathSecurityError(msg)

    return resolved


__all__ = ["PathSecurityError", "validate_path"]
