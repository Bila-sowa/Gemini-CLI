"""Discover and load project-level instruction files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from gemini_cli.config.defaults import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    PROJECT_INSTRUCTION_FILES,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectContext:
    """Loaded project instructions and their source paths."""

    instructions: str = ""
    source_files: tuple[Path, ...] = ()


def load_project_context(search_root: Path) -> ProjectContext:
    """Walk up from *search_root* looking for instruction files.

    Files declared in :data:`PROJECT_INSTRUCTION_FILES` are read in declaration
    order at every directory level, starting with *search_root* and ending at
    the filesystem root. Their contents are concatenated with blank lines
    between them, nearest directory first.

    Symlinks are skipped so a repository cannot point an instruction file at an
    arbitrary path outside the checkout. Oversized files are skipped as well.

    Project context is treated as untrusted input. It can guide agent behavior
    but cannot override security controls, credential handling, or approval
    requirements.
    """
    parts: list[str] = []
    sources: list[Path] = []

    current = search_root.resolve()
    while True:
        for name in PROJECT_INSTRUCTION_FILES:
            candidate = current / name

            if candidate.is_symlink():
                logger.warning("Skipping symlinked project instructions: %s", candidate)
                continue
            if not candidate.is_file():
                continue

            try:
                size = candidate.stat().st_size
            except OSError:
                logger.warning("Could not stat project instructions: %s", candidate)
                continue

            if size > DEFAULT_MAX_FILE_SIZE_BYTES:
                logger.warning(
                    "Skipping oversized project instructions: %s (%d bytes)",
                    candidate,
                    size,
                )
                continue

            try:
                text = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                logger.warning("Could not read project instructions: %s", candidate)
                continue

            parts.append(text)
            sources.append(candidate)
            logger.info("Loaded project instructions from %s", candidate)

        parent = current.parent
        if parent == current:
            break
        current = parent

    return ProjectContext(
        instructions="\n\n".join(parts),
        source_files=tuple(sources),
    )
