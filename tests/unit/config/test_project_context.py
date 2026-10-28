"""Tests for project instruction discovery."""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from gemini_cli.config import project_context
from gemini_cli.config.project_context import ProjectContext, load_project_context


def write(path: Path, content: str) -> Path:
    """Write a UTF-8 text file, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_finds_instruction_file_in_search_root(tmp_path: Path) -> None:
    write(tmp_path / "GEMINI.md", "root instructions")

    context = load_project_context(tmp_path)

    assert context.instructions == "root instructions"
    assert list(context.source_files) == [(tmp_path / "GEMINI.md").resolve()]


def test_walks_up_to_parent_directory(tmp_path: Path) -> None:
    write(tmp_path / "AGENTS.md", "parent instructions")
    child = tmp_path / "src" / "package"
    child.mkdir(parents=True)

    context = load_project_context(child)

    assert context.instructions == "parent instructions"
    assert list(context.source_files) == [(tmp_path / "AGENTS.md").resolve()]


def test_preserves_declaration_order_within_directory(tmp_path: Path) -> None:
    write(tmp_path / "GEMINI.md", "gemini")
    write(tmp_path / "AGENTS.md", "agents")
    write(tmp_path / "AGENT.md", "agent")

    context = load_project_context(tmp_path)

    assert context.instructions == "gemini\n\nagents\n\nagent"
    assert list(context.source_files) == [
        (tmp_path / "GEMINI.md").resolve(),
        (tmp_path / "AGENTS.md").resolve(),
        (tmp_path / "AGENT.md").resolve(),
    ]


def test_orders_nearest_directory_first(tmp_path: Path) -> None:
    write(tmp_path / "GEMINI.md", "root")
    child = tmp_path / "child"
    write(child / "GEMINI.md", "child")

    context = load_project_context(child)

    assert context.instructions == "child\n\nroot"
    assert list(context.source_files) == [
        (child / "GEMINI.md").resolve(),
        (tmp_path / "GEMINI.md").resolve(),
    ]


def test_returns_empty_context_when_no_files(tmp_path: Path) -> None:
    context = load_project_context(tmp_path)

    assert context == ProjectContext()
    assert context.instructions == ""
    assert context.source_files == ()


def test_source_files_are_absolute(tmp_path: Path) -> None:
    write(tmp_path / "GEMINI.md", "content")

    context = load_project_context(tmp_path)

    assert all(path.is_absolute() for path in context.source_files)
    assert all(path == path.resolve() for path in context.source_files)


def test_skips_unreadable_file_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unreadable = write(tmp_path / "GEMINI.md", "unreadable")
    write(tmp_path / "AGENTS.md", "readable")
    original_read_text = Path.read_text

    def fake_read_text(
        self: Path,
        *args: object,
        **kwargs: object,
    ) -> str:
        if self == unreadable.resolve():
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    context = load_project_context(tmp_path)

    assert context.instructions == "readable"
    assert list(context.source_files) == [(tmp_path / "AGENTS.md").resolve()]


def test_skips_invalid_utf8_file(tmp_path: Path) -> None:
    (tmp_path / "GEMINI.md").write_bytes(b"\xff\xfe\x00")
    write(tmp_path / "AGENTS.md", "readable")

    context = load_project_context(tmp_path)

    assert context.instructions == "readable"
    assert list(context.source_files) == [(tmp_path / "AGENTS.md").resolve()]


def test_skips_symlinked_instruction_file(tmp_path: Path) -> None:
    target = write(tmp_path / "outside-instructions.md", "should not load")
    link = tmp_path / "GEMINI.md"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlinks are not available on this platform")

    context = load_project_context(tmp_path)

    assert context == ProjectContext()


def test_skips_oversized_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write(tmp_path / "GEMINI.md", "12345")
    write(tmp_path / "AGENTS.md", "ok")
    monkeypatch.setattr(project_context, "DEFAULT_MAX_FILE_SIZE_BYTES", 4)

    context = load_project_context(tmp_path)

    assert context.instructions == "ok"
    assert list(context.source_files) == [(tmp_path / "AGENTS.md").resolve()]


def test_accepts_file_at_size_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write(tmp_path / "GEMINI.md", "1234")
    monkeypatch.setattr(project_context, "DEFAULT_MAX_FILE_SIZE_BYTES", 4)

    context = load_project_context(tmp_path)

    assert context.instructions == "1234"


def test_search_root_may_be_a_file(tmp_path: Path) -> None:
    write(tmp_path / "GEMINI.md", "instructions")
    note = write(tmp_path / "note.txt", "not an instruction file")

    context = load_project_context(note)

    assert context.instructions == "instructions"


def test_project_context_is_frozen() -> None:
    context = ProjectContext(instructions="text")

    with pytest.raises(FrozenInstanceError):
        context.instructions = "changed"  # type: ignore[misc]
