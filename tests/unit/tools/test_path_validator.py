"""Tests for filesystem path validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemini_cli.tools.filesystem import PathSecurityError, validate_path


def test_accepts_path_inside_root(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")

    resolved = validate_path(target, [tmp_path], must_exist=True, must_be_file=True)

    assert resolved == target.resolve()


def test_accepts_path_when_root_is_not_resolved(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")

    resolved = validate_path(target, [tmp_path / "sub" / ".."])

    assert resolved == target.resolve()


def test_normalizes_parent_segments_staying_inside_root(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")

    resolved = validate_path(tmp_path / "sub" / ".." / "file.txt", [tmp_path])

    assert resolved == target.resolve()


def test_rejects_path_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    target = outside / "file.txt"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(PathSecurityError, match="outside the allowed roots"):
        validate_path(target, [tmp_path])


def test_rejects_traversal_escaping_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    target = outside / "file.txt"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(PathSecurityError):
        validate_path(tmp_path / ".." / outside.name / "file.txt", [tmp_path])


def test_rejects_symlink_pointing_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    target = outside / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    with pytest.raises(PathSecurityError):
        validate_path(link, [tmp_path])


def test_accepts_symlink_pointing_inside_root(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    resolved = validate_path(link, [tmp_path], must_exist=True, must_be_file=True)

    assert resolved == target.resolve()


def test_rejects_everything_without_allowed_roots(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(PathSecurityError, match="No allowed roots"):
        validate_path(target, [])


@pytest.mark.parametrize("path", ["", "   "])
def test_rejects_empty_path(path: str, tmp_path: Path) -> None:
    with pytest.raises(PathSecurityError, match="empty"):
        validate_path(path, [tmp_path])


def test_rejects_nul_byte(tmp_path: Path) -> None:
    with pytest.raises(PathSecurityError, match="NUL"):
        validate_path(f"{tmp_path}/bad\x00name", [tmp_path])


def test_requires_existing_path_when_requested(tmp_path: Path) -> None:
    with pytest.raises(PathSecurityError, match="does not exist"):
        validate_path(tmp_path / "missing.txt", [tmp_path], must_exist=True)


def test_requires_file_when_requested(tmp_path: Path) -> None:
    directory = tmp_path / "folder"
    directory.mkdir()

    with pytest.raises(PathSecurityError, match="not a regular file"):
        validate_path(directory, [tmp_path], must_exist=True, must_be_file=True)


def test_accepts_path_under_second_root(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    target = second / "file.txt"
    target.write_text("x", encoding="utf-8")

    resolved = validate_path(target, [first, second])

    assert resolved == target.resolve()
