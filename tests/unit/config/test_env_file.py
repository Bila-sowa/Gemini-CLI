"""Tests for the narrow dotenv writer used by the TUI."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from gemini_cli.config import env_file as env_file_module
from gemini_cli.config.env_file import EnvFileError, set_env_value


def test_creates_new_file_with_private_permissions(tmp_path: Path) -> None:
    path = tmp_path / ".env"

    set_env_value(path, "GEMINI_MODEL", "gemini-4-argon")

    assert path.read_text(encoding="utf-8") == "GEMINI_MODEL=gemini-4-argon\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_updates_existing_value_and_preserves_other_lines(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "# local configuration\nGEMINI_API_KEY=secret\nGEMINI_MODEL=old\n",
        encoding="utf-8",
    )

    set_env_value(path, "GEMINI_MODEL", "gemini-4-carbon")

    assert path.read_text(encoding="utf-8") == (
        "# local configuration\nGEMINI_API_KEY=secret\nGEMINI_MODEL=gemini-4-carbon\n"
    )


def test_appends_missing_value(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY=secret\n", encoding="utf-8")

    set_env_value(path, "GEMINI_MODEL", "gemini-4-neon")

    assert path.read_text(encoding="utf-8") == (
        "GEMINI_API_KEY=secret\nGEMINI_MODEL=gemini-4-neon\n"
    )


def test_preserves_crlf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"# comment\r\nGEMINI_MODEL=old\r\n")

    set_env_value(path, "GEMINI_MODEL", "gemini-4-hydrogen")

    assert path.read_bytes() == b"# comment\r\nGEMINI_MODEL=gemini-4-hydrogen\r\n"


def test_preserves_missing_final_newline(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("GEMINI_MODEL=old", encoding="utf-8")

    set_env_value(path, "GEMINI_MODEL", "gemini-4-argon")

    assert path.read_text(encoding="utf-8") == "GEMINI_MODEL=gemini-4-argon"


def test_preserves_export_prefix(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("export GEMINI_MODEL=old\n", encoding="utf-8")

    set_env_value(path, "GEMINI_MODEL", "gemini-4-argon")

    assert path.read_text(encoding="utf-8") == "export GEMINI_MODEL=gemini-4-argon\n"


def test_preserves_inline_comment(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("GEMINI_MODEL=old # keep me\n", encoding="utf-8")

    set_env_value(path, "GEMINI_MODEL", "gemini-4-argon")

    assert path.read_text(encoding="utf-8") == "GEMINI_MODEL=gemini-4-argon # keep me\n"


def test_updates_case_insensitive_key_variants(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("gemini_model=old\n", encoding="utf-8")

    set_env_value(path, "GEMINI_MODEL", "gemini-4-argon")

    assert path.read_text(encoding="utf-8") == "GEMINI_MODEL=gemini-4-argon\n"


def test_updates_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("GEMINI_MODEL=old\nGEMINI_MODEL=older\n", encoding="utf-8")

    set_env_value(path, "GEMINI_MODEL", "gemini-4-argon")

    assert path.read_text(encoding="utf-8") == (
        "GEMINI_MODEL=gemini-4-argon\nGEMINI_MODEL=gemini-4-argon\n"
    )


def test_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.env"
    target.write_text("GEMINI_MODEL=old\n", encoding="utf-8")
    path = tmp_path / ".env"
    path.symlink_to(target)

    with pytest.raises(EnvFileError, match="symlink"):
        set_env_value(path, "GEMINI_MODEL", "gemini-4-argon")

    assert target.read_text(encoding="utf-8") == "GEMINI_MODEL=old\n"


def test_rejects_non_file_path(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.mkdir()

    with pytest.raises(EnvFileError, match="non-file"):
        set_env_value(path, "GEMINI_MODEL", "gemini-4-argon")


@pytest.mark.parametrize("key", ["", "GEMINI MODEL", "GEMINI.MODEL", "1MODEL"])
def test_rejects_invalid_keys(tmp_path: Path, key: str) -> None:
    with pytest.raises(EnvFileError, match="key"):
        set_env_value(tmp_path / ".env", key, "value")


@pytest.mark.parametrize("value", ["", "with space", "line\nbreak", "nul\0byte"])
def test_rejects_invalid_values(tmp_path: Path, value: str) -> None:
    with pytest.raises(EnvFileError, match="value"):
        set_env_value(tmp_path / ".env", "GEMINI_MODEL", value)


def test_rejects_non_utf8_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(EnvFileError, match="UTF-8"):
        set_env_value(path, "GEMINI_MODEL", "gemini-4-argon")


def test_failure_leaves_original_file_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / ".env"
    path.write_text("GEMINI_MODEL=old\n", encoding="utf-8")

    def failing_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("disk full")

    monkeypatch.setattr(env_file_module.os, "replace", failing_replace)

    with pytest.raises(OSError, match="disk full"):
        set_env_value(path, "GEMINI_MODEL", "gemini-4-argon")

    assert path.read_text(encoding="utf-8") == "GEMINI_MODEL=old\n"
    assert [entry.name for entry in tmp_path.iterdir()] == [".env"]


def test_preserves_existing_permissions(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("GEMINI_MODEL=old\n", encoding="utf-8")
    path.chmod(0o640)

    set_env_value(path, "GEMINI_MODEL", "gemini-4-argon")

    assert stat.S_IMODE(path.stat().st_mode) == 0o640
