"""Tests for the public configuration API."""

from __future__ import annotations

from pathlib import Path

import pytest

import gemini_cli.config as config
from gemini_cli.config import ProjectContext, Settings, load_project_context
from gemini_cli.config import project_context as project_context_module
from gemini_cli.config import settings as settings_module

EXPECTED_EXPORTS = {"ProjectContext", "Settings", "load_project_context"}


def test_public_names_are_importable() -> None:
    assert ProjectContext is not None
    assert Settings is not None
    assert load_project_context is not None


def test_all_lists_exactly_the_public_surface() -> None:
    assert set(config.__all__) == EXPECTED_EXPORTS
    assert len(config.__all__) == len(EXPECTED_EXPORTS)


def test_exports_are_the_same_objects_as_in_submodules() -> None:
    assert ProjectContext is project_context_module.ProjectContext
    assert load_project_context is project_context_module.load_project_context
    assert Settings is settings_module.Settings


def test_load_project_context_is_usable_through_package_import(
    tmp_path: Path,
) -> None:
    (tmp_path / "GEMINI.md").write_text("public api", encoding="utf-8")

    context = load_project_context(tmp_path)

    assert isinstance(context, ProjectContext)
    assert context.instructions == "public api"


def test_settings_is_usable_through_package_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    settings = Settings(_env_file=None)

    assert isinstance(settings, Settings)
    assert settings.api_key == ""
