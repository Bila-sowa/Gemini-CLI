"""Acceptance checks: every package module must be importable."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import gemini_cli

PACKAGE_ROOT = Path(gemini_cli.__file__).parent


def _module_names() -> list[str]:
    """Collect dotted module names for every file in the package."""
    names: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


@pytest.mark.parametrize("module_name", _module_names())
def test_module_imports(module_name: str) -> None:
    """Importing a module must not require credentials or configuration."""
    importlib.import_module(module_name)
