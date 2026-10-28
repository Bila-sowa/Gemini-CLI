"""Configuration package — public surface."""

from gemini_cli.config.project_context import ProjectContext, load_project_context
from gemini_cli.config.settings import Settings

__all__ = ["ProjectContext", "Settings", "load_project_context"]
