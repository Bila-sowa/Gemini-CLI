"""System prompt composition — public surface."""

from gemini_cli.llm.prompts.system import (
    ROLE_TEMPLATES,
    TEMPLATES_DIR,
    PromptError,
    UnknownRoleError,
    build_system_prompt,
    neutralize_delimiters,
)

__all__ = [
    "ROLE_TEMPLATES",
    "TEMPLATES_DIR",
    "PromptError",
    "UnknownRoleError",
    "build_system_prompt",
    "neutralize_delimiters",
]
