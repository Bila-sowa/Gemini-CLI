"""System-instruction assembly and Jinja template rendering."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError

TEMPLATES_DIR = Path(__file__).parent / "templates"

ROLE_TEMPLATES: dict[str, str] = {
    "basic": "role_basic.jinja2",
    "code": "role_code.jinja2",
    "creator": "role_creator.jinja2",
    "analyser": "role_analyser.jinja2",
    "poet": "role_poet.jinja2",
}

# Markers used to isolate untrusted project instructions from trusted system
# instructions. Any copy embedded in the instructions themselves is removed so
# the untrusted block cannot be closed early.
_DELIMITER_PATTERN = re.compile(
    r"<<<\s*(?:END[\s_-]*)?UNTRUSTED[\s_-]*PROJECT[\s_-]*INSTRUCTIONS\s*>>>",
    re.IGNORECASE,
)


class PromptError(RuntimeError):
    """Raised when a system prompt cannot be assembled."""


class UnknownRoleError(PromptError):
    """Raised when the requested assistant role has no template."""

    def __init__(self, role: str) -> None:
        available = ", ".join(sorted(ROLE_TEMPLATES))
        super().__init__(f"Unknown role '{role}'. Available: {available}")
        self.role = role


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def neutralize_delimiters(text: str) -> str:
    """Remove untrusted-block markers so project text cannot break the block."""
    return _DELIMITER_PATTERN.sub("[delimiter removed from project instructions]", text)


def _render(environment: Environment, template_name: str, **context: Any) -> str:
    try:
        template = environment.get_template(template_name)
    except TemplateError as exc:
        msg = f"Prompt template '{template_name}' is missing or invalid: {exc}"
        raise PromptError(msg) from exc

    try:
        return template.render(context)
    except TemplateError as exc:
        msg = f"Failed to render prompt template '{template_name}': {exc}"
        raise PromptError(msg) from exc


def build_system_prompt(
    role: str,
    project_instructions: str = "",
    tool_names: Sequence[str] | None = None,
) -> str:
    """Render the system prompt for the active role.

    The prompt always starts with the shared rules and the role description.
    Tool instructions and project instructions are appended as separate
    sections. Project instructions are untrusted input: they are delimited and
    cannot override the shared rules, security controls, or approvals.

    Raises:
        UnknownRoleError: if *role* has no template.
        PromptError: if a template is missing or fails to render.
    """
    template_name = ROLE_TEMPLATES.get(role.strip().lower())
    if template_name is None:
        raise UnknownRoleError(role)

    environment = _environment()
    sections = [_render(environment, template_name)]

    if tool_names:
        sections.append(
            _render(environment, "tools_preamble.jinja2", tool_names=list(tool_names))
        )

    if project_instructions.strip():
        sections.append(
            _render(
                environment,
                "project_context.jinja2",
                instructions=neutralize_delimiters(project_instructions.strip()),
            )
        )

    return "\n\n".join(section.strip() for section in sections if section.strip())


__all__ = [
    "ROLE_TEMPLATES",
    "TEMPLATES_DIR",
    "PromptError",
    "UnknownRoleError",
    "build_system_prompt",
    "neutralize_delimiters",
]
