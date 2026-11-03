"""Application runtime assembly for the CLI.

The runtime is built lazily by commands that actually need the agent. Commands
such as ``config show`` or ``--help`` must work before an API key exists, so
credential resolution is deliberately not part of application startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gemini_cli.auth.api_key import resolve_api_key
from gemini_cli.config import Settings, load_project_context
from gemini_cli.core.agent import Agent
from gemini_cli.core.context_manager import ContextManager
from gemini_cli.core.quota_tracker import QuotaTracker
from gemini_cli.llm import GeminiClient, get_model
from gemini_cli.llm.prompts import build_system_prompt
from gemini_cli.sandbox import (
    ApprovalCallback,
    ApprovalManager,
    SandboxBackend,
    select_backend,
)
from gemini_cli.tools import ToolRegistry
from gemini_cli.tools.code import CodeSearchTool
from gemini_cli.tools.filesystem import (
    EditFileTool,
    FindFilesTool,
    GrepTool,
    ReadFileTool,
    WriteFileTool,
)
from gemini_cli.tools.shell import RunCommandTool
from gemini_cli.tools.web import FetchUrlTool


@dataclass
class CliRuntime:
    """Everything one CLI invocation needs to talk to the agent."""

    settings: Settings
    system_prompt: str
    client: GeminiClient
    registry: ToolRegistry
    approval: ApprovalManager
    context: ContextManager
    quota: QuotaTracker
    agent: Agent

    def close(self) -> None:
        """Release the underlying LLM client."""
        self.client.close()


def register_builtin_tools(
    registry: ToolRegistry,
    settings: Settings,
    backend: SandboxBackend,
) -> None:
    """Register every built-in tool with the given registry."""
    tools = (
        ReadFileTool(settings),
        WriteFileTool(settings),
        EditFileTool(settings),
        FindFilesTool(settings),
        GrepTool(settings),
        FetchUrlTool(),
        CodeSearchTool(settings),
        RunCommandTool(settings, backend),
    )
    for tool in tools:
        registry.register(tool)


def _with_overrides(
    settings: Settings,
    model: str | None,
    role: str | None,
) -> Settings:
    updates: dict[str, object] = {}
    if model:
        updates["model"] = model
    if role:
        updates["role"] = role.strip().lower()
    if not updates:
        return settings
    return settings.model_copy(update=updates)


def _with_allowed_roots(settings: Settings) -> Settings:
    """Default filesystem access to the working directory.

    A configured list is never widened: only an empty list is replaced.
    """
    if settings.allowed_roots:
        return settings
    return settings.model_copy(update={"allowed_roots": [Path.cwd()]})


async def build_runtime(
    settings: Settings,
    *,
    model: str | None = None,
    role: str | None = None,
    approval_callback: ApprovalCallback | None = None,
) -> CliRuntime:
    """Resolve credentials and assemble the agent runtime."""
    resolved = _with_allowed_roots(_with_overrides(settings, model, role))
    resolved_key = resolve_api_key(resolved)

    project = load_project_context(Path.cwd())
    backend = await select_backend(resolved)

    registry = ToolRegistry()
    register_builtin_tools(registry, resolved, backend)

    system_prompt = build_system_prompt(
        resolved.role,
        project.instructions,
        [tool.name for tool in registry.list_tools()],
    )

    model_info = get_model(resolved.model)
    context = ContextManager(
        max_tokens=max(1, model_info.context_window - resolved.max_output_tokens),
    )
    quota = QuotaTracker(
        five_hour_limit=resolved.quota_5h_token_limit,
        weekly_limit=resolved.quota_weekly_token_limit,
    )
    client = GeminiClient(resolved, api_key=resolved_key.key)
    approval = ApprovalManager(approval_callback)
    agent = Agent(
        client,
        registry,
        approval,
        context,
        quota,
        resolved,
        system_prompt,
    )

    return CliRuntime(
        settings=resolved,
        system_prompt=system_prompt,
        client=client,
        registry=registry,
        approval=approval,
        context=context,
        quota=quota,
        agent=agent,
    )


__all__ = ["CliRuntime", "build_runtime", "register_builtin_tools"]
