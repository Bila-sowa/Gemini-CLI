"""Tool registry: registration, schema export, and safe dispatch.

The registry is the only place where model-generated tool calls are turned
into validated Python objects. Arguments come from an untrusted model, so every
call is validated against the tool's Pydantic schema and every expected failure
is returned as a :class:`ToolResult` instead of raising into the agent loop.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from gemini_cli.telemetry import ToolCallEvent, emit_tool_call
from gemini_cli.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Gemini function declarations must start with a letter or underscore and may
# contain letters, digits, underscores, and dashes, up to 64 characters.
TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")

MAX_ERROR_CHARS = 500

DispatchStatus = Literal["success", "timeout", "error"]


class ToolNotFoundError(LookupError):
    """Raised when a tool name is not registered."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        names = ", ".join(available) if available else "none"
        super().__init__(f"Unknown tool '{name}'. Available: {names}")


class DuplicateToolError(ValueError):
    """Raised when a tool name is already registered."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"A tool named '{name}' is already registered.")


class InvalidToolError(ValueError):
    """Raised when a tool does not satisfy the registry contract."""


def _truncate(text: str) -> str:
    if len(text) <= MAX_ERROR_CHARS:
        return text
    return f"{text[: MAX_ERROR_CHARS - 3]}..."


def _failure(message: str) -> ToolResult:
    return ToolResult(success=False, output="", error=_truncate(message))


def _format_validation_error(exc: ValidationError) -> str:
    """Describe validation failures without echoing the rejected input.

    Pydantic includes ``input_value`` by default, which could expose file
    contents, prompts, or secrets in the agent observation and logs.
    """
    parts: list[str] = []
    for error in exc.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = error.get("msg", "invalid value")
        parts.append(f"{location}: {message}" if location else message)
    return _truncate("; ".join(parts) or "invalid arguments")


def _status_for(exc: BaseException) -> DispatchStatus:
    return "timeout" if isinstance(exc, TimeoutError) else "error"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _execution_error(name: str, exc: BaseException) -> str:
    detail = str(exc).strip()
    message = f"Tool '{name}' failed: {type(exc).__name__}"
    if detail:
        message = f"{message}: {_truncate(detail)}"
    return message


class ToolRegistry:
    """Central registry for all tools available to the agent.

    Registration is expected during startup, before any dispatch happens. The
    registry does not synchronize concurrent registration.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool.

        Raises:
            InvalidToolError: if the tool name, description, or schema is
                invalid.
            DuplicateToolError: if a tool with the same name is registered.
        """
        name = tool.name
        if not name or not name.strip():
            msg = "Tool name must not be empty."
            raise InvalidToolError(msg)
        if TOOL_NAME_PATTERN.fullmatch(name) is None:
            msg = (
                f"Invalid tool name '{name}'. Names must start with a letter or "
                "underscore and contain only letters, digits, underscores, or "
                "dashes (maximum 64 characters)."
            )
            raise InvalidToolError(msg)
        if not tool.description.strip():
            msg = f"Tool '{name}' must have a non-empty description."
            raise InvalidToolError(msg)

        schema = tool.parameters_schema
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            msg = f"Tool '{name}' parameters_schema must be a Pydantic model class."
            raise InvalidToolError(msg)

        if name in self._tools:
            raise DuplicateToolError(name)

        self._tools[name] = tool
        logger.debug("Registered tool %s", name)

    def get(self, name: str) -> BaseTool:
        """Look up a tool by name.

        Raises:
            ToolNotFoundError: if the tool is not registered.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(name, sorted(self._tools)) from None

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools in registration order."""
        return list(self._tools.values())

    def export_schemas(self) -> list[dict[str, Any]]:
        """Export model-facing declarations for every registered tool.

        The shape matches :meth:`gemini_cli.llm.models.ToolDefinition.from_export`.
        Internal metadata such as risk level is intentionally not exported.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters_json_schema": tool.parameters_schema.model_json_schema(),
            }
            for tool in self._tools.values()
        ]

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Validate arguments, execute the tool, and return the result.

        Expected failures — unknown tool, invalid arguments, a raising tool, or
        an invalid return value — are returned as ``ToolResult(success=False)``.
        Cancellation is not caught.
        """
        try:
            tool = self.get(name)
        except ToolNotFoundError as exc:
            logger.warning("%s", exc)
            return _failure(str(exc))

        started = time.perf_counter()

        try:
            validated = tool.parameters_schema.model_validate(arguments)
        except ValidationError as exc:
            detail = _format_validation_error(exc)
            logger.warning("Invalid arguments for tool %s: %s", name, detail)
            self._emit_call(name, started, "error")
            return _failure(f"Invalid arguments for tool '{name}': {detail}")

        try:
            result = await tool.execute(validated)
        except Exception as exc:
            status = _status_for(exc)
            logger.exception("Tool %s failed", name)
            self._emit_call(name, started, status)
            return _failure(_execution_error(name, exc))

        if not isinstance(result, ToolResult):
            logger.error(
                "Tool %s returned %s instead of ToolResult",
                name,
                type(result).__name__,
            )
            self._emit_call(name, started, "error")
            return _failure(f"Tool '{name}' returned an invalid result.")

        self._emit_call(name, started, "success" if result.success else "error")
        return result

    @staticmethod
    def _emit_call(name: str, started: float, status: DispatchStatus) -> None:
        emit_tool_call(
            ToolCallEvent(
                tool_name=name,
                duration_ms=_elapsed_ms(started),
                status=status,
            )
        )


__all__ = [
    "DuplicateToolError",
    "InvalidToolError",
    "ToolNotFoundError",
    "ToolRegistry",
]
