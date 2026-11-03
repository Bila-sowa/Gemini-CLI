"""Convert LLM streaming events into stable internal events.

The interactive TUI and the non-interactive command consume :class:`AgentEvent`
objects. They never import the Gemini SDK or the LLM client's raw
:class:`~gemini_cli.llm.StreamEvent` type, so a provider change stays contained
inside the LLM package.

Usage and completion markers from the client are intentionally not converted
here: the agent loop consumes them to update accounting and decides when the
whole turn is done.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from gemini_cli.llm import StreamEvent, ToolCallRequest
from gemini_cli.tools.base import ToolResult


class AgentEventType(StrEnum):
    """Kinds of event emitted by the agent loop."""

    TEXT_CHUNK = "text_chunk"
    """Partial text from the model."""

    TOOL_CALL_START = "tool_call_start"
    """The model requested a tool and the agent starts handling it."""

    TOOL_CALL_END = "tool_call_end"
    """Tool handling completed, including denied and failed calls."""

    THINKING = "thinking"
    """Model reasoning, when the provider exposes it."""

    ERROR = "error"
    """A recoverable error the UI should show before the turn ends."""

    DONE = "done"
    """The agent turn is complete."""


@dataclass(frozen=True)
class AgentEvent:
    """One stable event consumed by presentation layers."""

    event_type: AgentEventType
    data: Any = None
    metadata: dict[str, Any] | None = None


def from_stream_event(event: StreamEvent) -> AgentEvent | None:
    """Map one LLM stream event to an agent event, if it is user-visible.

    ``usage`` and ``done`` events return ``None`` because they are bookkeeping
    for the agent loop rather than output for the user.
    """
    if event.event_type == "text_delta":
        text = event.data if isinstance(event.data, str) else ""
        if event.metadata.get("thought"):
            return AgentEvent(AgentEventType.THINKING, text)
        return AgentEvent(AgentEventType.TEXT_CHUNK, text)

    if event.event_type == "tool_call_delta":
        call = event.data
        if not isinstance(call, ToolCallRequest):
            return None
        return AgentEvent(
            AgentEventType.TOOL_CALL_START,
            call,
            {"tool": call.name, "call_id": call.id},
        )

    if event.event_type == "error":
        metadata = dict(event.metadata) if event.metadata else {}
        return AgentEvent(
            AgentEventType.ERROR,
            str(event.data),
            metadata or None,
        )

    return None


def tool_call_end(
    call: ToolCallRequest,
    result: ToolResult,
    *,
    status: str,
) -> AgentEvent:
    """Build the completion event for one tool call.

    ``status`` is a short machine-readable label such as ``success``,
    ``denied``, or ``error``.
    """
    return AgentEvent(
        AgentEventType.TOOL_CALL_END,
        result,
        {"tool": call.name, "call_id": call.id, "status": status},
    )


def done_event() -> AgentEvent:
    """Build the event that closes an agent turn."""
    return AgentEvent(AgentEventType.DONE)


def error_event(message: str, *, recoverable: bool = True) -> AgentEvent:
    """Build an error event with a user-safe message."""
    return AgentEvent(
        AgentEventType.ERROR,
        message,
        {"recoverable": recoverable},
    )


__all__ = [
    "AgentEvent",
    "AgentEventType",
    "done_event",
    "error_event",
    "from_stream_event",
    "tool_call_end",
]
