"""Core agent runtime — public surface."""

from gemini_cli.core.agent import Agent, ChatClient
from gemini_cli.core.context_manager import (
    ContextManager,
    estimate_message_tokens,
    estimate_messages,
    estimate_tokens,
)
from gemini_cli.core.context_tracker import ContextWindowStatus, ContextWindowTracker
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import (
    AgentEvent,
    AgentEventType,
    done_event,
    error_event,
    from_stream_event,
    tool_call_end,
)

__all__ = [
    "Agent",
    "AgentEvent",
    "AgentEventType",
    "ChatClient",
    "ContextManager",
    "ContextWindowStatus",
    "ContextWindowTracker",
    "Session",
    "done_event",
    "error_event",
    "estimate_message_tokens",
    "estimate_messages",
    "estimate_tokens",
    "from_stream_event",
    "tool_call_end",
]
