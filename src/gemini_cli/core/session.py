"""Conversation session state.

A session owns everything that belongs to one conversation: ordered messages,
assistant tool calls, tool observations, cumulative token usage, and the
iteration counter used by the agent loop's guard rail.

It deliberately reuses the LLM client's message types instead of defining a
second conversation model. That keeps history replay exact: the same objects
the agent builds are the ones the client converts into provider requests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from gemini_cli.llm import Message, ToolCallRequest, UsageInfo
from gemini_cli.tools.base import ToolResult

# A tool observation becomes part of the next model request. Cap it here as a
# last line of defence; individual tools already bound their own output.
MAX_OBSERVATION_CHARS = 16_384


def _truncate(text: str) -> str:
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    return f"{text[: MAX_OBSERVATION_CHARS - 3]}..."


def _observation_text(result: ToolResult) -> str:
    """Render a structured tool result as the text sent back to the model."""
    if result.success:
        return _truncate(result.output)

    detail = result.error or "Tool failed without an error message."
    if result.output:
        detail = f"{detail}\n\n{result.output}"
    return _truncate(detail)


@dataclass
class Session:
    """Holds the complete state of one conversation."""

    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=datetime.now)
    messages: list[Message] = field(default_factory=list)
    model: str = ""
    role: str = "basic"
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    iteration_count: int = 0

    # Tool name by call id. Gemini requires the function name on every tool
    # response, but ``add_tool_result`` only receives the call id.
    _tool_names: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def add_user_message(self, content: str) -> Message:
        """Append a user message and return it."""
        message = Message(role="user", content=content)
        self.messages.append(message)
        return message

    def add_assistant_message(
        self,
        content: str = "",
        tool_calls: Sequence[ToolCallRequest] = (),
    ) -> Message:
        """Append an assistant turn with optional tool calls."""
        calls = tuple(tool_calls)
        message = Message(role="assistant", content=content, tool_calls=calls)
        self.messages.append(message)
        self._remember(calls)
        return message

    def add_tool_call(self, tool_call: ToolCallRequest) -> Message:
        """Attach one tool call to the trailing assistant turn.

        If the conversation does not end with an assistant message, a new
        assistant turn is started. This matches how providers emit function
        calls: one model turn can request several tools.
        """
        last = self.last_message()
        if last is None or last.role != "assistant":
            return self.add_assistant_message(tool_calls=(tool_call,))

        message = Message(
            role="assistant",
            content=last.content,
            tool_call_id=last.tool_call_id,
            name=last.name,
            tool_calls=(*last.tool_calls, tool_call),
        )
        self.messages[-1] = message
        self._remember((tool_call,))
        return message

    def add_tool_result(self, call_id: str, result: ToolResult) -> Message:
        """Append the observation for a previously recorded tool call.

        Raises:
            ValueError: if *call_id* does not belong to this session.
        """
        try:
            name = self._tool_names[call_id]
        except KeyError:
            msg = f"No tool call with id {call_id!r} in this session."
            raise ValueError(msg) from None

        message = Message(
            role="tool",
            content=_observation_text(result),
            tool_call_id=call_id,
            name=name,
        )
        self.messages.append(message)
        return message

    def get_history(self) -> list[Message]:
        """Return a copy of the conversation history.

        The copy prevents callers from appending to the session's list while
        the agent is iterating over it.
        """
        return list(self.messages)

    def update_usage(self, usage: UsageInfo) -> None:
        """Add one response's token usage to the session totals."""
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens

    def record_iteration(self) -> int:
        """Count one think-act-observe cycle and return the new total."""
        self.iteration_count += 1
        return self.iteration_count

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed by this session."""
        return self.total_input_tokens + self.total_output_tokens

    def last_message(self) -> Message | None:
        """Return the most recent message, if any."""
        return self.messages[-1] if self.messages else None

    def _remember(self, calls: Sequence[ToolCallRequest]) -> None:
        for call in calls:
            self._tool_names[call.id] = call.name


__all__ = ["MAX_OBSERVATION_CHARS", "Session"]
