"""Deterministic streaming client for agent-loop tests.

Every response is a scripted list of ``StreamEvent`` objects. The fake records
the arguments of each call so tests can assert which history, tools, and
system instruction the agent sent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from gemini_cli.llm import (
    LLMError,
    Message,
    StreamEvent,
    ToolCallRequest,
    ToolDefinition,
    UsageInfo,
)

ResponseScript = Sequence[StreamEvent]


def usage(
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> UsageInfo:
    """Build consistent usage metadata."""
    return UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def text_response(
    text: str,
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[StreamEvent]:
    """Scripted response that contains text only."""
    return [
        StreamEvent("text_delta", data=text),
        StreamEvent("usage", data=usage(input_tokens, output_tokens)),
        StreamEvent("done", data="fake-model"),
    ]


def thinking_response(
    thinking: str,
    text: str,
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[StreamEvent]:
    """Scripted response with a reasoning part followed by visible text."""
    return [
        StreamEvent("text_delta", data=thinking, metadata={"thought": True}),
        StreamEvent("text_delta", data=text),
        StreamEvent("usage", data=usage(input_tokens, output_tokens)),
        StreamEvent("done", data="fake-model"),
    ]


def tool_call_response(
    calls: Sequence[ToolCallRequest],
    *,
    text: str = "",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[StreamEvent]:
    """Scripted response that requests one or more tools."""
    events: list[StreamEvent] = []
    if text:
        events.append(StreamEvent("text_delta", data=text))
    events.extend(StreamEvent("tool_call_delta", data=call) for call in calls)
    events.append(StreamEvent("usage", data=usage(input_tokens, output_tokens)))
    events.append(StreamEvent("done", data="fake-model"))
    return events


def error_response(
    message: str,
    *,
    status: str = "error",
) -> list[StreamEvent]:
    """Scripted response that fails before producing content."""
    return [
        StreamEvent(
            "error",
            data=LLMError(message),
            metadata={"status": status},
        ),
        StreamEvent("usage", data=UsageInfo()),
        StreamEvent("done", data="fake-model"),
    ]


class FakeLLMClient:
    """Return scripted responses in order and record each request."""

    def __init__(self, responses: Sequence[ResponseScript | BaseException]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate_stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None = None,
        system_instruction: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools or []),
                "system_instruction": system_instruction,
            }
        )
        if not self._responses:
            msg = "FakeLLMClient has no scripted response left."
            raise AssertionError(msg)

        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response

        for event in response:
            yield event

    @property
    def remaining(self) -> int:
        """Number of scripted responses that were not consumed."""
        return len(self._responses)


__all__ = [
    "FakeLLMClient",
    "error_response",
    "text_response",
    "thinking_response",
    "tool_call_response",
    "usage",
]
