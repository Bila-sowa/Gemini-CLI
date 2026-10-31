"""Deterministic SDK-shaped fixtures for LLM client tests.

The builders produce real ``google.genai`` response objects so the client's
mapping code is exercised, but no network call is ever made.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from types import SimpleNamespace
from typing import Any

from google.genai import types


def make_usage(
    *,
    prompt: int = 10,
    candidates: int = 5,
    cached: int = 0,
    total: int | None = None,
) -> types.GenerateContentResponseUsageMetadata:
    """Build token usage metadata."""
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        cached_content_token_count=cached,
        total_token_count=total if total is not None else prompt + candidates,
    )


def make_function_call(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    call_id: str | None = None,
    partial_args: Sequence[types.PartialArg] | None = None,
    will_continue: bool | None = None,
) -> types.FunctionCall:
    """Build a function call part payload."""
    return types.FunctionCall(
        id=call_id,
        name=name,
        args=arguments,
        partial_args=list(partial_args) if partial_args is not None else None,
        will_continue=will_continue,
    )


def make_text_chunk(
    text: str,
    *,
    thought: bool = False,
    usage: types.GenerateContentResponseUsageMetadata | None = None,
) -> types.GenerateContentResponse:
    """Build a streamed text chunk."""
    part = types.Part(text=text, thought=True if thought else None)
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))],
        usage_metadata=usage,
    )


def make_function_call_chunk(
    call: types.FunctionCall,
    *,
    usage: types.GenerateContentResponseUsageMetadata | None = None,
) -> types.GenerateContentResponse:
    """Build a streamed function-call chunk."""
    part = types.Part(function_call=call)
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))],
        usage_metadata=usage,
    )


def make_response(
    *,
    text: str | None = None,
    function_calls: Iterable[types.FunctionCall] = (),
    usage: types.GenerateContentResponseUsageMetadata | None = None,
    finish_reason: types.FinishReason | None = types.FinishReason.STOP,
) -> types.GenerateContentResponse:
    """Build a complete (non-streaming) model response."""
    parts: list[types.Part] = []
    if text:
        parts.append(types.Part(text=text))
    parts.extend(types.Part(function_call=call) for call in function_calls)

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=parts),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=usage,
    )


class FakeModels:
    """Async stand-in for ``client.aio.models``."""

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.generate_responses: list[types.GenerateContentResponse] = []
        self.generate_errors: list[Exception] = []
        self.chunks: list[types.GenerateContentResponse] = []
        self.stream_error: Exception | None = None

    async def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.generate_calls.append(kwargs)
        if self.generate_errors:
            raise self.generate_errors.pop(0)
        return self.generate_responses.pop(0)

    async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        self.stream_calls.append(kwargs)
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[types.GenerateContentResponse]:
        for chunk in self.chunks:
            yield chunk
        if self.stream_error is not None:
            raise self.stream_error


class FakeSdkClient:
    """Minimal replacement for ``google.genai.Client``."""

    def __init__(self) -> None:
        self.models = FakeModels()
        self.aio = SimpleNamespace(models=self.models)
        self.closed = False

    def close(self) -> None:
        self.closed = True
