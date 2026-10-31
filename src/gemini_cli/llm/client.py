"""Async Gemini client: request mapping, streaming, retry, and fallback.

The client is the only module in ``gemini_cli.llm`` that imports the provider
SDK. Everything it accepts and returns uses the package's own domain types, so
higher layers never depend on ``google.genai``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from google import genai
from google.genai import types

from gemini_cli.config import Settings
from gemini_cli.llm.models import (
    ModelInfo,
    ToolDefinition,
    UnknownModelError,
    get_fallback,
    get_model,
)
from gemini_cli.llm.retry import RetryPolicy, status_code, with_retry
from gemini_cli.telemetry import ModelCallEvent, emit_model_call

logger = logging.getLogger(__name__)

MessageRole = Literal["user", "assistant", "tool"]
StreamEventType = Literal["text_delta", "tool_call_delta", "usage", "done", "error"]
CallStatus = Literal["success", "error", "timeout", "rate_limited"]

MAX_ERROR_DETAIL_CHARS = 300
_FALLBACK_STATUS_CODES = frozenset({404, 429, 500, 503})
_PARTIAL_SEGMENT = re.compile(r"^([^\[\]]+)(?:\[(\d+)\])?$")


@dataclass(frozen=True)
class ToolCallRequest:
    """A complete function call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """Application-level conversation message.

    ``name`` is required for ``role="tool"`` messages because Gemini needs the
    function name to attach a response. ``tool_calls`` carries the calls of an
    assistant turn so the history can be replayed to the model.
    """

    role: MessageRole
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: tuple[ToolCallRequest, ...] = ()


@dataclass(frozen=True)
class UsageInfo:
    """Token accounting reported by the model."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ModelResponse:
    """A single non-streaming model response."""

    content: str | None
    tool_calls: list[ToolCallRequest]
    usage: UsageInfo
    model: str
    finish_reason: str


@dataclass(frozen=True)
class StreamEvent:
    """One event of a streaming response."""

    event_type: StreamEventType
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised when a Gemini request fails irrecoverably."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _create_client(settings: Settings, api_key: str) -> genai.Client:
    """Build the SDK client; isolated so tests can substitute a fake."""
    http_options = types.HttpOptions(timeout=settings.request_timeout_ms)
    return genai.Client(api_key=api_key, http_options=http_options)


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()


def _call_status(exc: BaseException) -> CallStatus:
    if _is_timeout(exc):
        return "timeout"
    if status_code(exc) == 429:
        return "rate_limited"
    return "error"


def _to_llm_error(exc: BaseException) -> LLMError:
    if isinstance(exc, LLMError):
        return exc

    code = status_code(exc)
    detail = str(exc).strip()
    if len(detail) > MAX_ERROR_DETAIL_CHARS:
        detail = f"{detail[: MAX_ERROR_DETAIL_CHARS - 3]}..."
    prefix = (
        f"Gemini request failed (status {code})" if code else "Gemini request failed"
    )
    message = f"{prefix}: {detail}" if detail else prefix
    return LLMError(message, status=code)


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _apply_partial_arg(arguments: dict[str, Any], partial: types.PartialArg) -> None:
    """Merge one streamed ``PartialArg`` into *arguments* using its JSON path."""
    path = (partial.json_path or "").strip()
    if path in {"", "$"}:
        return

    if path.startswith("$."):
        path = path[2:]
    raw_segments = path.lstrip("$.").split(".")

    segments: list[tuple[str, str | None]] = []
    for raw_segment in raw_segments:
        match = _PARTIAL_SEGMENT.match(raw_segment)
        if match is None:
            return
        segments.append((match.group(1), match.group(2)))

    if partial.string_value is not None:
        value: Any = partial.string_value
    elif partial.number_value is not None:
        value = partial.number_value
    elif partial.bool_value is not None:
        value = partial.bool_value
    else:
        value = None

    container: Any = arguments
    for index, (key, position) in enumerate(segments):
        if not isinstance(container, dict):
            return
        is_last = index == len(segments) - 1
        next_position = None if is_last else segments[index + 1][1]
        empty_child: Any = [] if next_position is not None else {}

        if position is None:
            if is_last:
                container[key] = value
                return
            child = container.get(key)
            if not isinstance(child, (dict, list)):
                child = empty_child
                container[key] = child
            container = child
            continue

        items = container.get(key)
        if not isinstance(items, list):
            items = []
            container[key] = items
        slot = int(position)
        while len(items) <= slot:
            items.append(None)
        if is_last:
            items[slot] = value
            return
        child = items[slot]
        if not isinstance(child, (dict, list)):
            child = empty_child
            items[slot] = child
        container = child


def _arguments_from_call(call: types.FunctionCall) -> dict[str, Any]:
    arguments = dict(call.args or {})
    for partial in call.partial_args or []:
        _apply_partial_arg(arguments, partial)
    return arguments


def _tool_call_from_call(call: types.FunctionCall, index: int) -> ToolCallRequest:
    return ToolCallRequest(
        id=call.id or f"call_{index}",
        name=call.name or "",
        arguments=_arguments_from_call(call),
    )


@dataclass
class _PendingCall:
    """Builder for a function call whose arguments arrive in several parts."""

    name: str
    call_id: str
    arguments: dict[str, Any] = field(default_factory=dict)


class ToolCallAccumulator:
    """Rebuild complete tool calls from streamed chunks.

    Gemini can split one call across chunks. Only fully assembled calls are
    emitted, so a partial or malformed argument object never reaches tool
    dispatch.
    """

    def __init__(self) -> None:
        self._pending: _PendingCall | None = None
        self._counter = 0

    def feed(self, call: types.FunctionCall) -> ToolCallRequest | None:
        """Add a streamed function-call part; return a call when it is complete."""
        name = call.name or ""
        if self._pending is None or (name and self._pending.name != name):
            self._counter += 1
            self._pending = _PendingCall(
                name=name,
                call_id=call.id or f"call_{self._counter}",
            )

        self._pending.arguments.update(_arguments_from_call(call))

        if call.will_continue:
            return None

        completed = self._pending
        self._pending = None
        return ToolCallRequest(
            id=completed.call_id,
            name=completed.name,
            arguments=dict(completed.arguments),
        )

    def flush(self) -> ToolCallRequest | None:
        """Return the pending call, if the stream ended without completion."""
        if self._pending is None:
            return None
        completed = self._pending
        self._pending = None
        return ToolCallRequest(
            id=completed.call_id,
            name=completed.name,
            arguments=dict(completed.arguments),
        )


def _usage_from(
    metadata: types.GenerateContentResponseUsageMetadata | None,
) -> UsageInfo:
    if metadata is None:
        return UsageInfo()

    input_tokens = metadata.prompt_token_count or 0
    output_tokens = metadata.candidates_token_count or 0
    total_tokens = metadata.total_token_count or (input_tokens + output_tokens)
    return UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=metadata.cached_content_token_count or 0,
        total_tokens=total_tokens,
    )


def _contents_from(messages: Sequence[Message]) -> list[types.Content]:
    contents: list[types.Content] = []
    for message in messages:
        if message.role == "user":
            contents.append(
                types.Content(role="user", parts=[types.Part(text=message.content)])
            )
        elif message.role == "assistant":
            parts: list[types.Part] = []
            if message.content:
                parts.append(types.Part(text=message.content))
            parts.extend(
                types.Part(
                    function_call=types.FunctionCall(
                        id=call.id,
                        name=call.name,
                        args=dict(call.arguments),
                    )
                )
                for call in message.tool_calls
            )
            if not parts:
                parts.append(types.Part(text=""))
            contents.append(types.Content(role="model", parts=parts))
        else:
            if not message.name:
                msg = "Tool messages require the tool 'name'."
                raise ValueError(msg)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=message.tool_call_id,
                                name=message.name,
                                response={"result": message.content},
                            )
                        )
                    ],
                )
            )
    return contents


def _tools_from(tools: Sequence[ToolDefinition]) -> list[types.Tool]:
    declarations = [
        types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.parameters_json_schema or None,
        )
        for tool in tools
    ]
    return [types.Tool(function_declarations=declarations)]


def _response_from(
    response: types.GenerateContentResponse,
    model_name: str,
) -> ModelResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []
    finish_reason = ""

    candidates = response.candidates or []
    if candidates:
        candidate = candidates[0]
        if candidate.content and candidate.content.parts:
            for index, part in enumerate(candidate.content.parts):
                if part.function_call is not None:
                    tool_calls.append(_tool_call_from_call(part.function_call, index))
                elif part.text and not part.thought:
                    text_parts.append(part.text)
        if candidate.finish_reason is not None:
            finish_reason = _enum_value(candidate.finish_reason)

    content = "".join(text_parts) or None
    return ModelResponse(
        content=content,
        tool_calls=tool_calls,
        usage=_usage_from(response.usage_metadata),
        model=model_name,
        finish_reason=finish_reason,
    )


class GeminiClient:
    """Async wrapper around the Gemini SDK with retry, fallback, and telemetry."""

    def __init__(self, settings: Settings, *, api_key: str) -> None:
        key = api_key.strip()
        if not key:
            msg = "A Gemini API key is required."
            raise ValueError(msg)

        self._settings = settings
        self._api_key = key
        self._client: genai.Client | None = None

    def __repr__(self) -> str:
        return f"GeminiClient(model={self.model!r})"

    @property
    def model(self) -> str:
        """The configured model name."""
        return self._settings.model

    def close(self) -> None:
        """Release the underlying SDK client and its HTTP resources."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def _sdk_client(self) -> genai.Client:
        if self._client is None:
            self._client = _create_client(self._settings, self._api_key)
        return self._client

    def _retry_policy(self) -> RetryPolicy:
        return RetryPolicy(max_attempts=max(1, self._settings.retry_attempts))

    def _timeout_seconds(self) -> float:
        return max(0.001, self._settings.request_timeout_ms / 1000)

    def _build_config(
        self,
        system_instruction: str | None,
        tools: Sequence[ToolDefinition] | None,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            max_output_tokens=self._settings.max_output_tokens,
            tools=cast("Any", _tools_from(tools)) if tools else None,
        )

    async def _call_generate(
        self,
        model_name: str,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        async with asyncio.timeout(self._timeout_seconds()):
            return await self._sdk_client().aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )

    def _emit_call(
        self,
        model_name: str,
        duration_seconds: float,
        usage: UsageInfo,
        status: CallStatus,
        retry_count: int,
    ) -> None:
        emit_model_call(
            ModelCallEvent(
                model=model_name,
                duration_ms=round(duration_seconds * 1000, 3),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_tokens=usage.cached_tokens,
                status=status,
                retry_count=retry_count,
            )
        )

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None = None,
        system_instruction: str | None = None,
    ) -> ModelResponse:
        """Send a conversation and return the complete model response.

        Transient failures are retried according to the configured policy.
        When retries are exhausted on a supported error, the fallback model from
        the registry is tried exactly once.

        Raises:
            UnknownModelError: if the configured model is not registered.
            LLMError: if every attempt fails.
        """
        contents = _contents_from(messages)
        config = self._build_config(system_instruction, tools)
        policy = self._retry_policy()
        current = get_model(self._settings.model)

        while True:
            attempts = 0
            started = time.perf_counter()

            async def attempt(
                model_name: str = current.name,
            ) -> types.GenerateContentResponse:
                nonlocal attempts
                attempts += 1
                return await self._call_generate(model_name, contents, config)

            try:
                response = await with_retry(attempt, policy)
            except Exception as exc:
                self._emit_call(
                    current.name,
                    time.perf_counter() - started,
                    UsageInfo(),
                    _call_status(exc),
                    max(0, attempts - 1),
                )
                fallback = self._fallback_for(current, exc)
                if fallback is None:
                    raise _to_llm_error(exc) from exc
                logger.warning(
                    "Model %s failed (%s); falling back to %s",
                    current.name,
                    type(exc).__name__,
                    fallback.name,
                )
                current = fallback
                continue

            usage = _usage_from(response.usage_metadata)
            self._emit_call(
                current.name,
                time.perf_counter() - started,
                usage,
                "success",
                max(0, attempts - 1),
            )
            return _response_from(response, current.name)

    @staticmethod
    def _fallback_for(current: ModelInfo, exc: BaseException) -> ModelInfo | None:
        if isinstance(exc, UnknownModelError):
            return None
        if status_code(exc) not in _FALLBACK_STATUS_CODES:
            return None
        return get_fallback(current.name)

    async def generate_stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None = None,
        system_instruction: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a model response as domain events.

        Text is yielded as soon as it arrives. Tool calls are buffered until
        their arguments are complete. A stream is not retried, because replaying
        it would duplicate text already shown to the user.

        Raises:
            UnknownModelError: if the configured model is not registered.
        """
        model_name = get_model(self._settings.model).name
        contents = _contents_from(messages)
        config = self._build_config(system_instruction, tools)
        accumulator = ToolCallAccumulator()
        usage = UsageInfo()
        status: CallStatus = "success"
        started = time.perf_counter()

        try:
            stream = await self._sdk_client().aio.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )
            async for chunk in stream:
                for event in self._events_from_chunk(chunk, accumulator):
                    if event.event_type == "usage":
                        usage = event.data
                        continue
                    yield event

            remaining = accumulator.flush()
            if remaining is not None:
                yield StreamEvent("tool_call_delta", data=remaining)
        except Exception as exc:
            status = _call_status(exc)
            yield StreamEvent(
                "error",
                data=_to_llm_error(exc),
                metadata={"status": status},
            )
        finally:
            self._emit_call(
                model_name,
                time.perf_counter() - started,
                usage,
                status,
                0,
            )

        yield StreamEvent("usage", data=usage)
        yield StreamEvent("done", data=model_name)

    @staticmethod
    def _events_from_chunk(
        chunk: types.GenerateContentResponse,
        accumulator: ToolCallAccumulator,
    ) -> Iterator[StreamEvent]:
        candidates = chunk.candidates or []
        if candidates:
            candidate = candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.function_call is not None:
                        completed = accumulator.feed(part.function_call)
                        if completed is not None:
                            yield StreamEvent("tool_call_delta", data=completed)
                    elif part.text:
                        metadata = {"thought": True} if part.thought else {}
                        yield StreamEvent(
                            "text_delta", data=part.text, metadata=metadata
                        )

        if chunk.usage_metadata is not None:
            yield StreamEvent("usage", data=_usage_from(chunk.usage_metadata))


__all__ = [
    "GeminiClient",
    "LLMError",
    "Message",
    "ModelResponse",
    "StreamEvent",
    "ToolCallAccumulator",
    "ToolCallRequest",
    "UsageInfo",
]
