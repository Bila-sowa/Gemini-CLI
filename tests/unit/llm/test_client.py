"""Tests for the async Gemini client: mapping, retry, fallback, streaming."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, get_type_hints

import pytest
from google.genai import types

from gemini_cli.config import Settings
from gemini_cli.llm import client as client_module
from gemini_cli.llm import retry as retry_module
from gemini_cli.llm.client import (
    GeminiClient,
    LLMError,
    Message,
    ModelResponse,
    StreamEvent,
    ToolCallAccumulator,
    ToolCallRequest,
    UsageInfo,
)
from gemini_cli.llm.models import ToolDefinition
from gemini_cli.telemetry.logger import (
    LOGGER_NAME,
    TELEMETRY_LOGGER_NAME,
    configure_logging,
)
from tests.fixtures.fake_gemini import (
    FakeSdkClient,
    make_function_call,
    make_function_call_chunk,
    make_response,
    make_text_chunk,
    make_usage,
)

TEST_KEY = "test-api-key"


class FakeApiError(Exception):
    """HTTP-shaped error raised by the fake SDK."""

    def __init__(self, code: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"HTTP {code}")
        self.status_code = code
        self.response = SimpleNamespace(status_code=code, headers=dict(headers or {}))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep retry backoff instantaneous in tests."""

    async def fake_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(retry_module, "asyncio", SimpleNamespace(sleep=fake_sleep))


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for name in ("GEMINI_MODEL", "GEMINI_RETRY_ATTEMPTS", "GEMINI_TELEMETRY_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    return Settings(_env_file=None)


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> FakeSdkClient:
    fake = FakeSdkClient()
    monkeypatch.setattr(client_module, "_create_client", lambda settings, api_key: fake)
    return fake


@pytest.fixture
def client(settings: Settings, fake_sdk: FakeSdkClient) -> GeminiClient:
    return GeminiClient(settings, api_key=TEST_KEY)


@pytest.fixture
def telemetry_logs(tmp_path) -> Iterator[Any]:
    log_dir = tmp_path / "logs"
    configure_logging("DEBUG", True, log_dir=log_dir)
    yield log_dir
    for name in (LOGGER_NAME, TELEMETRY_LOGGER_NAME):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.disabled = False


async def test_client_can_be_instantiated_with_a_dummy_key(settings: Settings) -> None:
    gemini = GeminiClient(settings, api_key="dummy-key")

    assert gemini.model == "gemini-2.5-flash"
    assert gemini._client is None
    assert "dummy-key" not in repr(gemini)


@pytest.mark.parametrize("api_key", ["", "   "])
def test_empty_api_key_is_rejected(settings: Settings, api_key: str) -> None:
    with pytest.raises(ValueError, match="API key"):
        GeminiClient(settings, api_key=api_key)


async def test_generate_maps_request_and_response(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
) -> None:
    fake_sdk.models.generate_responses = [
        make_response(
            text="Hello!", usage=make_usage(prompt=12, candidates=4, cached=7)
        )
    ]
    tools = [
        ToolDefinition(
            name="read_file",
            description="Read a file.",
            parameters_json_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        )
    ]

    response = await client.generate(
        [Message(role="user", content="Hi")],
        tools=tools,
        system_instruction="Be helpful.",
    )

    call = fake_sdk.models.generate_calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["contents"][0].role == "user"
    assert call["contents"][0].parts[0].text == "Hi"
    assert call["config"].system_instruction == "Be helpful."
    assert call["config"].max_output_tokens == 8192
    declaration = call["config"].tools[0].function_declarations[0]
    assert declaration.name == "read_file"
    assert declaration.parameters_json_schema["type"] == "object"

    assert response.content == "Hello!"
    assert response.tool_calls == []
    assert response.model == "gemini-2.5-flash"
    assert response.finish_reason == "STOP"
    assert response.usage == UsageInfo(
        input_tokens=12,
        output_tokens=4,
        cached_tokens=7,
        total_tokens=16,
    )


async def test_generate_maps_tool_history(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
) -> None:
    fake_sdk.models.generate_responses = [make_response(text="done")]
    call = ToolCallRequest(id="call-1", name="read_file", arguments={"path": "a.txt"})
    messages = [
        Message(role="user", content="Read a.txt"),
        Message(role="assistant", content="", tool_calls=(call,)),
        Message(
            role="tool",
            content="file body",
            tool_call_id="call-1",
            name="read_file",
        ),
    ]

    await client.generate(messages)

    contents = fake_sdk.models.generate_calls[0]["contents"]
    assert [content.role for content in contents] == ["user", "model", "user"]

    function_call = contents[1].parts[0].function_call
    assert function_call.id == "call-1"
    assert function_call.name == "read_file"
    assert function_call.args == {"path": "a.txt"}

    function_response = contents[2].parts[0].function_response
    assert function_response.id == "call-1"
    assert function_response.name == "read_file"
    assert function_response.response == {"result": "file body"}


async def test_tool_message_requires_a_tool_name(client: GeminiClient) -> None:
    with pytest.raises(ValueError, match="name"):
        await client.generate(
            [Message(role="tool", content="body", tool_call_id="call-1")]
        )


async def test_generate_maps_tool_calls_in_response(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
) -> None:
    fake_sdk.models.generate_responses = [
        make_response(
            function_calls=[
                make_function_call("read_file", {"path": "a.txt"}, call_id="call-7"),
                make_function_call("grep", {"query": "needle"}),
            ],
            usage=make_usage(),
        )
    ]

    response = await client.generate([Message(role="user", content="Read it")])

    assert response.content is None
    assert response.tool_calls == [
        ToolCallRequest(id="call-7", name="read_file", arguments={"path": "a.txt"}),
        ToolCallRequest(id="call_1", name="grep", arguments={"query": "needle"}),
    ]


async def test_generate_retries_transient_errors(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
) -> None:
    fake_sdk.models.generate_errors = [FakeApiError(500), FakeApiError(503)]
    fake_sdk.models.generate_responses = [make_response(text="ok")]

    response = await client.generate([Message(role="user", content="Hi")])

    assert response.content == "ok"
    assert len(fake_sdk.models.generate_calls) == 3


async def test_generate_stops_after_max_attempts(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
) -> None:
    fake_sdk.models.generate_errors = [FakeApiError(503) for _ in range(3)]

    with pytest.raises(LLMError) as excinfo:
        await client.generate([Message(role="user", content="Hi")])

    assert len(fake_sdk.models.generate_calls) == 3
    assert excinfo.value.status == 503
    assert "503" in str(excinfo.value)


async def test_fallback_model_is_used_after_server_errors(
    settings: Settings,
    fake_sdk: FakeSdkClient,
) -> None:
    pro_settings = Settings(_env_file=None, model="gemini-2.5-pro")
    gemini = GeminiClient(pro_settings, api_key=TEST_KEY)
    fake_sdk.models.generate_errors = [FakeApiError(500) for _ in range(3)]
    fake_sdk.models.generate_responses = [make_response(text="from fallback")]

    response = await gemini.generate([Message(role="user", content="Hi")])

    models_called = [call["model"] for call in fake_sdk.models.generate_calls]
    assert models_called == [
        "gemini-2.5-pro",
        "gemini-2.5-pro",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]
    assert response.model == "gemini-2.5-flash"
    assert response.content == "from fallback"
    assert settings.model == "gemini-2.5-flash"


async def test_client_errors_are_not_retried_or_fallbacked(
    settings: Settings,
    fake_sdk: FakeSdkClient,
) -> None:
    pro_settings = Settings(_env_file=None, model="gemini-2.5-pro")
    gemini = GeminiClient(pro_settings, api_key=TEST_KEY)
    fake_sdk.models.generate_errors = [FakeApiError(400)]

    with pytest.raises(LLMError) as excinfo:
        await gemini.generate([Message(role="user", content="Hi")])

    assert excinfo.value.status == 400
    assert len(fake_sdk.models.generate_calls) == 1


async def test_flash_has_no_fallback(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
) -> None:
    fake_sdk.models.generate_errors = [FakeApiError(503) for _ in range(3)]

    with pytest.raises(LLMError):
        await client.generate([Message(role="user", content="Hi")])

    assert len(fake_sdk.models.generate_calls) == 3


async def test_zero_retry_attempts_means_a_single_attempt(
    settings: Settings,
    fake_sdk: FakeSdkClient,
) -> None:
    no_retry_settings = Settings(_env_file=None, retry_attempts=0)
    gemini = GeminiClient(no_retry_settings, api_key=TEST_KEY)
    fake_sdk.models.generate_errors = [FakeApiError(500)]

    with pytest.raises(LLMError):
        await gemini.generate([Message(role="user", content="Hi")])

    assert len(fake_sdk.models.generate_calls) == 1


async def test_timeout_is_not_retried(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
) -> None:
    fake_sdk.models.generate_errors = [TimeoutError("deadline exceeded")]

    with pytest.raises(LLMError) as excinfo:
        await client.generate([Message(role="user", content="Hi")])

    assert len(fake_sdk.models.generate_calls) == 1
    assert "deadline exceeded" in str(excinfo.value)


async def test_generate_stream_yields_text_then_done(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
) -> None:
    fake_sdk.models.chunks = [
        make_text_chunk("Hel"),
        make_text_chunk("lo", usage=make_usage(prompt=9, candidates=2)),
    ]

    events = [
        event
        async for event in client.generate_stream([Message(role="user", content="Hi")])
    ]

    assert [event.event_type for event in events] == [
        "text_delta",
        "text_delta",
        "usage",
        "done",
    ]
    assert "".join(str(event.data) for event in events[:2]) == "Hello"
    assert events[2].data.input_tokens == 9
    assert events[3].data == "gemini-2.5-flash"


async def test_generate_stream_marks_thought_parts(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
) -> None:
    fake_sdk.models.chunks = [make_text_chunk("thinking...", thought=True)]

    events = [
        event
        async for event in client.generate_stream([Message(role="user", content="Hi")])
    ]

    assert events[0].event_type == "text_delta"
    assert events[0].metadata == {"thought": True}


async def test_generate_stream_buffers_split_tool_calls(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
) -> None:
    first = make_function_call(
        "read_file", {"path": "a.txt"}, call_id="call-1", will_continue=True
    )
    second = make_function_call(
        "read_file",
        partial_args=[
            types.PartialArg(json_path="$.start_line", number_value=3.0),
            types.PartialArg(json_path="$.options.mode", string_value="r"),
        ],
        will_continue=False,
    )
    fake_sdk.models.chunks = [
        make_function_call_chunk(first),
        make_function_call_chunk(second, usage=make_usage()),
    ]

    events = [
        event
        async for event in client.generate_stream([Message(role="user", content="Hi")])
    ]

    tool_events = [event for event in events if event.event_type == "tool_call_delta"]
    assert len(tool_events) == 1
    assert tool_events[0].data == ToolCallRequest(
        id="call-1",
        name="read_file",
        arguments={
            "path": "a.txt",
            "start_line": 3.0,
            "options": {"mode": "r"},
        },
    )


async def test_generate_stream_flushes_unfinished_tool_call(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
) -> None:
    pending = make_function_call(
        "grep", {"query": "needle"}, call_id="call-2", will_continue=True
    )
    fake_sdk.models.chunks = [make_function_call_chunk(pending)]

    events = [
        event
        async for event in client.generate_stream([Message(role="user", content="Hi")])
    ]

    tool_events = [event for event in events if event.event_type == "tool_call_delta"]
    assert len(tool_events) == 1
    assert tool_events[0].data.arguments == {"query": "needle"}


async def test_generate_stream_reports_errors(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
) -> None:
    fake_sdk.models.chunks = [make_text_chunk("partial")]
    fake_sdk.models.stream_error = FakeApiError(503)

    events = [
        event
        async for event in client.generate_stream([Message(role="user", content="Hi")])
    ]

    assert [event.event_type for event in events] == [
        "text_delta",
        "error",
        "usage",
        "done",
    ]
    assert isinstance(events[1].data, LLMError)
    assert events[1].data.status == 503
    assert len(fake_sdk.models.stream_calls) == 1


async def test_close_releases_the_sdk_client(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
) -> None:
    assert client._sdk_client() is fake_sdk

    client.close()

    assert fake_sdk.closed is True
    assert client._client is None


async def test_telemetry_records_metadata_without_secrets(
    settings: Settings,
    fake_sdk: FakeSdkClient,
    telemetry_logs,
    caplog: pytest.LogCaptureFixture,
) -> None:
    api_key = "AIzaSyVerySecretKey1234567890"
    gemini = GeminiClient(settings, api_key=api_key)
    fake_sdk.models.generate_errors = [FakeApiError(400)]

    with caplog.at_level(logging.DEBUG), pytest.raises(LLMError):
        await gemini.generate([Message(role="user", content="PROMPT-CONTENT-UNIQUE")])

    records = (telemetry_logs / "telemetry.jsonl").read_text(encoding="utf-8")
    payload = json.loads(records.splitlines()[0])["event"]

    assert payload["model"] == "gemini-2.5-flash"
    assert payload["status"] == "error"
    assert payload["retry_count"] == 0
    assert api_key not in records
    assert "PROMPT-CONTENT-UNIQUE" not in records
    assert all(api_key not in record.getMessage() for record in caplog.records)


async def test_timeout_telemetry_status(
    client: GeminiClient,
    fake_sdk: FakeSdkClient,
    telemetry_logs,
) -> None:
    fake_sdk.models.generate_errors = [TimeoutError("slow")]

    with pytest.raises(LLMError):
        await client.generate([Message(role="user", content="Hi")])

    payload = json.loads(
        (telemetry_logs / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert payload["event"]["status"] == "timeout"


def test_public_annotations_do_not_leak_sdk_types() -> None:
    annotations: list[Any] = []
    for target in (
        GeminiClient.generate,
        GeminiClient.generate_stream,
        GeminiClient.__init__,
        inspect.signature(GeminiClient.generate),
    ):
        if isinstance(target, inspect.Signature):
            annotations.extend(target.parameters.values())
        else:
            annotations.extend(get_type_hints(target).values())

    for item in (Message, ModelResponse, StreamEvent, ToolCallRequest, UsageInfo):
        annotations.extend(get_type_hints(item).values())

    for annotation in annotations:
        rendered = str(getattr(annotation, "annotation", annotation))
        assert "google.genai" not in rendered


def test_accumulator_emits_complete_calls_only() -> None:
    accumulator = ToolCallAccumulator()
    partial = make_function_call(
        "grep", {"query": "a"}, call_id="c1", will_continue=True
    )

    assert accumulator.feed(partial) is None

    completed = accumulator.feed(make_function_call("grep", {"path": "src"}))
    assert completed == ToolCallRequest(
        id="c1", name="grep", arguments={"query": "a", "path": "src"}
    )


def test_partial_args_support_array_indexes() -> None:
    accumulator = ToolCallAccumulator()
    accumulator.feed(make_function_call("edit", {}, call_id="c2", will_continue=True))

    completed = accumulator.feed(
        make_function_call(
            "edit",
            partial_args=[
                types.PartialArg(json_path="$.edits[1].text", string_value="x")
            ],
        )
    )

    assert completed is not None
    assert completed.arguments == {"edits": [None, {"text": "x"}]}
