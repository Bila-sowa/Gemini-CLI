"""Tests for bounded retry behavior on transient API errors."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from gemini_cli.llm import retry as retry_module
from gemini_cli.llm.retry import (
    RetryPolicy,
    retry_after_seconds,
    status_code,
    with_retry,
)


class FakeApiError(Exception):
    """Minimal stand-in for an HTTP error returned by a client library."""

    def __init__(self, code: int, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(f"HTTP {code}")
        self.status_code = code
        self.response = SimpleNamespace(status_code=code, headers=dict(headers or {}))


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the module's ``asyncio`` reference with a recording stub."""
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(retry_module, "asyncio", SimpleNamespace(sleep=fake_sleep))
    return recorded


async def test_retries_until_success(sleeps: list[float]) -> None:
    attempts = 0

    async def func() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise FakeApiError(500)
        return "ok"

    result = await with_retry(func, RetryPolicy(jitter=False))

    assert result == "ok"
    assert attempts == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize("code", [400, 401, 403, 404])
async def test_client_errors_are_not_retried(code: int, sleeps: list[float]) -> None:
    attempts = 0

    async def func() -> None:
        nonlocal attempts
        attempts += 1
        raise FakeApiError(code)

    with pytest.raises(FakeApiError):
        await with_retry(func, RetryPolicy(jitter=False))

    assert attempts == 1
    assert sleeps == []


@pytest.mark.parametrize("code", [429, 500, 503])
async def test_transient_statuses_are_retried(code: int, sleeps: list[float]) -> None:
    attempts = 0

    async def func() -> None:
        nonlocal attempts
        attempts += 1
        raise FakeApiError(code)

    with pytest.raises(FakeApiError):
        await with_retry(func, RetryPolicy(max_attempts=2, jitter=False))

    assert attempts == 2
    assert sleeps == [1.0]


async def test_exhausts_all_attempts_and_reraises(sleeps: list[float]) -> None:
    attempts = 0

    async def func() -> None:
        nonlocal attempts
        attempts += 1
        raise FakeApiError(503)

    with pytest.raises(FakeApiError, match="HTTP 503"):
        await with_retry(func, RetryPolicy(max_attempts=3, jitter=False))

    assert attempts == 3
    assert sleeps == [1.0, 2.0]


async def test_retry_after_header_is_respected(sleeps: list[float]) -> None:
    attempts = 0

    async def func() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FakeApiError(429, {"Retry-After": "7"})
        return "ok"

    await with_retry(func, RetryPolicy(jitter=False))

    assert sleeps == [7.0]


async def test_retry_after_is_capped_by_max_delay(sleeps: list[float]) -> None:
    attempts = 0

    async def func() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FakeApiError(429, {"Retry-After": "120"})
        return "ok"

    await with_retry(func, RetryPolicy(max_delay_seconds=30.0, jitter=False))

    assert sleeps == [30.0]


async def test_retry_after_accepts_http_date(sleeps: list[float]) -> None:
    retry_at = datetime.now(UTC) + timedelta(seconds=5)
    attempts = 0

    async def func() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FakeApiError(
                503, {"Retry-After": format_datetime(retry_at, usegmt=True)}
            )
        return "ok"

    await with_retry(func, RetryPolicy(jitter=False))

    assert len(sleeps) == 1
    assert 3.0 <= sleeps[0] <= 6.0


async def test_backoff_is_exponential_and_capped(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
) -> None:
    delays = [0.5, 1.5]
    called_with: list[tuple[float, float]] = []

    def fake_uniform(low: float, high: float) -> float:
        called_with.append((low, high))
        return delays[len(called_with) - 1]

    monkeypatch.setattr(retry_module.random, "uniform", fake_uniform)

    async def func() -> None:
        raise FakeApiError(500)

    with pytest.raises(FakeApiError):
        await with_retry(
            func,
            RetryPolicy(max_attempts=3, base_delay_seconds=1.0, max_delay_seconds=4.0),
        )

    assert called_with == [(0.0, 1.0), (0.0, 2.0)]
    assert sleeps == [0.5, 1.5]


async def test_non_http_errors_are_not_retried(sleeps: list[float]) -> None:
    async def func() -> None:
        raise ValueError("bad arguments")

    with pytest.raises(ValueError, match="bad arguments"):
        await with_retry(func, RetryPolicy())

    assert sleeps == []


async def test_max_attempts_must_be_positive() -> None:
    async def func() -> None:  # pragma: no cover - never called
        return None

    with pytest.raises(ValueError, match="max_attempts"):
        await with_retry(func, RetryPolicy(max_attempts=0))


def test_status_code_from_gemini_api_error() -> None:
    error = genai_errors.APIError(429, {"error": {"message": "quota"}})

    assert status_code(error) == 429


def test_status_code_from_httpx_error() -> None:
    request = httpx.Request("GET", "https://example.invalid")
    response = httpx.Response(503, request=request, headers={"Retry-After": "3"})
    error = httpx.HTTPStatusError("unavailable", request=request, response=response)

    assert status_code(error) == 503
    assert retry_after_seconds(error) == 3.0


def test_retry_after_absent_returns_none() -> None:
    assert retry_after_seconds(FakeApiError(500)) is None
    assert retry_after_seconds(ValueError("plain")) is None


def test_retry_after_handles_invalid_value() -> None:
    assert retry_after_seconds(FakeApiError(429, {"Retry-After": "soon"})) is None
