"""Bounded exponential backoff for transient Gemini API errors."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 503})


@dataclass(frozen=True)
class RetryPolicy:
    """How many times and how patiently to retry a transient failure."""

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    jitter: bool = True
    retryable_status_codes: frozenset[int] = DEFAULT_RETRYABLE_STATUS_CODES


def status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from common HTTP error shapes.

    Supports ``google.genai.errors.APIError`` (``.code``),
    ``httpx.HTTPStatusError`` (``.response.status_code``), and objects exposing
    ``status_code`` or ``status``. Returns ``None`` when no code is present.
    """
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def retry_after_seconds(exc: BaseException) -> float | None:
    """Return the delay requested by the server via ``Retry-After``, if any."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    value = headers.get("Retry-After")
    if value is None:
        return None

    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass

    try:
        retry_at = parsedate_to_datetime(str(value))
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def _is_retryable(exc: BaseException, codes: frozenset[int]) -> bool:
    code = status_code(exc)
    return code is not None and code in codes


def _backoff_delay(attempt: int, policy: RetryPolicy) -> float:
    """Exponential delay for a 1-based *attempt*, capped and optionally jittered."""
    delay = min(
        policy.base_delay_seconds * (2.0 ** (attempt - 1)), policy.max_delay_seconds
    )
    if policy.jitter:
        return random.uniform(0.0, delay)
    return delay


async def with_retry(
    func: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
) -> T:
    """Call *func* until it succeeds or a non-retryable error appears.

    Retries only HTTP statuses listed in ``policy.retryable_status_codes``
    (for example 429, 500, 503). Client errors such as 400, 401, 403, and 404
    are raised immediately. ``Retry-After`` headers are honored when present
    and capped by ``policy.max_delay_seconds``.

    Raises:
        ValueError: if ``policy.max_attempts`` is less than 1.
        BaseException: the last error raised by *func* after the final attempt.
    """
    if policy.max_attempts < 1:
        msg = "RetryPolicy.max_attempts must be at least 1."
        raise ValueError(msg)

    last_error: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await func()
        except Exception as exc:
            if not _is_retryable(exc, policy.retryable_status_codes):
                raise

            last_error = exc
            if attempt >= policy.max_attempts:
                break

            delay = _backoff_delay(attempt, policy)
            retry_after = retry_after_seconds(exc)
            if retry_after is not None:
                delay = max(delay, retry_after)
            delay = min(delay, policy.max_delay_seconds)

            logger.warning(
                "Retryable API error (status %s) on attempt %d/%d; "
                "retrying %s in %.2fs",
                status_code(exc),
                attempt,
                policy.max_attempts,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)

    if last_error is None:  # pragma: no cover - defensive
        msg = "Retry loop exited without a result or an error."
        raise RuntimeError(msg)
    raise last_error
