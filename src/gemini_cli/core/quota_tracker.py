"""Client-side token and quota accounting.

The Gemini API does not expose a real-time "tokens remaining" endpoint, so the
tracker records the usage the API reports for every response and evaluates it
against rolling windows. Limits are local policy, not a provider guarantee:
they are configurable and only drive the status display and the ``/quota``
panel.

State is best-effort. A missing, unreadable, or unwritable quota file never
fails a model request; the tracker falls back to in-memory accounting.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from gemini_cli.config.defaults import (
    DEFAULT_QUOTA_5H_TOKEN_LIMIT,
    DEFAULT_QUOTA_WEEKLY_TOKEN_LIMIT,
    GLOBAL_CONFIG_DIR_NAME,
)

logger = logging.getLogger(__name__)

FIVE_HOUR_WINDOW = timedelta(hours=5)
WEEKLY_WINDOW = timedelta(days=7)
QUOTA_FILE_NAME = "quota.json"
STATE_VERSION = 1
MAX_RECORDS = 10_000


@dataclass(frozen=True)
class QuotaWindowStatus:
    """Consumption inside one rolling quota window."""

    used_tokens: int
    limit_tokens: int
    percentage: float
    time_remaining: timedelta


@dataclass(frozen=True)
class UsageRecord:
    """One response's token usage, kept for window calculations."""

    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Tokens billed for this response."""
        return self.input_tokens + self.output_tokens


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def _record_from(raw: object) -> UsageRecord | None:
    if not isinstance(raw, dict):
        return None
    timestamp = _parse_timestamp(raw.get("timestamp"))
    model = raw.get("model")
    input_tokens = raw.get("input_tokens")
    output_tokens = raw.get("output_tokens")
    if (
        timestamp is None
        or not isinstance(model, str)
        or not isinstance(input_tokens, int)
        or not isinstance(output_tokens, int)
        or input_tokens < 0
        or output_tokens < 0
    ):
        return None
    return UsageRecord(
        timestamp=timestamp,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _default_storage_path() -> Path | None:
    try:
        home = Path.home()
    except RuntimeError:
        logger.warning("Cannot determine home directory; quota stays in memory.")
        return None
    return home / GLOBAL_CONFIG_DIR_NAME / QUOTA_FILE_NAME


def _validate_limit(name: str, value: int) -> int:
    if value <= 0:
        msg = f"{name} must be positive."
        raise ValueError(msg)
    return value


class QuotaTracker:
    """Track token consumption across rolling five-hour and weekly windows."""

    def __init__(
        self,
        storage_path: Path | None = None,
        *,
        five_hour_limit: int = DEFAULT_QUOTA_5H_TOKEN_LIMIT,
        weekly_limit: int = DEFAULT_QUOTA_WEEKLY_TOKEN_LIMIT,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._storage_path = (
            storage_path if storage_path is not None else _default_storage_path()
        )
        self._five_hour_limit = _validate_limit("five_hour_limit", five_hour_limit)
        self._weekly_limit = _validate_limit("weekly_limit", weekly_limit)
        self._now = now if now is not None else datetime.now
        self._records = self._load()

    @property
    def storage_path(self) -> Path | None:
        """Where usage is persisted, or ``None`` for in-memory accounting."""
        return self._storage_path

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record one response's usage and persist the updated state."""
        if input_tokens < 0 or output_tokens < 0:
            msg = "Token counts must not be negative."
            raise ValueError(msg)

        now = self._now()
        self._records.append(
            UsageRecord(
                timestamp=now,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        self._prune(now)
        self._save()

    def get_5h_status(self) -> QuotaWindowStatus:
        """Return consumption inside the rolling five-hour window."""
        return self._status(FIVE_HOUR_WINDOW, self._five_hour_limit)

    def get_weekly_status(self) -> QuotaWindowStatus:
        """Return consumption inside the rolling seven-day window."""
        return self._status(WEEKLY_WINDOW, self._weekly_limit)

    def _status(
        self,
        window: timedelta,
        limit: int,
    ) -> QuotaWindowStatus:
        now = self._now()
        cutoff = now - window
        active = [record for record in self._records if record.timestamp > cutoff]
        used = sum(record.total_tokens for record in active)

        if active:
            oldest = min(record.timestamp for record in active)
            remaining = oldest + window - now
            if remaining < timedelta(0):
                remaining = timedelta(0)
        else:
            remaining = timedelta(0)

        return QuotaWindowStatus(
            used_tokens=used,
            limit_tokens=limit,
            percentage=round(used / limit * 100, 2),
            time_remaining=remaining,
        )

    def _prune(self, now: datetime) -> None:
        cutoff = now - WEEKLY_WINDOW
        self._records = [
            record for record in self._records if record.timestamp > cutoff
        ]
        if len(self._records) > MAX_RECORDS:
            self._records = self._records[-MAX_RECORDS:]

    def _load(self) -> list[UsageRecord]:
        path = self._storage_path
        if path is None or not path.exists():
            return []

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable quota file %s.", path)
            return []

        raw_records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(raw_records, list):
            logger.warning("Ignoring quota file with an invalid layout: %s.", path)
            return []

        records = [
            record for raw in raw_records if (record := _record_from(raw)) is not None
        ]
        records.sort(key=lambda record: record.timestamp)
        return records[-MAX_RECORDS:]

    def _save(self) -> None:
        path = self._storage_path
        if path is None:
            return

        payload = {
            "version": STATE_VERSION,
            "records": [
                {
                    "timestamp": record.timestamp.isoformat(),
                    "model": record.model,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                }
                for record in self._records
            ],
        }

        temporary: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, raw_name = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=".quota-",
                suffix=".tmp",
            )
            temporary = Path(raw_name)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temporary, path)
            path.chmod(0o600)
        except OSError:
            logger.warning("Could not persist quota usage to %s.", path)
            if temporary is not None:
                temporary.unlink(missing_ok=True)


__all__ = [
    "FIVE_HOUR_WINDOW",
    "MAX_RECORDS",
    "QUOTA_FILE_NAME",
    "STATE_VERSION",
    "WEEKLY_WINDOW",
    "QuotaTracker",
    "QuotaWindowStatus",
    "UsageRecord",
]
