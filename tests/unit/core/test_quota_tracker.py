"""Tests for the local quota tracker."""

from __future__ import annotations

import json
import logging
import os
import stat
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gemini_cli.core import quota_tracker
from gemini_cli.core.quota_tracker import QuotaTracker, UsageRecord


class Clock:
    """Mutable clock so rolling windows are deterministic."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


@pytest.fixture
def clock() -> Clock:
    return Clock(datetime(2026, 1, 5, 12, 0, 0))


def tracker_for(
    path: Path,
    clock: Clock,
    *,
    five_hour_limit: int = 100,
    weekly_limit: int = 1_000,
) -> QuotaTracker:
    return QuotaTracker(
        path,
        five_hour_limit=five_hour_limit,
        weekly_limit=weekly_limit,
        now=clock,
    )


def write_state(path: Path, records: list[object]) -> None:
    path.write_text(
        json.dumps({"version": 1, "records": records}),
        encoding="utf-8",
    )


def test_usage_record_total() -> None:
    record = UsageRecord(
        timestamp=datetime(2026, 1, 5, 12),
        model="gemini-2.5-flash",
        input_tokens=10,
        output_tokens=5,
    )

    assert record.total_tokens == 15


def test_missing_file_starts_empty(tmp_path: Path, clock: Clock) -> None:
    tracker = tracker_for(tmp_path / "quota.json", clock)

    status = tracker.get_5h_status()

    assert status.used_tokens == 0
    assert status.limit_tokens == 100
    assert status.percentage == 0.0
    assert status.time_remaining == timedelta(0)
    assert tracker.get_weekly_status().used_tokens == 0


def test_records_usage_in_both_windows(tmp_path: Path, clock: Clock) -> None:
    tracker = tracker_for(tmp_path / "quota.json", clock)

    tracker.record_usage("gemini-2.5-flash", 10, 5)

    five_hour = tracker.get_5h_status()
    weekly = tracker.get_weekly_status()
    assert five_hour.used_tokens == 15
    assert five_hour.percentage == 15.0
    assert weekly.used_tokens == 15
    assert weekly.limit_tokens == 1_000
    assert weekly.percentage == 1.5


def test_accumulates_multiple_records(tmp_path: Path, clock: Clock) -> None:
    tracker = tracker_for(tmp_path / "quota.json", clock)

    tracker.record_usage("gemini-2.5-flash", 10, 5)
    tracker.record_usage("gemini-2.5-pro", 20, 10)

    assert tracker.get_5h_status().used_tokens == 45


def test_persists_and_reloads(tmp_path: Path, clock: Clock) -> None:
    path = tmp_path / "quota.json"
    first = tracker_for(path, clock)

    first.record_usage("gemini-2.5-flash", 10, 5)
    second = tracker_for(path, clock)

    assert path.exists()
    assert second.get_5h_status().used_tokens == 15


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")
def test_state_file_is_private(tmp_path: Path, clock: Clock) -> None:
    path = tmp_path / "quota.json"
    tracker = tracker_for(path, clock)

    tracker.record_usage("gemini-2.5-flash", 1, 1)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_five_hour_window_excludes_old_usage(
    tmp_path: Path,
    clock: Clock,
) -> None:
    tracker = tracker_for(tmp_path / "quota.json", clock)
    tracker.record_usage("gemini-2.5-flash", 10, 5)

    clock.advance(timedelta(hours=6))

    assert tracker.get_5h_status().used_tokens == 0
    assert tracker.get_weekly_status().used_tokens == 15


def test_weekly_window_excludes_old_usage(tmp_path: Path, clock: Clock) -> None:
    tracker = tracker_for(tmp_path / "quota.json", clock)
    tracker.record_usage("gemini-2.5-flash", 10, 5)

    clock.advance(timedelta(days=8))

    assert tracker.get_5h_status().used_tokens == 0
    assert tracker.get_weekly_status().used_tokens == 0


def test_time_remaining_counts_down_from_the_oldest_record(
    tmp_path: Path,
    clock: Clock,
) -> None:
    tracker = tracker_for(tmp_path / "quota.json", clock)
    tracker.record_usage("gemini-2.5-flash", 10, 5)

    assert tracker.get_5h_status().time_remaining == timedelta(hours=5)

    clock.advance(timedelta(hours=2))
    assert tracker.get_5h_status().time_remaining == timedelta(hours=3)


def test_empty_window_has_no_remaining_time(
    tmp_path: Path,
    clock: Clock,
) -> None:
    tracker = tracker_for(tmp_path / "quota.json", clock)

    assert tracker.get_weekly_status().time_remaining == timedelta(0)


def test_percentage_is_rounded(tmp_path: Path, clock: Clock) -> None:
    tracker = tracker_for(
        tmp_path / "quota.json",
        clock,
        five_hour_limit=300,
    )

    tracker.record_usage("gemini-2.5-flash", 60, 40)

    assert tracker.get_5h_status().percentage == 33.33


def test_percentage_can_exceed_one_hundred(
    tmp_path: Path,
    clock: Clock,
) -> None:
    tracker = tracker_for(
        tmp_path / "quota.json",
        clock,
        five_hour_limit=10,
    )

    tracker.record_usage("gemini-2.5-flash", 20, 0)

    assert tracker.get_5h_status().percentage == 200.0


def test_rejects_negative_usage(tmp_path: Path, clock: Clock) -> None:
    tracker = tracker_for(tmp_path / "quota.json", clock)

    with pytest.raises(ValueError, match="negative"):
        tracker.record_usage("gemini-2.5-flash", -1, 0)


@pytest.mark.parametrize(
    ("five_hour_limit", "weekly_limit", "match"),
    [
        (0, 1_000, "five_hour_limit"),
        (-1, 1_000, "five_hour_limit"),
        (100, 0, "weekly_limit"),
    ],
)
def test_rejects_non_positive_limits(
    tmp_path: Path,
    clock: Clock,
    five_hour_limit: int,
    weekly_limit: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        tracker_for(
            tmp_path / "quota.json",
            clock,
            five_hour_limit=five_hour_limit,
            weekly_limit=weekly_limit,
        )


def test_corrupt_file_is_ignored(
    tmp_path: Path,
    clock: Clock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "quota.json"
    path.write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        tracker = tracker_for(path, clock)

    assert tracker.get_5h_status().used_tokens == 0
    assert "unreadable quota file" in caplog.text


def test_invalid_records_are_skipped(tmp_path: Path, clock: Clock) -> None:
    path = tmp_path / "quota.json"
    write_state(
        path,
        [
            {"timestamp": "not-a-date", "model": "x", "input_tokens": 1},
            {
                "timestamp": clock.now.isoformat(),
                "model": "gemini-2.5-flash",
                "input_tokens": 4,
                "output_tokens": 6,
            },
            "not-a-record",
        ],
    )

    tracker = tracker_for(path, clock)

    assert tracker.get_5h_status().used_tokens == 10


def test_timezone_aware_timestamps_are_normalized(
    tmp_path: Path,
    clock: Clock,
) -> None:
    path = tmp_path / "quota.json"
    aware = clock.now.astimezone()
    write_state(
        path,
        [
            {
                "timestamp": aware.isoformat(),
                "model": "gemini-2.5-flash",
                "input_tokens": 3,
                "output_tokens": 2,
            }
        ],
    )

    tracker = tracker_for(path, clock)

    assert tracker.get_5h_status().used_tokens == 5


def test_prune_drops_records_outside_the_weekly_window(
    tmp_path: Path,
    clock: Clock,
) -> None:
    path = tmp_path / "quota.json"
    tracker = tracker_for(path, clock)
    tracker.record_usage("gemini-2.5-flash", 100, 0)

    clock.advance(timedelta(days=8))
    tracker.record_usage("gemini-2.5-flash", 3, 2)

    reloaded = tracker_for(path, clock)

    assert reloaded.get_weekly_status().used_tokens == 5


def test_record_count_is_capped(
    tmp_path: Path,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quota_tracker, "MAX_RECORDS", 3)
    path = tmp_path / "quota.json"
    tracker = tracker_for(path, clock)

    for _ in range(5):
        tracker.record_usage("gemini-2.5-flash", 1, 0)

    reloaded = tracker_for(path, clock)

    assert reloaded.get_5h_status().used_tokens == 3


def test_persistence_failure_does_not_raise(
    tmp_path: Path,
    clock: Clock,
) -> None:
    path = tmp_path / "quota.json"
    path.mkdir()
    tracker = tracker_for(path, clock)

    tracker.record_usage("gemini-2.5-flash", 10, 5)

    assert tracker.get_5h_status().used_tokens == 15


def test_works_in_memory_without_a_home_directory(
    tmp_path: Path,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quota_tracker, "_default_storage_path", lambda: None)
    tracker = QuotaTracker(
        five_hour_limit=100,
        weekly_limit=1_000,
        now=clock,
    )

    tracker.record_usage("gemini-2.5-flash", 10, 5)

    assert tracker.storage_path is None
    assert tracker.get_5h_status().used_tokens == 15
