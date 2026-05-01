"""
news_manager.py
---------------
FTMO-safe macro calendar blackout manager.

Key design principles
=====================
1. **UTC-only** — all comparisons use timezone-aware UTC datetimes.
   dateutil.parser handles the Forex Factory EST/EDT offset automatically.

2. **Fail-closed** — if the calendar cannot be fetched OR parsed, every
   method treats the current moment as a danger period.  Better to miss
   a trade than to enter into NFP/CPI slippage.

3. **Local cache** — FF JSON is downloaded once and cached to
   ``~/.cache/ff_calendar.json``.  The cache is refreshed if it is older
   than CACHE_MAX_AGE_HOURS.  This survives short network outages and
   avoids hammering the FF server on every poll.

4. **Drop-in compatible** — ``is_in_blackout()`` keeps the same signature
   as the old stub so no callers need to change.

Usage
=====
    from news_manager import NewsManager

    # Pre-flight check in hot_window_executor
    if NewsManager.is_in_blackout():
        return HotDecision(fill=False, reason="news-block")

    # Richer check with event names
    events = NewsManager.get_danger_events(lookahead_mins=30, lookback_mins=15)
    if events:
        names = [e.get("title", "?") for e in events]
        logger.warning(f"Danger events: {names}")
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from dateutil import parser as dateutil_parser

logger = logging.getLogger("FTMO_Trader")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE_PATH = Path.home() / ".cache" / "ff_calendar.json"
CACHE_MAX_AGE_HOURS = 6          # refresh cache if older than this
REQUEST_TIMEOUT_SEC = 5

# Currencies that matter for GBPUSD / XAUUSD
WATCHED_CURRENCIES = {"GBP", "USD"}

# Only block on High-impact events (change to include "Medium" if desired)
WATCHED_IMPACTS = {"High"}

# Default danger window: block 30 min before and 15 min after the event
DEFAULT_LOOKAHEAD_MINS = 30
DEFAULT_LOOKBACK_MINS = 15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_cache() -> list[dict[str, Any]] | None:
    """Return cached events list, or None if cache is missing / stale."""
    if not CACHE_PATH.exists():
        return None
    age_hours = (time.time() - CACHE_PATH.stat().st_mtime) / 3600
    if age_hours > CACHE_MAX_AGE_HOURS:
        logger.info(f"[NewsManager] Cache stale ({age_hours:.1f}h old) — will refresh")
        return None
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.debug(f"[NewsManager] Loaded {len(data)} events from cache")
        return data
    except Exception as exc:
        logger.warning(f"[NewsManager] Cache read failed: {exc}")
        return None


def _save_cache(events: list[dict[str, Any]]) -> None:
    """Persist events list to local cache file."""
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as fh:
            json.dump(events, fh)
        logger.info(f"[NewsManager] Saved {len(events)} events to cache")
    except Exception as exc:
        logger.warning(f"[NewsManager] Cache write failed: {exc}")


def _fetch_remote() -> list[dict[str, Any]] | None:
    """
    Download this week's FF calendar JSON.
    Returns the parsed list on success, None on any failure.
    """
    try:
        resp = requests.get(FF_CALENDAR_URL, timeout=REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        events = resp.json()
        logger.info(f"[NewsManager] Fetched {len(events)} events from FF")
        return events
    except requests.exceptions.RequestException as exc:
        logger.error(f"[NewsManager] Remote fetch failed: {exc}")
        return None
    except Exception as exc:
        logger.error(f"[NewsManager] Unexpected error during fetch: {exc}")
        return None


def _get_events() -> list[dict[str, Any]] | None:
    """
    Return the event list from cache if fresh, otherwise fetch and cache.
    Returns None ONLY when both cache and remote fail — callers must treat
    None as a danger signal (fail-closed).
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    remote = _fetch_remote()
    if remote is not None:
        _save_cache(remote)
        return remote

    # Both cache and remote failed
    logger.error("[NewsManager] FAIL-CLOSED: no calendar data available")
    return None


def _parse_event_time(date_str: str) -> datetime | None:
    """
    Parse an FF date string into a UTC-aware datetime.
    dateutil.parser handles EST, EDT, and ISO-8601 offsets automatically.
    Returns None if parsing fails.
    """
    try:
        dt = dateutil_parser.parse(date_str)
        if dt.tzinfo is None:
            # FF occasionally serves naive strings — assume US Eastern (UTC-4 EDT)
            logger.debug(f"[NewsManager] Naive datetime '{date_str}' — assuming UTC-4")
            dt = dt.replace(tzinfo=timezone(timedelta(hours=-4)))
        return dt.astimezone(timezone.utc)
    except Exception as exc:
        logger.warning(f"[NewsManager] Could not parse event time '{date_str}': {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NewsManager:
    """
    FTMO-compliant macro news blackout manager.

    All time comparisons are UTC-aware.  Fails closed: if the calendar
    cannot be loaded, every query returns True (danger / blocked).
    """

    # Manual override windows added via add_blackout()
    # Each entry: {"start": datetime (UTC-aware), "end": datetime (UTC-aware)}
    _manual_windows: list[dict[str, datetime]] = []

    # ------------------------------------------------------------------
    # Primary check — drop-in replacement for the old stub
    # ------------------------------------------------------------------

    @classmethod
    def is_in_blackout(
        cls,
        lookahead_mins: int = DEFAULT_LOOKAHEAD_MINS,
        lookback_mins: int = DEFAULT_LOOKBACK_MINS,
    ) -> bool:
        """
        Returns True if trading should be blocked right now.

        Blocks when:
        - A High-impact GBP or USD event is within the danger window, OR
        - Calendar data is unavailable (fail-closed), OR
        - Current time falls inside a manually added blackout window.
        """
        now_utc = datetime.now(timezone.utc)

        # 1. Manual override windows
        for w in cls._manual_windows:
            if w["start"] <= now_utc <= w["end"]:
                logger.warning(
                    f"[NewsManager] Manual blackout active: "
                    f"{w['start'].isoformat()} → {w['end'].isoformat()}"
                )
                return True

        # 2. Live calendar
        danger = cls.get_danger_events(
            lookahead_mins=lookahead_mins,
            lookback_mins=lookback_mins,
        )

        # get_danger_events returns a non-empty sentinel list on failure
        if danger:
            names = [e.get("title", "UNKNOWN") for e in danger]
            logger.warning(f"[NewsManager] BLACKOUT — danger events: {names}")
            return True

        return False

    # ------------------------------------------------------------------
    # Richer query — used by hot_window_executor pre-flight
    # ------------------------------------------------------------------

    @classmethod
    def get_danger_events(
        cls,
        lookahead_mins: int = DEFAULT_LOOKAHEAD_MINS,
        lookback_mins: int = DEFAULT_LOOKBACK_MINS,
    ) -> list[dict[str, Any]]:
        """
        Return a list of High-impact events near the current UTC time.

        Returns a non-empty sentinel list ``[{"title": "CALENDAR_UNAVAILABLE"}]``
        when calendar data cannot be loaded — callers should treat any
        non-empty return as a danger signal.

        Parameters
        ----------
        lookahead_mins : int
            Block trades this many minutes BEFORE an event fires.
        lookback_mins : int
            Block trades this many minutes AFTER an event fires.
        """
        events = _get_events()
        if events is None:
            # Fail-closed sentinel
            return [{"title": "CALENDAR_UNAVAILABLE", "impact": "High", "currency": "?"}]

        now_utc = datetime.now(timezone.utc)
        danger: list[dict[str, Any]] = []

        for ev in events:
            # Filter by currency and impact
            if ev.get("currency") not in WATCHED_CURRENCIES:
                continue
            if ev.get("impact") not in WATCHED_IMPACTS:
                continue

            # Parse event time
            date_str = ev.get("date", "")
            event_utc = _parse_event_time(date_str)
            if event_utc is None:
                continue

            # Check danger window
            delta_mins = (event_utc - now_utc).total_seconds() / 60.0
            if -lookback_mins <= delta_mins <= lookahead_mins:
                danger.append({**ev, "_delta_mins": round(delta_mins, 1)})

        return danger

    # ------------------------------------------------------------------
    # Manual override — kept from old stub
    # ------------------------------------------------------------------

    @classmethod
    def add_blackout(cls, start_str: str, end_str: str) -> None:
        """
        Manually add a blackout window (e.g. FTMO maintenance, public holiday).

        Parameters
        ----------
        start_str / end_str : str
            ISO-8601 strings.  Timezone-naive strings are assumed UTC.
            Example: "2026-05-06 08:00", "2026-05-06 09:00"
        """
        try:
            start_dt = dateutil_parser.parse(start_str)
            end_dt = dateutil_parser.parse(end_str)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            cls._manual_windows.append({"start": start_dt, "end": end_dt})
            logger.info(
                f"[NewsManager] Manual blackout added: "
                f"{start_dt.isoformat()} → {end_dt.isoformat()}"
            )
        except Exception as exc:
            logger.error(f"[NewsManager] add_blackout failed: {exc}")

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    @staticmethod
    def refresh_cache() -> bool:
        """
        Force a fresh download of the FF calendar (ignores cache age).
        Returns True on success, False on failure.
        Call this once on Sunday evening as the weekly cron job.
        """
        events = _fetch_remote()
        if events is not None:
            _save_cache(events)
            return True
        logger.error("[NewsManager] Manual refresh failed")
        return False

    @staticmethod
    def cache_status() -> dict[str, Any]:
        """Return a dict with cache path, age in hours, and event count."""
        if not CACHE_PATH.exists():
            return {"exists": False, "path": str(CACHE_PATH)}
        age_hours = (time.time() - CACHE_PATH.stat().st_mtime) / 3600
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as fh:
                count = len(json.load(fh))
        except Exception:
            count = -1
        return {
            "exists": True,
            "path": str(CACHE_PATH),
            "age_hours": round(age_hours, 2),
            "event_count": count,
            "stale": age_hours > CACHE_MAX_AGE_HOURS,
        }
