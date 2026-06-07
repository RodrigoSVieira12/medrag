"""Cost-control rate limiting for the public demo (Option A).

Enforces the limits defined in `config.py`:
  - per-session question cap   (PUBLIC_MAX_QUESTIONS_PER_SESSION)
  - site-wide daily cap        (PUBLIC_MAX_QUESTIONS_PER_DAY_GLOBAL)
  - per-session cooldown       (PUBLIC_COOLDOWN_SECONDS)

State is in-memory: simple, dependency-free, and fine for a single-process
portfolio demo. It resets on restart and is NOT shared across multiple worker
processes — if you ever scale out, move this to Redis. None of this replaces
the monthly spend cap you set in the Anthropic Console (the real backstop).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import date

import config


@dataclass
class _SessionState:
    count: int = 0
    last_ts: float = 0.0


@dataclass
class LimitDecision:
    allowed: bool
    reason: str = ""
    remaining_session: int = 0


class RateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionState] = {}
        self._global_day: date = date.today()
        self._global_count: int = 0

    def _roll_day_if_needed(self) -> None:
        today = date.today()
        if today != self._global_day:
            self._global_day = today
            self._global_count = 0
            self._sessions.clear()  # fresh per-session budgets each day

    def check(self, session_id: str) -> LimitDecision:
        """Decide whether this session may ask another question right now.

        On success, it RESERVES the slot (increments counters), so call this
        exactly once per question attempt.
        """
        with self._lock:
            self._roll_day_if_needed()

            if self._global_count >= config.PUBLIC_MAX_QUESTIONS_PER_DAY_GLOBAL:
                return LimitDecision(
                    False,
                    "The site's daily question limit has been reached. "
                    "Please try again tomorrow.",
                )

            state = self._sessions.setdefault(session_id, _SessionState())

            if state.count >= config.PUBLIC_MAX_QUESTIONS_PER_SESSION:
                return LimitDecision(
                    False,
                    f"You've reached the demo limit of "
                    f"{config.PUBLIC_MAX_QUESTIONS_PER_SESSION} questions per session.",
                    remaining_session=0,
                )

            elapsed = time.monotonic() - state.last_ts
            if state.last_ts and elapsed < config.PUBLIC_COOLDOWN_SECONDS:
                wait = int(config.PUBLIC_COOLDOWN_SECONDS - elapsed) + 1
                return LimitDecision(
                    False,
                    f"Please wait {wait}s before asking again.",
                    remaining_session=config.PUBLIC_MAX_QUESTIONS_PER_SESSION - state.count,
                )

            # Reserve the slot.
            state.count += 1
            state.last_ts = time.monotonic()
            self._global_count += 1
            return LimitDecision(
                True,
                remaining_session=config.PUBLIC_MAX_QUESTIONS_PER_SESSION - state.count,
            )

    def remaining(self, session_id: str) -> int:
        with self._lock:
            self._roll_day_if_needed()
            state = self._sessions.get(session_id)
            used = state.count if state else 0
            return max(0, config.PUBLIC_MAX_QUESTIONS_PER_SESSION - used)
