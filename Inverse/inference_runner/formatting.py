"""Logging and formatting helpers for inference runner output."""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

import numpy as np

from .config import RecoveryAttempt


def _format_float_for_log(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3e}"
    except Exception:
        return str(value)


def _format_int_for_log(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return str(int(value))
    except Exception:
        return str(value)


def _format_plain_float_for_log(value: Any) -> str:
    if value is None:
        return f"{'-':>12}"
    try:
        v = float(value)
    except Exception:
        return f"{str(value):>12}"
    if not np.isfinite(v):
        return f"{str(v):>12}"
    if abs(v) < 5e-7:
        v = 0.0
    return f"{v:>12.6f}"


def _format_guess_for_log(value: Any) -> str:
    from .recovery import _guess_to_array

    arr, _ = _guess_to_array(value)
    if arr.size == 1:
        return _format_plain_float_for_log(arr[0])
    vals = ", ".join(_format_plain_float_for_log(v) for v in arr.tolist())
    return f"[{vals}]"


def _summarize_exception(exc: Optional[Exception]) -> str:
    """Build a compact exception summary without long file paths/trace text."""
    if exc is None:
        return "Unknown"
    msg = str(exc).strip()
    if not msg:
        return type(exc).__name__
    lines = [line.strip() for line in msg.splitlines() if line.strip()]
    if not lines:
        return type(exc).__name__
    preferred = None
    for line in lines:
        upper = line.upper()
        if "DIVERGED_" in upper or "FAILED TO CONVERGE" in upper or "CONVERGE" in upper:
            preferred = line
            break
    if preferred is None:
        preferred = lines[-1]
    preferred = re.sub(r"(?:[A-Za-z]:)?/(?:[^/\s:]+/)*[^/\s:]+", "<path>", preferred)
    preferred = re.sub(r"\s+", " ", preferred).strip()
    return f"{type(exc).__name__}: {preferred}"


def _format_recovery_summary(attempts: Sequence[RecoveryAttempt]) -> str:
    """Compact human-readable summary for failure messages."""
    if not attempts:
        return "no attempts logged"
    parts = []
    for a in attempts:
        short_reason = a.reason
        if len(short_reason) > 180:
            short_reason = short_reason[:177] + "..."
        parts.append(
            f"(attempt={a.attempt_index}, phase={a.phase}, status={a.status}, reason={short_reason})"
        )
    return "; ".join(parts)
