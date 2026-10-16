"""Small shared timetable parsing helpers."""
from __future__ import annotations


def gtfs_time_seconds(value: str) -> int:
    """Convert GTFS ``HH:MM:SS`` to seconds without wrapping after 24 hours."""
    try:
        parts = str(value).split(":")
        if len(parts) != 3:
            raise ValueError
        hours, minutes, seconds = (int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid GTFS time {value!r}") from exc
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"invalid GTFS time {value!r}")
    return hours * 3600 + minutes * 60 + seconds
