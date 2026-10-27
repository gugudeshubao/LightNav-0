"""Read-only continuity accounting; never renews a path or changes control."""
import math
import statistics


def continuity_summary(observations, max_age_s=0.5):
    if not math.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError("max_age_s must be finite and positive")
    previous = None
    gaps = []
    intervals = []
    missing = 0
    for row in observations:
        if not row.get("accepted"):
            continue
        captured, consumed = row.get("captured_at_s"), row.get("consumed_at_s")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                   and math.isfinite(x) for x in (captured, consumed)):
            missing += 1
            previous = None
            continue
        if consumed < captured:
            raise ValueError("reply consumed before capture")
        if previous is not None:
            if captured < previous:
                raise ValueError("accepted captures out of order")
            intervals.append((captured - previous) * 1000)
            gaps.append((consumed - previous - max_age_s) * 1000)
        previous = captured
    return {
        "scope": "reply adoption only; excludes subsequent control solve/scheduling delay",
        "pairs": len(gaps), "missing_clock_rows": missing,
        "uncovered_pairs": sum(gap > 1e-6 for gap in gaps),
        "median_signed_gap_ms": statistics.median(gaps) if gaps else None,
        "max_signed_gap_ms": max(gaps) if gaps else None,
        "total_uncovered_ms": sum(max(0, gap) for gap in gaps),
        "median_capture_interval_ms": statistics.median(intervals) if intervals else None,
    }
