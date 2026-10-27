import pytest

from vln_mujoco.timing import continuity_summary
from vln_mujoco.vln_client import VlnClient


def row(capture, consume, accepted=True):
    return {"captured_at_s": capture, "consumed_at_s": consume, "accepted": accepted}


def test_timely_replies_can_still_have_uncovered_handoff():
    result = continuity_summary([row(10, 10.3), row(10.3, 10.63)])
    assert result["uncovered_pairs"] == 1
    assert result["total_uncovered_ms"] == pytest.approx(130)


def test_overlap_and_exact_deadline_are_not_gaps():
    result = continuity_summary([row(10, 10.2), row(10.2, 10.4), row(10.4, 10.7)])
    assert result["uncovered_pairs"] == 0


def test_rejected_reply_does_not_renew_previous_path():
    result = continuity_summary([row(10, 10.3), row(10.2, 10.8, False), row(11, 11.3)])
    assert result["pairs"] == 1
    assert result["total_uncovered_ms"] == pytest.approx(800)


def test_missing_clock_is_not_invented():
    result = continuity_summary([row(10, 10.2), row(None, 10.5), row(11, 11.2)])
    assert result["pairs"] == 0
    assert result["missing_clock_rows"] == 1


def test_bad_clock_order_rejected():
    with pytest.raises(ValueError, match="before capture"):
        continuity_summary([row(10, 9)])


def test_optional_model_timings_filtered_without_affecting_control():
    assert VlnClient._parse_timings({"vit_ms": 12, "bad": float("nan"), "bool": True,
                                    "negative": -1, "text": "3"}) == {"vit_ms": 12.0}
    assert VlnClient._parse_timings(None) == {}
    assert VlnClient._input_tokens({"input_tokens": 100}) == 100
    assert VlnClient._input_tokens({"input_tokens": True}) is None
