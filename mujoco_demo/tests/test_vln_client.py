import queue
import threading

import pytest

from vln_mujoco.vln_client import VlnClient, normalize_server_url


def test_normalize_server_url() -> None:
    assert normalize_server_url("localhost:8000/ws") == "ws://localhost:8000/ws"
    assert normalize_server_url("wss://example.com/vln") == "wss://example.com/vln"


@pytest.mark.parametrize("value", ["http://example.com", "ws://user@example.com", "ws://host:99999"])
def test_reject_invalid_server_url(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_server_url(value)


def test_parse_nested_waypoints() -> None:
    assert VlnClient._parse_waypoints([[0.2, 0.0, 0.1], {"actions": [0.4, 0.1, 0.2]}]) == (
        (0.2, 0.0, 0.1),
        (0.4, 0.1, 0.2),
    )


def admission_client():
    client = VlnClient.__new__(VlnClient)
    client._lock = threading.Lock()
    client._session = 1
    client._instruction = "go forward"
    client._connected = True
    client._pending_frame = None
    client._prefetch_frames = False
    client._inflight_session = None
    client._results = queue.Queue()
    return client


def test_no_prefetch_preserves_admitted_frame_and_waits_for_reply():
    client = admission_client()
    assert client.offer_frame(1, b"first")
    assert not client.offer_frame(2, b"replacement")
    assert client._take_frame(1)[:2] == (1, b"first")
    assert not client.offer_frame(3, b"during-inference")
    result = object()
    client._publish_result(1, result)
    assert client.take_results() == [result]
    assert client.offer_frame(4, b"fresh-after-reply")
    assert client._take_frame(1)[:2] == (4, b"fresh-after-reply")


def test_old_reply_cannot_publish_or_release_new_session():
    client = admission_client()
    client._session = 2
    client._inflight_session = 2
    client._publish_result(1, object())
    assert client.take_results() == []
    assert client._inflight_session == 2
    assert not client.offer_frame(5, b"still-busy")

