from graphify_search import utils
from graphify_search.errors import EndpointUnavailableError


def test_atomic_write_bytes_round_trips_and_replaces(tmp_path):
    target = tmp_path / "v.npy"
    utils.atomic_write_bytes(target, b"\x00\x01")
    utils.atomic_write_bytes(target, b"\x02")
    assert target.read_bytes() == b"\x02"
    assert [p.name for p in tmp_path.iterdir()] == ["v.npy"]


def test_endpoint_unavailable_carries_a_hint():
    err = EndpointUnavailableError("no answer from http://x", hint="start LM Studio")
    assert str(err) == "no answer from http://x"
    assert err.hint == "start LM Studio"
