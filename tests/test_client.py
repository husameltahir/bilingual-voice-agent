from __future__ import annotations

import httpx
import pytest
import respx

from voice_agent.client import BASE_URL, ElevenLabsClient, ElevenLabsError, NotFoundError

from .conftest import TEST_KEY


def make(**kw) -> ElevenLabsClient:
    sleeps: list[float] = []
    client = ElevenLabsClient(api_key=TEST_KEY, sleep=sleeps.append, **kw)
    client.sleeps = sleeps  # type: ignore[attr-defined]
    return client


@respx.mock(base_url=BASE_URL)
def test_sends_api_key_header(respx_mock) -> None:
    route = respx_mock.post("/v1/convai/agents/create").respond(200, json={"agent_id": "a1"})
    assert make().create_agent({"name": "x"}) == {"agent_id": "a1"}
    assert route.calls.last.request.headers["xi-api-key"] == TEST_KEY


@respx.mock(base_url=BASE_URL)
def test_retries_429_then_succeeds(respx_mock) -> None:
    route = respx_mock.get("/v1/convai/test-invocations/inv1").mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "3"}, json={"detail": "slow down"}),
            httpx.Response(503),
            httpx.Response(200, json={"id": "inv1", "test_runs": []}),
        ]
    )
    client = make()
    assert client.get_test_invocation("inv1")["id"] == "inv1"
    assert route.call_count == 3
    assert client.sleeps == [3.0, 2.0]


@respx.mock(base_url=BASE_URL)
def test_gives_up_after_max_retries(respx_mock) -> None:
    respx_mock.post("/v1/convai/tools").respond(500, json={"detail": "boom"})
    client = make(max_retries=2)
    with pytest.raises(ElevenLabsError) as exc:
        client.create_tool({})
    assert "HTTP 500" in str(exc.value)
    assert len(client.sleeps) == 2


@respx.mock(base_url=BASE_URL)
def test_error_message_never_contains_key(respx_mock) -> None:
    respx_mock.post("/v1/convai/secrets").respond(
        422, json={"detail": f"bad request, key was {TEST_KEY}"}
    )
    with pytest.raises(ElevenLabsError) as exc:
        make().create_secret({"type": "new", "name": "n", "value": "v"})
    assert TEST_KEY not in str(exc.value)
    assert "HTTP 422" in str(exc.value)
    assert TEST_KEY not in repr(make())


@respx.mock(base_url=BASE_URL)
def test_401_hint_and_404_type(respx_mock) -> None:
    respx_mock.delete("/v1/convai/agents/a1").respond(404, json={"detail": "not found"})
    respx_mock.delete("/v1/convai/agents/a2").respond(401, json={"detail": "invalid key"})
    with pytest.raises(NotFoundError):
        make().delete_agent("a1")
    with pytest.raises(ElevenLabsError, match="ELEVENLABS_API_KEY"):
        make().delete_agent("a2")


@respx.mock(base_url=BASE_URL)
def test_signed_url(respx_mock) -> None:
    route = respx_mock.get("/v1/convai/conversation/get-signed-url").respond(
        200, json={"signed_url": "wss://example.com/x"}
    )
    assert make().get_signed_url("a1") == "wss://example.com/x"
    assert route.calls.last.request.url.params["agent_id"] == "a1"


@respx.mock(base_url=BASE_URL)
def test_run_tests_body_and_update_test_is_put(respx_mock) -> None:
    run = respx_mock.post("/v1/convai/agents/a1/run-tests").respond(200, json={"id": "inv"})
    put = respx_mock.put("/v1/convai/agent-testing/t1").respond(200, json={"id": "t1"})
    client = make()
    client.run_tests("a1", ["t1", "t2"])
    client.update_test("t1", {"name": "x"})
    import json

    assert json.loads(run.calls.last.request.content) == {
        "tests": [{"test_id": "t1"}, {"test_id": "t2"}]
    }
    assert put.called


def test_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(ElevenLabsError, match="ELEVENLABS_API_KEY is not set"):
        ElevenLabsClient()
