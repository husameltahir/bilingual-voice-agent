from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import respx

from voice_agent.client import BASE_URL
from voice_agent.setup_agent import main

from .conftest import TOOL_SECRET

ENV = {"VOICE_TOOL_SECRET": TOOL_SECRET}


def _dry_run(brand: Path, state: Path, env=ENV) -> tuple[int, dict]:
    out = io.StringIO()
    code = main(["--brand", str(brand), "--state", str(state)], env=env, out=out)
    return code, json.loads(out.getvalue()) if out.getvalue() else {}


def _mock_create_routes(m: respx.MockRouter) -> dict[str, respx.Route]:
    kb_ids = iter(["doc_1", "doc_2", "doc_3"])
    return {
        "secret": m.post("/v1/convai/secrets").respond(
            200, json={"type": "stored", "secret_id": "sec_1", "name": "northwind_tool_secret"}
        ),
        "tool": m.post("/v1/convai/tools").respond(200, json={"id": "tool_1"}),
        "kb": m.post("/v1/convai/knowledge-base/url").mock(
            side_effect=lambda _req: httpx.Response(200, json={"id": next(kb_ids), "name": "x"})
        ),
        "agent": m.post("/v1/convai/agents/create").respond(200, json={"agent_id": "agent_1"}),
    }


def test_dry_run_prints_payloads_and_masks_secret(brand_dir: Path) -> None:
    code, result = _dry_run(brand_dir / "brand.json", brand_dir / "state.local.json")
    assert code == 0
    text = json.dumps(result)
    assert TOOL_SECRET not in text
    steps = result["steps"]
    assert [s["path"] for s in steps] == [
        "/v1/convai/secrets",
        "/v1/convai/tools",
        "/v1/convai/knowledge-base/url",
        "/v1/convai/knowledge-base/url",
        "/v1/convai/knowledge-base/url",
        "/v1/convai/agents/create",
    ]
    assert steps[0]["body"] == {
        "type": "new",
        "name": "northwind_tool_secret",
        "value": "Bearer ****",
    }
    headers = steps[1]["body"]["tool_config"]["api_schema"]["request_headers"]
    assert headers == {"Authorization": {"secret_id": "<new secret id: northwind_tool_secret>"}}
    agent = steps[-1]["body"]
    assert agent["conversation_config"]["agent"]["prompt"]["tool_ids"] == [
        "<new tool id: moving_quote>"
    ]
    assert not (brand_dir / "state.local.json").exists()


def test_dry_run_works_with_placeholders_and_no_secret(example_brand_path, tmp_path) -> None:
    code, result = _dry_run(example_brand_path, tmp_path / "s.json", env={})
    assert code == 0
    assert result["steps"][0]["body"]["value"] == "Bearer <value of VOICE_TOOL_SECRET>"


def test_apply_refuses_placeholder_voice(example_brand_path, tmp_path, client_factory) -> None:
    code = main(
        ["--brand", str(example_brand_path), "--state", str(tmp_path / "s.json"), "--apply"],
        env=ENV,
        client_factory=client_factory,
    )
    assert code == 2
    assert not (tmp_path / "s.json").exists()


def test_apply_refuses_missing_tool_secret(brand_dir: Path, client_factory) -> None:
    code = main(
        ["--brand", str(brand_dir / "brand.json"), "--state", str(brand_dir / "s.json"), "--apply"],
        env={},
        client_factory=client_factory,
    )
    assert code == 2


@respx.mock(base_url=BASE_URL, assert_all_called=False)
def test_apply_creates_then_updates(respx_mock, brand_dir: Path, client_factory) -> None:
    state_path = brand_dir / "state.local.json"
    args = ["--brand", str(brand_dir / "brand.json"), "--state", str(state_path), "--apply"]
    created = _mock_create_routes(respx_mock)

    assert main(args, env=ENV, client_factory=client_factory) == 0
    state = json.loads(state_path.read_text())
    assert state["agent_id"] == "agent_1"
    assert state["secrets"] == {"northwind_tool_secret": "sec_1"}
    assert state["tools"] == {"moving_quote": "tool_1"}
    assert sorted(v["id"] for v in state["knowledge_base"].values()) == ["doc_1", "doc_2", "doc_3"]
    secret_body = json.loads(created["secret"].calls.last.request.content)
    assert secret_body["value"] == f"Bearer {TOOL_SECRET}"
    agent_body = json.loads(created["agent"].calls.last.request.content)
    assert agent_body["conversation_config"]["tts"]["voice_id"] == "voiceEn123"
    assert agent_body["conversation_config"]["language_presets"]["ar"]["overrides"]["tts"] == {
        "voice_id": "voiceAr456"
    }
    assert [
        k["id"] for k in agent_body["conversation_config"]["agent"]["prompt"]["knowledge_base"]
    ] == [
        "doc_1",
        "doc_2",
        "doc_3",
    ]

    # Second run: everything is updated in place, nothing new is created.
    updates = {
        "secret": respx_mock.patch("/v1/convai/secrets/sec_1").respond(
            200, json={"type": "stored", "secret_id": "sec_1", "name": "n"}
        ),
        "tool": respx_mock.patch("/v1/convai/tools/tool_1").respond(200, json={"id": "tool_1"}),
        "agent": respx_mock.patch("/v1/convai/agents/agent_1").respond(
            200, json={"agent_id": "agent_1"}
        ),
    }
    assert main(args, env=ENV, client_factory=client_factory) == 0
    for route in created.values():
        assert route.call_count == (3 if route is created["kb"] else 1)
    for route in updates.values():
        assert route.call_count == 1
    assert json.loads(updates["secret"].calls.last.request.content)["type"] == "update"
    assert json.loads(state_path.read_text())["agent_id"] == "agent_1"


@respx.mock(base_url=BASE_URL, assert_all_called=False)
def test_apply_recreates_agent_deleted_elsewhere(respx_mock, brand_dir, client_factory) -> None:
    state_path = brand_dir / "state.local.json"
    state_path.write_text(
        json.dumps(
            {
                "agent_id": "gone",
                "secrets": {"northwind_tool_secret": "sec_1"},
                "tools": {"moving_quote": "tool_1"},
                "knowledge_base": {},
                "tests": {},
            }
        )
    )
    respx_mock.patch("/v1/convai/secrets/sec_1").respond(200, json={"secret_id": "sec_1"})
    respx_mock.patch("/v1/convai/tools/tool_1").respond(200, json={"id": "tool_1"})
    respx_mock.post("/v1/convai/knowledge-base/url").respond(200, json={"id": "doc_x"})
    respx_mock.patch("/v1/convai/agents/gone").respond(404, json={"detail": "not found"})
    create = respx_mock.post("/v1/convai/agents/create").respond(200, json={"agent_id": "new"})
    args = ["--brand", str(brand_dir / "brand.json"), "--state", str(state_path), "--apply"]
    assert main(args, env=ENV, client_factory=client_factory) == 0
    assert create.call_count == 1
    assert json.loads(state_path.read_text())["agent_id"] == "new"


@respx.mock(base_url=BASE_URL, assert_all_called=False)
def test_apply_removes_tool_dropped_from_brand(respx_mock, brand_dir, client_factory) -> None:
    state_path = brand_dir / "state.local.json"
    state_path.write_text(json.dumps({"tools": {"old_tool": "tool_old"}}))
    _mock_create_routes(respx_mock)
    delete = respx_mock.delete("/v1/convai/tools/tool_old").respond(204)
    args = ["--brand", str(brand_dir / "brand.json"), "--state", str(state_path), "--apply"]
    assert main(args, env=ENV, client_factory=client_factory) == 0
    assert delete.called
    assert "old_tool" not in json.loads(state_path.read_text())["tools"]


@respx.mock(base_url=BASE_URL)
def test_teardown_deletes_everything_in_state(respx_mock, tmp_path, client_factory) -> None:
    state_path = tmp_path / "state.local.json"
    state_path.write_text(
        json.dumps(
            {
                "agent_id": "agent_1",
                "secrets": {"s": "sec_1"},
                "tools": {"t": "tool_1"},
                "knowledge_base": {"https://example.com/a.md": {"id": "doc_1", "name": "A"}},
                "tests": {"en-x": "test_1"},
            }
        )
    )
    routes = [
        respx_mock.delete("/v1/convai/agents/agent_1").respond(204),
        respx_mock.delete("/v1/convai/agent-testing/test_1").respond(404, json={"detail": "gone"}),
        respx_mock.delete("/v1/convai/tools/tool_1").respond(204),
        respx_mock.delete("/v1/convai/knowledge-base/doc_1").respond(204),
        respx_mock.delete("/v1/convai/secrets/sec_1").respond(204),
    ]
    assert (
        main(["--teardown", "--state", str(state_path)], env={}, client_factory=client_factory) == 0
    )
    assert all(r.called for r in routes)
    order = [call.request.url.path for call in respx_mock.calls]
    assert order[0].endswith("/agents/agent_1") and order[-1].endswith("/secrets/sec_1")
    state = json.loads(state_path.read_text())
    assert state["agent_id"] is None
    assert not any(state[k] for k in ("secrets", "tools", "knowledge_base", "tests"))


def test_teardown_with_empty_state(tmp_path, client_factory) -> None:
    assert main(["--teardown", "--state", str(tmp_path / "none.json")], env={}) == 0
