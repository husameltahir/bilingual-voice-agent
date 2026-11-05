from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from voice_agent.client import BASE_URL
from voice_agent.config import (
    ConfigError,
    build_parameter_eval,
    build_test_payload,
    load_tests,
)
from voice_agent.evals import main, parse_sanitise, render_results, summarise, write_results


@pytest.fixture
def suite(example_tests_path):
    return load_tests(example_tests_path)


def _test(suite, test_id):
    return next(t for t in suite.tests if t.id == test_id)


# -- YAML to API payloads --------------------------------------------------------


def test_llm_payload(suite) -> None:
    payload = build_test_payload(_test(suite, "ar-msa-kb-school-year-groups"))
    assert payload["type"] == "llm"
    assert payload["name"] == "ar-msa-kb-school-year-groups"
    assert payload["chat_history"] == [
        {
            "role": "user",
            "message": "ابني عمره سبع سنوات. في أي صف سيكون في مدرسة بريطانية في دبي؟",
            "time_in_call_secs": 0,
        }
    ]
    assert payload["success_condition"].startswith("The reply is in Arabic")
    assert payload["success_examples"][0]["type"] == "success"
    assert payload["failure_examples"] == [
        {"response": "Your son will join Year 3.", "type": "failure"}
    ]


def test_tool_payload(suite) -> None:
    payload = build_test_payload(
        _test(suite, "en-tool-moving-quote"), tool_ids={"moving_quote": "tool_1"}
    )
    assert payload["type"] == "tool"
    assert [t["role"] for t in payload["chat_history"]] == ["user", "agent", "user"]
    assert [t["time_in_call_secs"] for t in payload["chat_history"]] == [0, 1, 2]
    params = payload["tool_call_parameters"]
    assert params["referenced_tool"] == {"id": "tool_1", "type": "webhook"}
    assert params["parameters"] == [
        {"path": "origin_country", "eval": {"type": "exact", "expected_value": "uk"}},
        {"path": "destination_city", "eval": {"type": "exact", "expected_value": "dubai"}},
        {"path": "bedrooms", "eval": {"type": "exact", "expected_value": "3"}},
    ]
    assert "success_condition" not in payload


def test_tool_payload_with_path_prefix(suite) -> None:
    payload = build_test_payload(
        _test(suite, "en-tool-moving-quote"),
        tool_ids={"moving_quote": "tool_1"},
        param_path_prefix="body.",
    )
    assert payload["tool_call_parameters"]["parameters"][0]["path"] == "body.origin_country"


def test_tool_payload_needs_tool_id(suite) -> None:
    with pytest.raises(ConfigError, match="setup_agent --apply"):
        build_test_payload(_test(suite, "en-tool-moving-quote"), tool_ids={})


def test_simulation_payload_prefers_simulation_condition(suite) -> None:
    payload = build_test_payload(_test(suite, "en-sim-switch-to-arabic"))
    assert payload["type"] == "simulation"
    assert payload["simulation_scenario"].startswith("A caller who asks in English")
    assert payload["simulation_max_turns"] == 4
    assert payload["success_condition"].startswith("Every agent reply after the switch")
    assert "chat_history" not in payload


@pytest.mark.parametrize(
    ("expected", "eval_"),
    [
        (95000, {"type": "exact", "expected_value": "95000"}),
        (95000.0, {"type": "exact", "expected_value": "95000"}),
        (True, {"type": "exact", "expected_value": "true"}),
        ("dubai", {"type": "exact", "expected_value": "dubai"}),
        ({"regex": "^9"}, {"type": "regex", "pattern": "^9"}),
        ({"llm": "a UAE city"}, {"type": "llm", "description": "a UAE city"}),
        ({"anything": True}, {"type": "anything"}),
    ],
)
def test_parameter_evals(expected, eval_) -> None:
    assert build_parameter_eval("x", expected) == {"path": "x", "eval": eval_}


def test_parameter_eval_rejects_unknown_check() -> None:
    with pytest.raises(ConfigError):
        build_parameter_eval("x", {"fuzzy": 1})


# -- sync ------------------------------------------------------------------------


@respx.mock(base_url=BASE_URL, assert_all_called=False)
def test_sync_creates_then_updates(respx_mock, brand_dir: Path, client_factory) -> None:
    state_path = brand_dir / "state.local.json"
    state_path.write_text(json.dumps({"agent_id": "agent_1", "tools": {"moving_quote": "tool_1"}}))
    counter = iter(range(100))
    create = respx_mock.post("/v1/convai/agent-testing/create").mock(
        side_effect=lambda _r: httpx.Response(200, json={"id": f"test_{next(counter)}"})
    )
    args = [
        "sync",
        "--brand",
        str(brand_dir / "brand.json"),
        "--tests",
        str(brand_dir / "tests.yaml"),
        "--state",
        str(state_path),
    ]
    assert main(args, client_factory=client_factory) == 0
    assert create.call_count == 6
    tests_state = json.loads(state_path.read_text())["tests"]
    assert len(tests_state) == 6

    put = respx_mock.route(method="PUT", path__startswith="/v1/convai/agent-testing/").respond(
        200, json={"id": "ok"}
    )
    assert main(args, client_factory=client_factory) == 0
    assert create.call_count == 6
    assert put.call_count == 6
    assert json.loads(state_path.read_text())["tests"] == tests_state


def test_sync_dry_run_sends_nothing(brand_dir: Path) -> None:
    out = io.StringIO()
    code = main(
        [
            "sync",
            "--dry-run",
            "--brand",
            str(brand_dir / "brand.json"),
            "--tests",
            str(brand_dir / "tests.yaml"),
            "--state",
            str(brand_dir / "s.json"),
        ],
        out=out,
    )
    assert code == 0
    steps = json.loads(out.getvalue())["steps"]
    assert len(steps) == 6
    assert all(s["path"] == "/v1/convai/agent-testing/create" for s in steps)
    assert not (brand_dir / "s.json").exists()


# -- run and results ---------------------------------------------------------------


def _synced_state(brand_dir: Path, suite) -> Path:
    state_path = brand_dir / "state.local.json"
    state_path.write_text(
        json.dumps(
            {
                "agent_id": "agent_1",
                "tools": {"moving_quote": "tool_1"},
                "tests": {t.id: f"remote-{t.id}" for t in suite.tests},
            }
        )
    )
    return state_path


def _invocation(suite, statuses: dict[str, str]) -> dict:
    return {
        "id": "inv_1",
        "test_runs": [
            {
                "test_run_id": f"run-{t.id}",
                "test_id": f"remote-{t.id}",
                "status": statuses.get(t.id, "passed"),
                "condition_result": {
                    "result": "success",
                    "rationale": {"summary": f"Northwind answer for {t.id} | ok"},
                },
            }
            for t in suite.tests
        ],
    }


@respx.mock(base_url=BASE_URL)
def test_run_polls_and_writes_results(respx_mock, brand_dir, suite, client_factory) -> None:
    state_path = _synced_state(brand_dir, suite)
    pending = _invocation(suite, {t.id: "pending" for t in suite.tests})
    done = _invocation(
        suite, {"en-tool-moving-quote": "failed", "ar-gulf-tool-moving-quote": "failed"}
    )
    run = respx_mock.post("/v1/convai/agents/agent_1/run-tests").respond(200, json=pending)
    poll = respx_mock.get("/v1/convai/test-invocations/inv_1").mock(
        side_effect=[httpx.Response(200, json=pending), httpx.Response(200, json=done)]
    )
    results = brand_dir / "results"
    code = main(
        [
            "run",
            "--brand",
            str(brand_dir / "brand.json"),
            "--tests",
            str(brand_dir / "tests.yaml"),
            "--state",
            str(state_path),
            "--results-dir",
            str(results),
            "--sanitise",
            "Northwind Relocation=>the platform",
            "--sanitise",
            "northwind=>the platform",
        ],
        client_factory=client_factory,
        sleep=lambda _s: None,
    )
    assert code == 0
    assert run.called and poll.call_count == 2
    sent = json.loads(run.calls.last.request.content)
    assert sent == {"tests": [{"test_id": f"remote-{t.id}"} for t in suite.tests]}

    latest = (results / "latest.md").read_text()
    assert "| en | llm | 2 | 2 | 100% |" in latest
    assert "| en | tool | 0 | 1 | 0% |" in latest
    assert "| ar | tool | 0 | 1 | 0% |" in latest
    assert "| en | simulation | 1 | 1 | 100% |" in latest
    assert "| **all** | **all** | **4** | **6** | **67%** |" in latest
    assert "| en-tool-moving-quote | en |  | tool | failed |" in latest
    assert "Northwind" not in latest and "northwind" not in latest
    assert "\\|" in latest  # pipes in notes are escaped
    raw_files = list((results / "raw").glob("*.json"))
    assert len(raw_files) == 1
    assert "northwind" not in raw_files[0].read_text().lower()


@respx.mock(base_url=BASE_URL, assert_all_called=False)
def test_run_times_out(respx_mock, brand_dir, suite, client_factory) -> None:
    state_path = _synced_state(brand_dir, suite)
    pending = _invocation(suite, {t.id: "pending" for t in suite.tests})
    respx_mock.post("/v1/convai/agents/agent_1/run-tests").respond(200, json=pending)
    respx_mock.get("/v1/convai/test-invocations/inv_1").respond(200, json=pending)
    code = main(
        [
            "run",
            "--brand",
            str(brand_dir / "brand.json"),
            "--tests",
            str(brand_dir / "tests.yaml"),
            "--state",
            str(state_path),
            "--results-dir",
            str(brand_dir / "results"),
            "--timeout",
            "0",
        ],
        client_factory=client_factory,
        sleep=lambda _s: None,
    )
    assert code == 1
    latest = (brand_dir / "results" / "latest.md").read_text()
    assert "had not finished" in latest
    assert "not finished" in latest


def test_run_requires_sync(brand_dir) -> None:
    (brand_dir / "s.json").write_text(json.dumps({"agent_id": "agent_1"}))
    code = main(
        [
            "run",
            "--brand",
            str(brand_dir / "brand.json"),
            "--tests",
            str(brand_dir / "tests.yaml"),
            "--state",
            str(brand_dir / "s.json"),
        ]
    )
    assert code == 1


def test_results_table_render(suite, tmp_path) -> None:
    ids = {t.id: f"remote-{t.id}" for t in suite.tests}
    inv = _invocation(suite, {"ar-msa-kb-school-year-groups": "failed"})
    rows = summarise(suite, inv, ids)
    md = render_results(
        rows, run_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC), llm="claude-haiku-4-5", finished=True
    )
    assert md.startswith("# Evaluation results")
    assert "02 January 2026 at 03:04 UTC" in md
    assert "| ar | llm | 0 | 1 | 0% |" in md
    latest, raw = write_results(
        suite,
        inv,
        ids,
        results_dir=tmp_path,
        llm="m",
        finished=True,
        sanitise=parse_sanitise(["Northwind=>X"]),
        run_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert latest.name == "latest.md" and raw.parent.name == "raw"
    assert "Northwind" not in latest.read_text()


def test_parse_sanitise_rejects_bad_rule() -> None:
    with pytest.raises(ConfigError):
        parse_sanitise(["no arrow here"])
