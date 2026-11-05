"""Sync tests from YAML to the ElevenLabs Tests API, run them and write results.

Usage::

    python -m voice_agent.evals sync --brand brand.json --tests tests.yaml [--dry-run]
    python -m voice_agent.evals run  --brand brand.json --tests tests.yaml \\
        [--sanitise "Brand Name=>the relocation platform"]

``sync`` creates each test once and updates it on later runs (test IDs are kept in
the state file). ``run`` calls ``POST /v1/convai/agents/{agent_id}/run-tests``, polls
``GET /v1/convai/test-invocations/{id}`` until every test has finished or the
timeout passes, then writes ``results/latest.md`` and the raw JSON to
``results/raw/``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from rich.console import Console

from .client import ElevenLabsClient, ElevenLabsError
from .config import (
    DEFAULT_BASE_CONFIG,
    ConfigError,
    TestSuite,
    build_test_payload,
    load_base_config,
    load_brand,
    load_tests,
)
from .setup_agent import DEFAULT_STATE_FILE, DryRunExecutor, LiveExecutor, State, upsert

err = Console(stderr=True, highlight=False)

FINISHED = {"passed", "failed", "cancelled"}
TYPE_ORDER = {"llm": 0, "tool": 1, "simulation": 2}


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def sync_tests(
    suite: TestSuite,
    state: State,
    ex: DryRunExecutor | LiveExecutor,
    *,
    param_path_prefix: str = "",
    persist: Callable[[], None] = lambda: None,
) -> list[dict[str, Any]]:
    """Create or update every test. Returns the payloads that were sent."""
    tool_ids = dict(state.data["tools"])
    if not ex.live:
        for t in suite.tests:
            if t.tool_call and t.tool_call.name not in tool_ids:
                tool_ids[t.tool_call.name] = f"<tool id: {t.tool_call.name}>"
    payloads = []
    for test in suite.tests:
        payload = build_test_payload(test, tool_ids=tool_ids, param_path_prefix=param_path_prefix)
        payloads.append(payload)
        state.data["tests"][test.id] = upsert(
            ex,
            kind="test",
            label=test.id,
            existing_id=state.data["tests"].get(test.id),
            create_path="/v1/convai/agent-testing/create",
            update_path="/v1/convai/agent-testing/{id}",
            create_body=payload,
            update_body=payload,
            id_field="id",
            update_method="PUT",
        )
        persist()
    wanted = {t.id for t in suite.tests}
    for stale in sorted(k for k in state.data["tests"] if k not in wanted):
        ex.note(f"test {stale} is in the state file but not in the tests file; left in place")
    return payloads


# ---------------------------------------------------------------------------
# Run and poll
# ---------------------------------------------------------------------------


def _all_finished(invocation: dict[str, Any], expected: int) -> bool:
    runs = invocation.get("test_runs") or []
    return len(runs) >= expected and all(r.get("status") in FINISHED for r in runs)


def run_and_wait(
    client: ElevenLabsClient,
    agent_id: str,
    test_ids: list[str],
    *,
    timeout_seconds: float = 600,
    poll_seconds: float = 5,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, Any], bool]:
    """Start the run and poll until finished. Returns (invocation, finished)."""
    started = client.run_tests(agent_id, test_ids)
    invocation_id = started.get("id")
    if not invocation_id:
        raise ElevenLabsError("run-tests response has no invocation 'id'")
    err.print(f"Started test invocation {invocation_id} with {len(test_ids)} tests.")
    deadline = clock() + timeout_seconds
    invocation = started
    while True:
        if _all_finished(invocation, len(test_ids)):
            return invocation, True
        if clock() >= deadline:
            return invocation, False
        sleep(poll_seconds)
        invocation = client.get_test_invocation(invocation_id)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def parse_sanitise(rules: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for rule in rules:
        if "=>" not in rule:
            raise ConfigError(f"--sanitise needs 'FIND=>REPLACE', got: {rule!r}")
        find, replace = rule.split("=>", 1)
        if not find.strip():
            raise ConfigError(f"--sanitise has an empty search text: {rule!r}")
        pairs.append((find.strip(), replace.strip()))
    return pairs


def sanitise_text(text: str, pairs: list[tuple[str, str]]) -> str:
    for find, replace in pairs:
        text = re.sub(re.escape(find), lambda _m, r=replace: r, text, flags=re.IGNORECASE)
    return text


def sanitise_obj(value: Any, pairs: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {k: sanitise_obj(v, pairs) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitise_obj(v, pairs) for v in value]
    if isinstance(value, str):
        return sanitise_text(value, pairs)
    return value


def _cell(text: Any, limit: int = 160) -> str:
    s = " ".join(str(text or "").split())
    if len(s) > limit:
        s = s[: limit - 3].rstrip() + "..."
    return s.replace("|", "\\|")


def summarise(
    suite: TestSuite, invocation: dict[str, Any], test_ids: Mapping[str, str]
) -> list[dict[str, Any]]:
    """One row per YAML test with its outcome."""
    by_remote = {r.get("test_id"): r for r in invocation.get("test_runs") or []}
    rows = []
    for test in suite.tests:
        run = by_remote.get(test_ids.get(test.id, ""), {})
        status = run.get("status") or "not run"
        if status == "pending":
            status = "not finished"
        rationale = ((run.get("condition_result") or {}).get("rationale") or {}).get("summary")
        rows.append(
            {
                "id": test.id,
                "language": test.language,
                "dialect": test.dialect or "",
                "type": test.type,
                "status": status,
                "notes": rationale or "",
            }
        )
    return rows


def render_results(
    rows: list[dict[str, Any]],
    *,
    run_at: datetime,
    llm: str,
    finished: bool,
) -> str:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["language"], row["type"])].append(row)

    lines = [
        "# Evaluation results",
        "",
        f"Run on {run_at.strftime('%d %B %Y at %H:%M UTC')} with the ElevenLabs Tests API. "
        f"Agent LLM: `{llm}`. {len(rows)} tests.",
    ]
    if not finished:
        lines += ["", "Some tests had not finished when the polling timeout passed."]
    lines += [
        "",
        "## By language and test type",
        "",
        "| Language | Type | Passed | Total | Pass rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for lang, typ in sorted(groups, key=lambda k: (k[0], TYPE_ORDER.get(k[1], 9))):
        group = groups[(lang, typ)]
        passed = sum(1 for r in group if r["status"] == "passed")
        lines.append(f"| {lang} | {typ} | {passed} | {len(group)} | {passed / len(group):.0%} |")
    total_passed = sum(1 for r in rows if r["status"] == "passed")
    rate = f"{total_passed / len(rows):.0%}" if rows else "n/a"
    lines.append(f"| **all** | **all** | **{total_passed}** | **{len(rows)}** | **{rate}** |")
    lines += [
        "",
        "## Per test",
        "",
        "| Test | Language | Dialect | Type | Result | Evaluator summary |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['language']} | {r['dialect']} | {r['type']} | "
            f"{r['status']} | {_cell(r['notes'])} |"
        )
    return "\n".join(lines) + "\n"


def write_results(
    suite: TestSuite,
    invocation: dict[str, Any],
    test_ids: Mapping[str, str],
    *,
    results_dir: Path,
    llm: str,
    finished: bool,
    sanitise: list[tuple[str, str]] | None = None,
    run_at: datetime | None = None,
) -> tuple[Path, Path]:
    run_at = run_at or datetime.now(UTC)
    pairs = sanitise or []
    rows = summarise(suite, invocation, test_ids)
    markdown = sanitise_text(render_results(rows, run_at=run_at, llm=llm, finished=finished), pairs)
    results_dir.mkdir(parents=True, exist_ok=True)
    latest = results_dir / "latest.md"
    latest.write_text(markdown, encoding="utf-8")
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / f"{run_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    raw.write_text(
        json.dumps(sanitise_obj(invocation, pairs), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return latest, raw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _agent_llm(bundle_llm: str | None, base_config_path: str) -> str:
    if bundle_llm:
        return bundle_llm
    base = load_base_config(base_config_path)
    return base["conversation_config"]["agent"]["prompt"]["llm"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m voice_agent.evals",
        description="Sync and run ElevenLabs agent tests defined in YAML.",
    )
    p.add_argument("command", choices=["sync", "run"])
    p.add_argument("--brand", required=True, help="path to the brand JSON file")
    p.add_argument("--tests", required=True, help="path to the tests YAML file")
    p.add_argument("--state", default=DEFAULT_STATE_FILE, help="state file (default: %(default)s)")
    p.add_argument("--dry-run", action="store_true", help="sync: print payloads, send nothing")
    p.add_argument(
        "--results-dir", default="results", help="run: output folder (default: %(default)s)"
    )
    p.add_argument("--timeout", type=float, default=600, help="run: seconds to wait (default: 600)")
    p.add_argument("--poll", type=float, default=5, help="run: seconds between polls (default: 5)")
    p.add_argument(
        "--sanitise",
        action="append",
        default=[],
        metavar="FIND=>REPLACE",
        help="run: replace text in the written results; repeat for more rules",
    )
    p.add_argument(
        "--param-path-prefix",
        default="",
        help="sync: prefix for tool parameter paths in tool tests (default: none)",
    )
    p.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG), help=argparse.SUPPRESS)
    return p


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[], ElevenLabsClient] = ElevenLabsClient,
    out=None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(find_dotenv(usecwd=True))
    out = out or sys.stdout
    try:
        bundle = load_brand(args.brand)
        suite = load_tests(args.tests, bundle.brand)
        state = State.load(args.state)

        if args.command == "sync":
            if args.dry_run:
                ex = DryRunExecutor()
                sync_tests(suite, state.copy(), ex, param_path_prefix=args.param_path_prefix)
                out.write(
                    json.dumps(
                        {"mode": "dry-run", "steps": ex.steps, "notes": ex.notes},
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                return 0
            with client_factory() as client:
                sync_tests(
                    suite,
                    state,
                    LiveExecutor(client),
                    param_path_prefix=args.param_path_prefix,
                    persist=state.save,
                )
            err.print(f"[green]Synced {len(suite.tests)} tests.[/green] IDs saved to {args.state}.")
            return 0

        # run
        sanitise = parse_sanitise(args.sanitise)
        agent_id = state.data.get("agent_id")
        if not agent_id:
            raise ConfigError(f"no agent_id in {args.state}. Run setup_agent --apply first.")
        missing = [t.id for t in suite.tests if t.id not in state.data["tests"]]
        if missing:
            raise ConfigError(f"tests not synced yet: {', '.join(missing)}. Run 'sync' first.")
        test_ids = {t.id: state.data["tests"][t.id] for t in suite.tests}
        with client_factory() as client:
            invocation, finished = run_and_wait(
                client,
                agent_id,
                list(test_ids.values()),
                timeout_seconds=args.timeout,
                poll_seconds=args.poll,
                sleep=sleep,
            )
        latest, raw = write_results(
            suite,
            invocation,
            test_ids,
            results_dir=Path(args.results_dir),
            llm=_agent_llm(bundle.brand.llm, args.base_config),
            finished=finished,
            sanitise=sanitise,
        )
        err.print(f"Wrote {latest} and {raw}.")
        if not finished:
            err.print("[yellow]warning:[/yellow] some tests had not finished; rerun later.")
            return 1
        return 0
    except (ConfigError, ElevenLabsError) as exc:
        err.print(f"[red]error:[/red] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
