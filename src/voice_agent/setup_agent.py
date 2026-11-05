"""Create or update the voice agent from a brand file.

Usage::

    python -m voice_agent.setup_agent --brand examples/brand.example.json            # dry run
    python -m voice_agent.setup_agent --brand path/to/brand.json --apply
    python -m voice_agent.setup_agent --teardown

Order: workspace secret, webhook tool(s), knowledge base documents, agent. Every ID
is saved to the state file (``state.local.json`` in the current folder by default),
so a second run updates the same resources instead of creating new ones.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from rich.console import Console

from .client import ElevenLabsClient, ElevenLabsError, NotFoundError
from .config import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_PROMPT_TEMPLATE,
    BrandBundle,
    ConfigError,
    build_agent_payload,
    build_kb_payload,
    build_prompt,
    build_secret_payload,
    build_secret_update_payload,
    build_tool_payload,
    kb_locator,
    load_base_config,
    load_brand,
)

DEFAULT_STATE_FILE = "state.local.json"
MASK = "****"

err = Console(stderr=True, highlight=False)


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------


class State:
    """IDs of everything this tool created, keyed so reruns can find them."""

    KEYS = ("secrets", "tools", "knowledge_base", "tests")

    def __init__(self, path: Path, data: dict[str, Any] | None = None):
        self.path = path
        self.data: dict[str, Any] = data or {}
        self.data.setdefault("agent_id", None)
        for key in self.KEYS:
            self.data.setdefault(key, {})

    @classmethod
    def load(cls, path: str | Path) -> State:
        path = Path(path)
        if path.is_file():
            try:
                return cls(path, json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:
                raise ConfigError(f"state file is not valid JSON: {path}: {exc}") from exc
        return cls(path)

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def copy(self) -> State:
        return State(self.path, copy.deepcopy(self.data))

    def is_empty(self) -> bool:
        return not self.data.get("agent_id") and not any(self.data[k] for k in self.KEYS)


# ---------------------------------------------------------------------------
# Executors: one code path for dry run and apply
# ---------------------------------------------------------------------------


def mask_secrets(value: Any, secrets: list[str]) -> Any:
    """Replace every occurrence of a secret value with ****."""
    if isinstance(value, dict):
        return {k: mask_secrets(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_secrets(v, secrets) for v in value]
    if isinstance(value, str):
        for s in sorted((s for s in secrets if s), key=len, reverse=True):
            value = value.replace(s, MASK)
        return value
    return value


class DryRunExecutor:
    """Records each call instead of sending it, with secrets masked."""

    live = False

    def __init__(self, secrets: list[str] | None = None):
        self.secrets = list(secrets or [])
        self.steps: list[dict[str, Any]] = []
        self.notes: list[str] = []

    def call(
        self,
        step: str,
        method: str,
        path: str,
        body: Any = None,
        fake: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {"step": step, "method": method, "path": path}
        if body is not None:
            entry["body"] = mask_secrets(body, self.secrets)
        self.steps.append(entry)
        return dict(fake or {})

    def note(self, message: str) -> None:
        self.notes.append(message)


class LiveExecutor:
    """Sends each call to ElevenLabs."""

    live = True

    def __init__(self, client: ElevenLabsClient, console: Console = err):
        self.client = client
        self.console = console

    def call(
        self,
        step: str,
        method: str,
        path: str,
        body: Any = None,
        fake: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.console.print(f"[bold]{step}[/bold]  {method} {path}")
        result = self.client.request(method, path, json=body)
        return result if isinstance(result, dict) else {}

    def note(self, message: str) -> None:
        self.console.print(f"[yellow]note:[/yellow] {message}")


DEFAULT_RAG_MODEL = "e5_mistral_7b_instruct"
RAG_POLL_SECONDS = 5.0
RAG_POLL_LIMIT = 60


def index_knowledge_base(
    ex: DryRunExecutor | LiveExecutor,
    doc_ids: list[str],
    model: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Start a RAG index for each document and wait until every one has succeeded."""
    pending = list(doc_ids)
    for attempt in range(RAG_POLL_LIMIT):
        still: list[str] = []
        for doc_id in pending:
            status = ex.call(
                f"index knowledge base document {doc_id}",
                "POST",
                f"/v1/convai/knowledge-base/{doc_id}/rag-index",
                {"model": model},
                fake={"status": "succeeded"},
            ).get("status")
            if status in (
                "failed",
                "rag_limit_exceeded",
                "document_too_small",
                "cannot_index_folder",
            ):
                raise ElevenLabsError(f"RAG index for document {doc_id}: {status}")
            if status != "succeeded":
                still.append(doc_id)
        if not still:
            return
        pending = still
        if attempt == 0:
            ex.note(f"waiting for {len(still)} RAG index(es) to finish")
        sleep(RAG_POLL_SECONDS)
    raise ElevenLabsError(f"RAG indexes not ready after {RAG_POLL_LIMIT} checks: {pending}")


def upsert(
    ex: DryRunExecutor | LiveExecutor,
    *,
    kind: str,
    label: str,
    existing_id: str | None,
    create_path: str,
    update_path: str,
    create_body: Any,
    update_body: Any,
    id_field: str,
    update_method: str = "PATCH",
) -> str:
    """Update the resource if the state file knows it, otherwise create it."""
    if existing_id:
        try:
            ex.call(
                f"update {kind} {label}",
                update_method,
                update_path.format(id=existing_id),
                update_body,
            )
            return existing_id
        except NotFoundError:
            ex.note(f"{kind} {label} ({existing_id}) no longer exists; creating it again")
    response = ex.call(
        f"create {kind} {label}",
        "POST",
        create_path,
        create_body,
        fake={id_field: f"<new {kind} id: {label}>"},
    )
    new_id = response.get(id_field)
    if not new_id:
        raise ElevenLabsError(f"create {kind} {label}: response has no '{id_field}'")
    return str(new_id)


def _delete(ex: DryRunExecutor | LiveExecutor, step: str, path: str) -> None:
    try:
        ex.call(step, "DELETE", path)
    except NotFoundError:
        ex.note(f"{step}: already gone")


# ---------------------------------------------------------------------------
# Setup and teardown
# ---------------------------------------------------------------------------


def run_setup(
    bundle: BrandBundle,
    base_config: dict[str, Any],
    prompt: str,
    state: State,
    ex: DryRunExecutor | LiveExecutor,
    env: Mapping[str, str],
    persist: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Run every step in order and return the final agent payload."""
    brand = bundle.brand
    data = state.data

    # 1. Workspace secrets (one per secret_name).
    secret_ids: dict[str, str] = {}
    for tool in brand.webhook_tools:
        if tool.secret_name is None or tool.secret_name in secret_ids:
            continue
        assert tool.secret_env is not None
        token = env.get(tool.secret_env) or f"<value of {tool.secret_env}>"
        secret_ids[tool.secret_name] = upsert(
            ex,
            kind="secret",
            label=tool.secret_name,
            existing_id=data["secrets"].get(tool.secret_name),
            create_path="/v1/convai/secrets",
            update_path="/v1/convai/secrets/{id}",
            create_body=build_secret_payload(tool.secret_name, token),
            update_body=build_secret_update_payload(tool.secret_name, token),
            id_field="secret_id",
        )
        data["secrets"][tool.secret_name] = secret_ids[tool.secret_name]
        persist()

    # 2. Webhook tools.
    tool_ids: list[str] = []
    for tool in brand.webhook_tools:
        secret_id = secret_ids.get(tool.secret_name) if tool.secret_name else None
        payload = build_tool_payload(tool, secret_id)
        tool_id = upsert(
            ex,
            kind="tool",
            label=tool.name,
            existing_id=data["tools"].get(tool.name),
            create_path="/v1/convai/tools",
            update_path="/v1/convai/tools/{id}",
            create_body=payload,
            update_body=payload,
            id_field="id",
        )
        data["tools"][tool.name] = tool_id
        tool_ids.append(tool_id)
        persist()

    # 3. Knowledge base documents (one per URL; an existing URL is reused).
    locators: list[dict[str, Any]] = []
    for doc in brand.knowledge_base:
        known = data["knowledge_base"].get(doc.url)
        if known:
            doc_id = known["id"]
            if known.get("name") != doc.name:
                ex.note(f"knowledge base name for {doc.url} changed; the agent uses the new name")
        else:
            response = ex.call(
                f"create knowledge base document {doc.name}",
                "POST",
                "/v1/convai/knowledge-base/url",
                build_kb_payload(doc),
                fake={"id": f"<new document id: {doc.name}>"},
            )
            doc_id = response.get("id")
            if not doc_id:
                raise ElevenLabsError(f"create knowledge base document {doc.name}: no 'id'")
        data["knowledge_base"][doc.url] = {"id": doc_id, "name": doc.name}
        locators.append(kb_locator(doc_id, doc.name))
        persist()

    # 4. Agent.
    agent_payload = build_agent_payload(
        bundle, base_config, tool_ids=tool_ids, knowledge_base=locators, prompt=prompt
    )
    data["agent_id"] = upsert(
        ex,
        kind="agent",
        label=brand.agent_name,
        existing_id=data.get("agent_id"),
        create_path="/v1/convai/agents/create",
        update_path="/v1/convai/agents/{id}",
        create_body=agent_payload,
        update_body=agent_payload,
        id_field="agent_id",
    )
    persist()

    # 4b. RAG indexes. Attaching a document does not index it, and an agent with
    # RAG on and no index retrieves nothing, so it declines questions its own
    # documents answer. POST is idempotent: it starts indexing or reports status.
    rag = agent_payload["conversation_config"]["agent"]["prompt"].get("rag") or {}
    if rag.get("enabled") and locators:
        index_knowledge_base(
            ex, [loc["id"] for loc in locators], rag.get("embedding_model") or DEFAULT_RAG_MODEL
        )

    # 5. Remove resources this state file created that the brand file no longer lists.
    wanted_tools = {t.name for t in brand.webhook_tools}
    for name in [n for n in data["tools"] if n not in wanted_tools]:
        _delete(ex, f"delete old tool {name}", f"/v1/convai/tools/{data['tools'][name]}")
        del data["tools"][name]
        persist()
    wanted_urls = {d.url for d in brand.knowledge_base}
    for url in [u for u in data["knowledge_base"] if u not in wanted_urls]:
        doc_id = data["knowledge_base"][url]["id"]
        _delete(
            ex, f"delete old knowledge base document {url}", f"/v1/convai/knowledge-base/{doc_id}"
        )
        del data["knowledge_base"][url]
        persist()
    wanted_secrets = {t.secret_name for t in brand.webhook_tools if t.secret_name}
    for name in [n for n in data["secrets"] if n not in wanted_secrets]:
        _delete(ex, f"delete old secret {name}", f"/v1/convai/secrets/{data['secrets'][name]}")
        del data["secrets"][name]
        persist()

    return agent_payload


def run_teardown(
    state: State,
    ex: DryRunExecutor | LiveExecutor,
    persist: Callable[[], None] = lambda: None,
) -> None:
    """Delete everything the state file lists: agent, tests, tools, documents, secrets."""
    data = state.data
    if data.get("agent_id"):
        _delete(ex, "delete agent", f"/v1/convai/agents/{data['agent_id']}")
        data["agent_id"] = None
        persist()
    for key in list(data["tests"]):
        _delete(ex, f"delete test {key}", f"/v1/convai/agent-testing/{data['tests'][key]}")
        del data["tests"][key]
        persist()
    for name in list(data["tools"]):
        _delete(ex, f"delete tool {name}", f"/v1/convai/tools/{data['tools'][name]}")
        del data["tools"][name]
        persist()
    for url in list(data["knowledge_base"]):
        doc_id = data["knowledge_base"][url]["id"]
        _delete(ex, f"delete knowledge base document {url}", f"/v1/convai/knowledge-base/{doc_id}")
        del data["knowledge_base"][url]
        persist()
    for name in list(data["secrets"]):
        _delete(ex, f"delete secret {name}", f"/v1/convai/secrets/{data['secrets'][name]}")
        del data["secrets"][name]
        persist()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _secret_values(bundle: BrandBundle, env: Mapping[str, str]) -> list[str]:
    values = []
    for tool in bundle.brand.webhook_tools:
        if tool.secret_env and env.get(tool.secret_env):
            values.append(env[tool.secret_env])
    return values


def _missing_secrets(bundle: BrandBundle, env: Mapping[str, str]) -> list[str]:
    return sorted(
        {
            t.secret_env
            for t in bundle.brand.webhook_tools
            if t.secret_env and not env.get(t.secret_env)
        }
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m voice_agent.setup_agent",
        description="Create or update an ElevenLabs voice agent from a brand file.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the payloads (default)")
    mode.add_argument("--apply", action="store_true", help="create or update at ElevenLabs")
    mode.add_argument(
        "--teardown", action="store_true", help="delete everything the state file lists"
    )
    p.add_argument("--brand", help="path to the brand JSON file")
    p.add_argument("--state", default=DEFAULT_STATE_FILE, help="state file (default: %(default)s)")
    p.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG), help=argparse.SUPPRESS)
    p.add_argument(
        "--prompt-template", default=str(DEFAULT_PROMPT_TEMPLATE), help=argparse.SUPPRESS
    )
    return p


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[], ElevenLabsClient] = ElevenLabsClient,
    env: Mapping[str, str] | None = None,
    out=None,
) -> int:
    args = build_parser().parse_args(argv)
    if env is None:
        load_dotenv(find_dotenv(usecwd=True))
        env = os.environ
    out = out or sys.stdout
    state = State.load(args.state)

    try:
        if args.teardown:
            if state.is_empty():
                err.print(f"Nothing to delete: {args.state} lists no resources.")
                return 0
            with client_factory() as client:
                run_teardown(state, LiveExecutor(client), persist=state.save)
            err.print(f"[green]Teardown complete.[/green] {args.state} is now empty.")
            return 0

        if not args.brand:
            err.print("[red]error:[/red] --brand is required for a dry run or --apply")
            return 2
        bundle = load_brand(args.brand)
        base = load_base_config(args.base_config)
        prompt = build_prompt(bundle, args.prompt_template)
        placeholders = bundle.brand.placeholder_voices()

        if not args.apply:
            if placeholders:
                err.print(
                    "[yellow]warning:[/yellow] voice_id is still a placeholder for: "
                    + ", ".join(placeholders)
                    + ". --apply will refuse to run until these are real voice IDs."
                )
            missing = _missing_secrets(bundle, env)
            if missing:
                err.print(
                    "[yellow]warning:[/yellow] not set (needed for --apply): " + ", ".join(missing)
                )
            ex = DryRunExecutor(_secret_values(bundle, env))
            run_setup(bundle, base, prompt, state.copy(), ex, env)
            result = {
                "mode": "dry-run",
                "brand": str(args.brand),
                "state_file": str(args.state),
                "steps": ex.steps,
                "notes": ex.notes,
            }
            out.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            err.print("Dry run only. Nothing was sent. Run again with --apply to create or update.")
            return 0

        if placeholders:
            err.print(
                "[red]error:[/red] refusing to --apply: voice_id is still a placeholder for "
                + ", ".join(placeholders)
                + ". Put real voice IDs in the brand file."
            )
            return 2
        missing = _missing_secrets(bundle, env)
        if missing:
            err.print(
                "[red]error:[/red] set these environment variables first: " + ", ".join(missing)
            )
            return 2
        with client_factory() as client:
            run_setup(bundle, base, prompt, state, LiveExecutor(client), env, persist=state.save)
        err.print(
            f"[green]Done.[/green] Agent {state.data['agent_id']}. IDs saved to {args.state}."
        )
        return 0
    except (ConfigError, ElevenLabsError) as exc:
        err.print(f"[red]error:[/red] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
