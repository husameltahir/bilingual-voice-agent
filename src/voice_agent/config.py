"""Load and validate the brand and tests files, and build ElevenLabs payloads.

The brand file holds everything that differs between deployments (names, voices,
first messages, knowledge base URLs, webhook tools, limits). The base config in
``agent/base.config.json`` holds the settings every deployment shares, with
``{{placeholders}}`` that are filled from the brand file.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# agent/ lives at the repository root; a wheel install carries a copy inside the package.
_REPO_AGENT_DIR = Path(__file__).resolve().parents[2] / "agent"
_PACKAGED_AGENT_DIR = Path(__file__).resolve().parent / "agent"
AGENT_DIR = (
    _REPO_AGENT_DIR if (_REPO_AGENT_DIR / "base.config.json").is_file() else _PACKAGED_AGENT_DIR
)
DEFAULT_BASE_CONFIG = AGENT_DIR / "base.config.json"
DEFAULT_PROMPT_TEMPLATE = AGENT_DIR / "prompt.template.md"

# Header used for webhook tool authentication. ElevenLabs request_headers values are
# either a plain string or a {"secret_id": ...} reference, with no way to join a
# literal prefix to a secret. So the stored workspace secret holds the whole header
# value, "Bearer <token>".
AUTH_HEADER = "Authorization"
AUTH_SCHEME = "Bearer "

PLACEHOLDER_VOICE = re.compile(r"^(VOICE_ID_[A-Z0-9_]+|)$")
LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")
TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
TEST_ID = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_WHOLE_PLACEHOLDER = re.compile(r"^\{\{\s*([a-zA-Z0-9_]+)\s*\}\}$")
_INLINE_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


class ConfigError(Exception):
    """Raised when a brand, tests or base config file is missing or invalid."""


class _Strict(BaseModel):
    """Rejects unknown keys, except comment keys that start with "_" or "$"."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_comment_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {
                k: v
                for k, v in data.items()
                if not (isinstance(k, str) and k.startswith(("_", "$")))
            }
        return data


# ---------------------------------------------------------------------------
# Brand file
# ---------------------------------------------------------------------------


class LanguageSettings(_Strict):
    first_message: str = Field(min_length=1)
    voice_id: str

    @property
    def voice_is_placeholder(self) -> bool:
        return bool(PLACEHOLDER_VOICE.match(self.voice_id.strip()))


class KnowledgeBaseDoc(_Strict):
    name: str = Field(min_length=1)
    url: str

    @field_validator("url")
    @classmethod
    def _http_url(cls, v: str) -> str:
        if not v.startswith(("https://", "http://")):
            raise ValueError("knowledge base url must start with https:// or http://")
        return v


class BodyProperty(_Strict):
    type: Literal["string", "number", "integer", "boolean"]
    description: str = Field(min_length=1)
    enum: list[str] | None = None

    @model_validator(mode="after")
    def _enum_only_for_strings(self) -> BodyProperty:
        if self.enum is not None and self.type != "string":
            raise ValueError("enum is only supported for string properties")
        return self


class BodySchema(_Strict):
    type: Literal["object"] = "object"
    description: str | None = None
    required: list[str] = Field(default_factory=list)
    properties: dict[str, BodyProperty] = Field(min_length=1)

    @model_validator(mode="after")
    def _required_are_defined(self) -> BodySchema:
        missing = [name for name in self.required if name not in self.properties]
        if missing:
            raise ValueError(f"required lists properties that are not defined: {missing}")
        return self


class WebhookTool(_Strict):
    name: str
    description: str = Field(min_length=1)
    url: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    secret_env: str | None = None
    secret_name: str | None = None
    timeout_seconds: int = Field(default=20, ge=1, le=120)
    body_schema: BodySchema | None = None

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not TOOL_NAME.match(v):
            raise ValueError("tool name may use letters, digits, '_' and '-' (max 64)")
        return v

    @field_validator("url")
    @classmethod
    def _https(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("webhook url must start with https://")
        return v

    @model_validator(mode="after")
    def _secret_pair(self) -> WebhookTool:
        if (self.secret_env is None) != (self.secret_name is None):
            raise ValueError("set both secret_env and secret_name, or neither")
        return self


class Limits(_Strict):
    max_duration_seconds: int = Field(gt=0)
    daily_limit: int = Field(gt=0)
    concurrency_limit: int = Field(gt=0)


class Privacy(_Strict):
    record_voice: bool
    retention_days: int = Field(ge=-1)
    delete_transcript_and_pii: bool | None = None


class Brand(_Strict):
    agent_name: str = Field(min_length=1)
    site_name: str = Field(min_length=1)
    prompt_file: str = Field(min_length=1)
    default_language: str
    languages: dict[str, LanguageSettings] = Field(min_length=1)
    llm: str | None = None
    knowledge_base: list[KnowledgeBaseDoc] = Field(default_factory=list)
    webhook_tools: list[WebhookTool] = Field(default_factory=list)
    limits: Limits
    privacy: Privacy

    @model_validator(mode="after")
    def _cross_checks(self) -> Brand:
        bad = [code for code in self.languages if not LANGUAGE_CODE.match(code)]
        if bad:
            raise ValueError(f"language codes must look like 'en' or 'ar': {bad}")
        if self.default_language not in self.languages:
            raise ValueError(
                f"default_language '{self.default_language}' is not a key of languages"
            )
        names = [t.name for t in self.webhook_tools]
        if len(names) != len(set(names)):
            raise ValueError("webhook tool names must be unique")
        urls = [d.url for d in self.knowledge_base]
        if len(urls) != len(set(urls)):
            raise ValueError("knowledge base urls must be unique")
        if self.llm is not None and not self.llm.strip():
            raise ValueError("llm must be null or a model id")
        return self

    def placeholder_voices(self) -> list[str]:
        """Language codes whose voice_id is still a placeholder."""
        return [code for code, lang in self.languages.items() if lang.voice_is_placeholder]


@dataclass
class BrandBundle:
    """A validated brand file plus the files it points to."""

    brand: Brand
    path: Path
    prompt_text: str

    @property
    def base_dir(self) -> Path:
        return self.path.parent


def _format_validation(err: ValidationError) -> str:
    lines = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "(root)"
        lines.append(f"  {loc}: {e['msg']}")
    return "\n".join(lines)


def load_brand(path: str | Path) -> BrandBundle:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"brand file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"brand file is not valid JSON: {path}: {exc}") from exc
    try:
        brand = Brand.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"brand file {path} is invalid:\n{_format_validation(exc)}") from exc
    prompt_path = (path.parent / brand.prompt_file).resolve()
    if not prompt_path.is_file():
        raise ConfigError(f"prompt_file not found: {prompt_path}")
    return BrandBundle(brand=brand, path=path, prompt_text=prompt_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests file
# ---------------------------------------------------------------------------


class ChatTurn(_Strict):
    # "user" and "agent" are the role names the Tests API uses for chat_history.
    role: Literal["user", "agent"]
    message: str = Field(min_length=1)


class ToolCallExpectation(_Strict):
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class SimulationSpec(_Strict):
    user_persona: str = Field(min_length=1)
    max_turns: int = Field(default=5, ge=1, le=50)
    success_condition: str | None = None


class TestCase(_Strict):
    __test__ = False  # stop pytest collecting this class

    id: str
    language: str
    dialect: Literal["msa", "gulf"] | None = None
    type: Literal["llm", "tool", "simulation"]
    source: str | None = None
    chat_history: list[ChatTurn] = Field(default_factory=list)
    success_condition: str | None = None
    success_examples: list[str] = Field(default_factory=list)
    failure_examples: list[str] = Field(default_factory=list)
    tool_call: ToolCallExpectation | None = None
    simulation: SimulationSpec | None = None

    @model_validator(mode="after")
    def _shape_matches_type(self) -> TestCase:
        if not TEST_ID.match(self.id):
            raise ValueError(f"test id '{self.id}' must be kebab-case")
        if not self.id.startswith(f"{self.language}-"):
            raise ValueError(
                f"test id '{self.id}' must start with its language, '{self.language}-'"
            )
        if self.type in ("llm", "tool"):
            if not self.chat_history:
                raise ValueError(f"{self.type} test '{self.id}' needs chat_history")
            if self.chat_history[-1].role != "user":
                raise ValueError(f"the last chat_history turn in '{self.id}' must be the user's")
        if self.type == "llm" and not self.success_condition:
            raise ValueError(f"llm test '{self.id}' needs success_condition")
        if self.type == "tool" and self.tool_call is None:
            raise ValueError(f"tool test '{self.id}' needs tool_call")
        if self.type != "tool" and self.tool_call is not None:
            raise ValueError(f"tool_call is only allowed on tool tests ('{self.id}')")
        if self.type == "simulation":
            if self.simulation is None:
                raise ValueError(f"simulation test '{self.id}' needs simulation")
            if not (self.simulation.success_condition or self.success_condition):
                raise ValueError(f"simulation test '{self.id}' needs a success_condition")
        if self.type != "simulation" and self.simulation is not None:
            raise ValueError(f"simulation is only allowed on simulation tests ('{self.id}')")
        return self


class TestSuite(_Strict):
    __test__ = False

    tests: list[TestCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> TestSuite:
        ids = [t.id for t in self.tests]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate test ids: {dupes}")
        return self


def load_tests(path: str | Path, brand: Brand | None = None) -> TestSuite:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"tests file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"tests file is not valid YAML: {path}: {exc}") from exc
    try:
        suite = TestSuite.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"tests file {path} is invalid:\n{_format_validation(exc)}") from exc
    if brand is not None:
        problems = []
        tool_names = {t.name for t in brand.webhook_tools}
        for t in suite.tests:
            if t.language not in brand.languages:
                problems.append(f"  {t.id}: language '{t.language}' is not in the brand file")
            if t.tool_call and t.tool_call.name not in tool_names:
                problems.append(f"  {t.id}: tool '{t.tool_call.name}' is not in the brand file")
        if problems:
            raise ConfigError("tests file does not match the brand file:\n" + "\n".join(problems))
    return suite


# ---------------------------------------------------------------------------
# Rendering and payload building
# ---------------------------------------------------------------------------


def render(template: Any, values: dict[str, Any]) -> Any:
    """Fill ``{{name}}`` placeholders in a JSON-like structure.

    A string that is exactly one placeholder is replaced by the value itself, so
    numbers, booleans, lists and objects keep their type. Placeholders inside a
    longer string are replaced by the value as text. Unknown names raise
    ConfigError. Inserted values are not rendered again, so a prompt may contain
    ElevenLabs dynamic variables such as ``{{system__time}}``.
    """
    if isinstance(template, dict):
        return {k: render(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, values) for v in template]
    if isinstance(template, str):
        whole = _WHOLE_PLACEHOLDER.match(template)
        if whole:
            key = whole.group(1)
            if key not in values:
                raise ConfigError(f"no value for placeholder {{{{{key}}}}}")
            return copy.deepcopy(values[key])

        def _sub(m: re.Match[str]) -> str:
            key = m.group(1)
            if key not in values:
                raise ConfigError(f"no value for placeholder {{{{{key}}}}}")
            return str(values[key])

        return _INLINE_PLACEHOLDER.sub(_sub, template)
    return template


def render_text(text: str, values: dict[str, str]) -> str:
    """Fill only the given placeholders in free text; leave any others alone."""

    def _sub(m: re.Match[str]) -> str:
        return values.get(m.group(1), m.group(0))

    return _INLINE_PLACEHOLDER.sub(_sub, text)


def load_base_config(path: str | Path = DEFAULT_BASE_CONFIG) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"base config not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"base config is not valid JSON: {path}: {exc}") from exc


def build_prompt(bundle: BrandBundle, template_path: str | Path = DEFAULT_PROMPT_TEMPLATE) -> str:
    template_path = Path(template_path)
    if not template_path.is_file():
        raise ConfigError(f"prompt template not found: {template_path}")
    values = {"site_name": bundle.brand.site_name, "agent_name": bundle.brand.agent_name}
    generic = render_text(template_path.read_text(encoding="utf-8"), values).rstrip()
    brand_part = render_text(bundle.prompt_text, values).strip()
    return f"{generic}\n\n{brand_part}\n"


def secret_header_value(token: str) -> str:
    """The full header value stored in the ElevenLabs workspace secret."""
    return f"{AUTH_SCHEME}{token}"


def build_secret_payload(name: str, token: str) -> dict[str, Any]:
    """Body for POST /v1/convai/secrets."""
    return {"type": "new", "name": name, "value": secret_header_value(token)}


def build_secret_update_payload(name: str, token: str) -> dict[str, Any]:
    """Body for PATCH /v1/convai/secrets/{secret_id}."""
    return {"type": "update", "name": name, "value": secret_header_value(token)}


def build_request_body_schema(schema: BodySchema) -> dict[str, Any]:
    """Convert the brand file's small JSON schema to ElevenLabs' request_body_schema."""
    properties: dict[str, Any] = {}
    for name, prop in schema.properties.items():
        item: dict[str, Any] = {"type": prop.type, "description": prop.description}
        if prop.enum is not None:
            item["enum"] = list(prop.enum)
        properties[name] = item
    out: dict[str, Any] = {
        "type": "object",
        "required": list(schema.required),
        "properties": properties,
    }
    if schema.description:
        out["description"] = schema.description
    return out


def build_tool_payload(tool: WebhookTool, secret_id: str | None) -> dict[str, Any]:
    """Body for POST /v1/convai/tools and PATCH /v1/convai/tools/{tool_id}."""
    api_schema: dict[str, Any] = {
        "url": tool.url,
        "method": tool.method,
        "content_type": "application/json",
    }
    if tool.secret_name is not None:
        if not secret_id:
            raise ConfigError(f"tool '{tool.name}' needs a secret id")
        api_schema["request_headers"] = {AUTH_HEADER: {"secret_id": secret_id}}
    if tool.body_schema is not None:
        api_schema["request_body_schema"] = build_request_body_schema(tool.body_schema)
    return {
        "tool_config": {
            "type": "webhook",
            "name": tool.name,
            "description": tool.description,
            "response_timeout_secs": tool.timeout_seconds,
            "api_schema": api_schema,
        }
    }


def build_kb_payload(doc: KnowledgeBaseDoc) -> dict[str, Any]:
    """Body for POST /v1/convai/knowledge-base/url."""
    return {"url": doc.url, "name": doc.name}


def kb_locator(doc_id: str, name: str) -> dict[str, Any]:
    """Entry for conversation_config.agent.prompt.knowledge_base."""
    return {"type": "url", "name": name, "id": doc_id, "usage_mode": "auto"}


def build_language_presets(brand: Brand) -> dict[str, Any]:
    """One preset per non-default language: first message and voice override."""
    presets: dict[str, Any] = {}
    for code, lang in brand.languages.items():
        if code == brand.default_language:
            continue
        presets[code] = {
            "overrides": {
                "agent": {"first_message": lang.first_message},
                "tts": {"voice_id": lang.voice_id},
            }
        }
    return presets


def build_agent_payload(
    bundle: BrandBundle,
    base: dict[str, Any],
    *,
    tool_ids: list[str],
    knowledge_base: list[dict[str, Any]],
    prompt: str,
) -> dict[str, Any]:
    """Body for POST /v1/convai/agents/create and PATCH /v1/convai/agents/{agent_id}."""
    brand = bundle.brand
    default = brand.languages[brand.default_language]
    values: dict[str, Any] = {
        "agent_name": brand.agent_name,
        "site_name": brand.site_name,
        "default_language": brand.default_language,
        "first_message": default.first_message,
        "voice_id": default.voice_id,
        "prompt": prompt,
        "tool_ids": list(tool_ids),
        "knowledge_base": list(knowledge_base),
        "language_presets": build_language_presets(brand),
        "max_duration_seconds": brand.limits.max_duration_seconds,
        "daily_limit": brand.limits.daily_limit,
        "concurrency_limit": brand.limits.concurrency_limit,
        "record_voice": brand.privacy.record_voice,
        "retention_days": brand.privacy.retention_days,
    }
    payload = render(base, values)
    if brand.llm:
        payload["conversation_config"]["agent"]["prompt"]["llm"] = brand.llm
    if brand.privacy.delete_transcript_and_pii is not None:
        payload["platform_settings"].setdefault("privacy", {})["delete_transcript_and_pii"] = (
            brand.privacy.delete_transcript_and_pii
        )
    if not knowledge_base:
        payload["conversation_config"]["agent"]["prompt"]["rag"] = {"enabled": False}
    return payload


# ---------------------------------------------------------------------------
# Tests payloads
# ---------------------------------------------------------------------------


def _expected_as_text(value: Any) -> str:
    """The Tests API compares tool parameters as strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return json.dumps(value, ensure_ascii=False)


def build_parameter_eval(path: str, expected: Any) -> dict[str, Any]:
    """Map one expected tool parameter to a UnitTestToolCallParameter.

    A plain value is an exact match. A mapping with one key picks another check:
    ``{regex: "^9"}``, ``{llm: "a city in the UAE"}`` or ``{anything: true}``.
    """
    if isinstance(expected, dict):
        if len(expected) != 1:
            raise ConfigError(f"parameter '{path}': use one of exact, regex, llm, anything")
        ((kind, arg),) = expected.items()
        if kind == "exact":
            ev: dict[str, Any] = {"type": "exact", "expected_value": _expected_as_text(arg)}
        elif kind == "regex":
            ev = {"type": "regex", "pattern": str(arg)}
        elif kind == "llm":
            ev = {"type": "llm", "description": str(arg)}
        elif kind == "anything":
            ev = {"type": "anything"}
        else:
            raise ConfigError(f"parameter '{path}': unknown check '{kind}'")
    else:
        ev = {"type": "exact", "expected_value": _expected_as_text(expected)}
    return {"path": path, "eval": ev}


def build_test_payload(
    test: TestCase,
    *,
    tool_ids: dict[str, str] | None = None,
    param_path_prefix: str = "",
) -> dict[str, Any]:
    """Body for POST /v1/convai/agent-testing/create and PUT /v1/convai/agent-testing/{id}."""
    payload: dict[str, Any] = {"name": test.id, "type": test.type}
    if test.chat_history:
        payload["chat_history"] = [
            {"role": turn.role, "message": turn.message, "time_in_call_secs": i}
            for i, turn in enumerate(test.chat_history)
        ]
    if test.type == "llm":
        payload["success_condition"] = test.success_condition
        payload["success_examples"] = [
            {"response": r, "type": "success"} for r in test.success_examples
        ]
        payload["failure_examples"] = [
            {"response": r, "type": "failure"} for r in test.failure_examples
        ]
    elif test.type == "tool":
        assert test.tool_call is not None
        tool_id = (tool_ids or {}).get(test.tool_call.name)
        if not tool_id:
            raise ConfigError(
                f"test '{test.id}' references tool '{test.tool_call.name}', which has no id "
                "in the state file. Run setup_agent --apply first."
            )
        payload["tool_call_parameters"] = {
            "referenced_tool": {"id": tool_id, "type": "webhook"},
            "parameters": [
                build_parameter_eval(f"{param_path_prefix}{name}", value)
                for name, value in test.tool_call.parameters.items()
            ],
        }
    else:
        assert test.simulation is not None
        payload["simulation_scenario"] = test.simulation.user_persona
        payload["simulation_max_turns"] = test.simulation.max_turns
        # Prefer the simulation's own condition; fall back to the top-level one.
        # The simulation and tool bodies have no success_examples or
        # failure_examples fields, so those stay in the YAML as documentation.
        payload["success_condition"] = test.simulation.success_condition or test.success_condition
    return payload
