from __future__ import annotations

from pathlib import Path

import pytest

from voice_agent.config import (
    ConfigError,
    load_brand,
    load_tests,
    render,
    render_text,
)


def test_example_brand_loads(example_brand_path: Path) -> None:
    bundle = load_brand(example_brand_path)
    brand = bundle.brand
    assert brand.site_name == "Northwind Relocation"
    assert brand.default_language == "en"
    assert set(brand.languages) == {"en", "ar"}
    assert len(brand.knowledge_base) == 3
    assert all(d.url.startswith("https://example.com/") for d in brand.knowledge_base)
    assert brand.placeholder_voices() == ["en", "ar"]
    assert "Northwind" in bundle.prompt_text


def test_comment_keys_are_ignored(write_brand) -> None:
    path = write_brand(lambda d: d.update({"_note": "hello", "$comment": "x"}))
    assert load_brand(path).brand.agent_name


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(default_language="fr"), "default_language"),
        (lambda d: d.update(unexpected=True), "unexpected"),
        (lambda d: d["knowledge_base"][0].update(url="ftp://example.com/x"), "https://"),
        (lambda d: d["knowledge_base"].append(dict(d["knowledge_base"][0])), "unique"),
        (lambda d: d["webhook_tools"][0].update(url="http://example.com/x"), "https://"),
        (lambda d: d["webhook_tools"][0].pop("secret_name"), "secret_env and secret_name"),
        (lambda d: d["webhook_tools"][0].update(name="bad name!"), "tool name"),
        (
            lambda d: d["webhook_tools"][0]["body_schema"]["required"].append("missing"),
            "not defined",
        ),
        (
            lambda d: d["webhook_tools"][0]["body_schema"]["properties"]["bedrooms"].update(
                enum=["1"]
            ),
            "enum",
        ),
        (lambda d: d["limits"].update(daily_limit=0), "daily_limit"),
        (lambda d: d["privacy"].pop("record_voice"), "record_voice"),
        (lambda d: d["languages"].update(EN={"first_message": "x", "voice_id": "v"}), "codes"),
    ],
)
def test_bad_brand_is_rejected(write_brand, mutate, message: str) -> None:
    path = write_brand(mutate)
    with pytest.raises(ConfigError) as exc:
        load_brand(path)
    assert message in str(exc.value)


def test_missing_prompt_file_is_reported(write_brand) -> None:
    path = write_brand(lambda d: d.update(prompt_file="nope.md"))
    with pytest.raises(ConfigError, match="prompt_file not found"):
        load_brand(path)


def test_invalid_json_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "brand.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_brand(path)


def test_example_tests_load_and_match_brand(example_brand_path, example_tests_path) -> None:
    brand = load_brand(example_brand_path).brand
    suite = load_tests(example_tests_path, brand)
    assert len(suite.tests) == 6
    assert {t.type for t in suite.tests} == {"llm", "tool", "simulation"}
    assert {t.language for t in suite.tests} == {"en", "ar"}
    tool_test = next(t for t in suite.tests if t.id == "en-tool-moving-quote")
    assert [turn.role for turn in tool_test.chat_history] == ["user", "agent", "user"]


def _write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "tests.yaml"
    path.write_text(text, encoding="utf-8")
    return path


GOOD_LLM = """
  - id: en-one
    language: en
    type: llm
    chat_history: [{role: user, message: "Hi"}]
    success_condition: "Says hello."
"""


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("tests: []", "tests"),
        ("tests:" + GOOD_LLM + GOOD_LLM, "duplicate"),
        (
            "tests:\n  - {id: ar-x, language: en, type: llm, success_condition: x,"
            " chat_history: [{role: user, message: hi}]}",
            "must start with its language",
        ),
        (
            "tests:\n  - {id: en-x, language: en, type: llm,"
            " chat_history: [{role: user, message: hi}]}",
            "success_condition",
        ),
        (
            "tests:\n  - {id: en-x, language: en, type: llm, success_condition: x,"
            " chat_history: [{role: user, message: hi}, {role: agent, message: yo}]}",
            "must be the user's",
        ),
        (
            "tests:\n  - {id: en-x, language: en, type: tool,"
            " chat_history: [{role: user, message: hi}]}",
            "needs tool_call",
        ),
        (
            "tests:\n  - {id: en-x, language: en, type: simulation,"
            " simulation: {user_persona: someone, max_turns: 3}}",
            "needs a success_condition",
        ),
        (
            "tests:\n  - {id: en-x, language: en, type: llm, success_condition: x,"
            " chat_history: [{role: system, message: hi}]}",
            "role",
        ),
        ("tests:\n  - {id: en-x, language: en, type: voice}", "type"),
    ],
)
def test_bad_tests_are_rejected(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ConfigError) as exc:
        load_tests(_write_yaml(tmp_path, body))
    assert message in str(exc.value)


def test_simulation_accepts_top_level_success_condition(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        "tests:\n  - {id: en-x, language: en, type: simulation, success_condition: top,"
        " success_examples: [a], failure_examples: [b],"
        " simulation: {user_persona: someone, max_turns: 3}}",
    )
    assert load_tests(path).tests[0].success_condition == "top"


def test_tests_are_checked_against_brand(tmp_path: Path, example_brand_path: Path) -> None:
    brand = load_brand(example_brand_path).brand
    path = _write_yaml(
        tmp_path,
        "tests:\n  - {id: fr-x, language: fr, type: tool,"
        " chat_history: [{role: user, message: hi}],"
        " tool_call: {name: unknown_tool, parameters: {a: 1}}}",
    )
    with pytest.raises(ConfigError) as exc:
        load_tests(path, brand)
    assert "language 'fr'" in str(exc.value)
    assert "tool 'unknown_tool'" in str(exc.value)


def test_render_keeps_types_and_fills_inline() -> None:
    template = {"a": "{{n}}", "b": ["{{flag}}"], "c": "Hello {{name}}!", "d": "{{obj}}"}
    out = render(template, {"n": 5, "flag": False, "name": "Ada", "obj": {"x": [1]}})
    assert out == {"a": 5, "b": [False], "c": "Hello Ada!", "d": {"x": [1]}}


def test_render_rejects_unknown_placeholder() -> None:
    with pytest.raises(ConfigError, match="missing"):
        render({"a": "{{missing}}"}, {})


def test_render_text_leaves_dynamic_variables() -> None:
    assert render_text("{{site_name}} at {{system__time}}", {"site_name": "N"}) == (
        "N at {{system__time}}"
    )
