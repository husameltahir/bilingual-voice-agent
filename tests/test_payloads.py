"""Snapshot-style checks on the payloads sent to ElevenLabs."""

from __future__ import annotations

import pytest

from voice_agent.config import (
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


@pytest.fixture
def bundle(example_brand_path):
    return load_brand(example_brand_path)


@pytest.fixture
def agent(bundle):
    return build_agent_payload(
        bundle,
        load_base_config(),
        tool_ids=["tool_1"],
        knowledge_base=[kb_locator("doc_1", "Visa guide")],
        prompt=build_prompt(bundle),
    )


def test_agent_language_and_presets(agent) -> None:
    cc = agent["conversation_config"]
    assert agent["name"] == "Relocation concierge (lab)"
    assert cc["agent"]["language"] == "en"
    assert cc["agent"]["first_message"].startswith("Hello.")
    assert cc["tts"] == {"model_id": "eleven_flash_v2_5", "voice_id": "VOICE_ID_EN"}
    assert cc["asr"] == {"provider": "scribe_realtime"}
    assert set(cc["language_presets"]) == {"ar"}
    ar = cc["language_presets"]["ar"]["overrides"]
    assert ar["agent"]["first_message"].startswith("مرحباً")
    assert ar["tts"] == {"voice_id": "VOICE_ID_AR"}


def test_agent_prompt_tools_and_knowledge_base(agent) -> None:
    prompt = agent["conversation_config"]["agent"]["prompt"]
    assert prompt["llm"] == "claude-haiku-4-5"
    assert prompt["tool_ids"] == ["tool_1"]
    assert prompt["knowledge_base"] == [
        {"type": "url", "name": "Visa guide", "id": "doc_1", "usage_mode": "auto"}
    ]
    assert prompt["rag"] == {"enabled": True}
    built_in = prompt["built_in_tools"]
    assert set(built_in) == {"end_call", "language_detection"}
    assert built_in["end_call"]["params"] == {"system_tool_type": "end_call"}
    assert built_in["language_detection"]["params"] == {"system_tool_type": "language_detection"}
    assert prompt["prompt"].startswith("# Speaking style")
    assert "voice assistant for Northwind Relocation" in prompt["prompt"]
    assert prompt["prompt"].rstrip().endswith("asking for a written quote.")


def test_agent_limits_privacy_auth_overrides(agent) -> None:
    cc = agent["conversation_config"]
    ps = agent["platform_settings"]
    assert cc["conversation"] == {"max_duration_seconds": 300}
    assert ps["call_limits"] == {
        "agent_concurrency_limit": 2,
        "daily_limit": 40,
        "bursting_enabled": False,
    }
    assert ps["privacy"] == {"record_voice": False, "retention_days": 30}
    assert ps["auth"] == {"enable_auth": True}
    assert ps["overrides"] == {
        "conversation_config_override": {"agent": {"language": True, "first_message": True}}
    }
    assert ps["guardrails"]["focus"] == {"is_enabled": True}
    assert ps["guardrails"]["prompt_injection"] == {"is_enabled": True}
    criteria = [c["id"] for c in ps["evaluation"]["criteria"]]
    assert criteria == [
        "answered_from_guides",
        "stayed_in_scope",
        "replied_in_callers_language",
        "no_invented_figures",
        "offered_next_step",
    ]
    assert set(ps["data_collection"]) == {"topic", "language", "used_salary_tool"}


def test_no_placeholders_left(agent) -> None:
    import json

    text = json.dumps(agent["platform_settings"]) + json.dumps(
        {k: v for k, v in agent["conversation_config"].items() if k != "agent"}
    )
    assert "{{" not in text


def test_brand_llm_and_privacy_override(write_brand) -> None:
    def mutate(d):
        d["llm"] = "gemini-2.5-flash"
        d["privacy"]["delete_transcript_and_pii"] = True

    bundle = load_brand(write_brand(mutate))
    payload = build_agent_payload(
        bundle, load_base_config(), tool_ids=[], knowledge_base=[], prompt="p"
    )
    assert payload["conversation_config"]["agent"]["prompt"]["llm"] == "gemini-2.5-flash"
    assert payload["platform_settings"]["privacy"]["delete_transcript_and_pii"] is True
    assert payload["conversation_config"]["agent"]["prompt"]["rag"] == {"enabled": False}


def test_tool_payload_uses_secret_reference(bundle) -> None:
    tool = bundle.brand.webhook_tools[0]
    payload = build_tool_payload(tool, "sec_1")
    cfg = payload["tool_config"]
    assert cfg["type"] == "webhook"
    assert cfg["name"] == "moving_quote"
    assert cfg["response_timeout_secs"] == 10
    api = cfg["api_schema"]
    assert api["url"] == "https://example.com/api/tools/moving-quote"
    assert api["method"] == "POST"
    assert api["request_headers"] == {"Authorization": {"secret_id": "sec_1"}}
    body = api["request_body_schema"]
    assert body["type"] == "object"
    assert body["required"] == ["origin_country", "destination_city", "bedrooms"]
    assert body["properties"]["destination_city"] == {
        "type": "string",
        "description": "UAE city the caller is moving to.",
        "enum": ["dubai", "abu_dhabi", "sharjah"],
    }
    assert body["properties"]["bedrooms"] == {
        "type": "integer",
        "description": "Number of bedrooms of furniture to ship, from 0 to 6.",
    }


def test_tool_payload_needs_secret_id(bundle) -> None:
    with pytest.raises(ConfigError):
        build_tool_payload(bundle.brand.webhook_tools[0], None)


def test_secret_payloads_hold_full_bearer_value() -> None:
    assert build_secret_payload("s", "abc") == {"type": "new", "name": "s", "value": "Bearer abc"}
    assert build_secret_update_payload("s", "abc") == {
        "type": "update",
        "name": "s",
        "value": "Bearer abc",
    }


def test_kb_payload(bundle) -> None:
    assert build_kb_payload(bundle.brand.knowledge_base[0]) == {
        "url": "https://example.com/guides/visas.md",
        "name": "Visa guide",
    }
