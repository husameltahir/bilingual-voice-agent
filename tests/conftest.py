from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from voice_agent.client import BASE_URL, ElevenLabsClient

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
TEST_KEY = "xi-test-key-123"
TOOL_SECRET = "tool-secret-value-456"


@pytest.fixture
def example_brand_path() -> Path:
    return EXAMPLES / "brand.example.json"


@pytest.fixture
def example_tests_path() -> Path:
    return EXAMPLES / "tests.example.yaml"


@pytest.fixture
def brand_dir(tmp_path: Path) -> Path:
    """A copy of the example brand with real-looking voice IDs."""
    shutil.copy(EXAMPLES / "prompt.example.md", tmp_path / "prompt.example.md")
    data = json.loads((EXAMPLES / "brand.example.json").read_text(encoding="utf-8"))
    data["languages"]["en"]["voice_id"] = "voiceEn123"
    data["languages"]["ar"]["voice_id"] = "voiceAr456"
    (tmp_path / "brand.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    shutil.copy(EXAMPLES / "tests.example.yaml", tmp_path / "tests.yaml")
    return tmp_path


@pytest.fixture
def write_brand(tmp_path: Path):
    """Write a modified copy of the example brand and return its path."""

    def _write(mutate) -> Path:
        shutil.copy(EXAMPLES / "prompt.example.md", tmp_path / "prompt.example.md")
        data: dict[str, Any] = json.loads(
            (EXAMPLES / "brand.example.json").read_text(encoding="utf-8")
        )
        mutate(data)
        path = tmp_path / "brand.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def client_factory():
    def _make() -> ElevenLabsClient:
        return ElevenLabsClient(api_key=TEST_KEY, base_url=BASE_URL, sleep=lambda _s: None)

    return _make
