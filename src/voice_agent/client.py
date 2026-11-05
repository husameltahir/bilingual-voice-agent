"""A thin ElevenLabs REST client for the Agents Platform (``/v1/convai``).

Paths and bodies follow the official ``elevenlabs`` Python SDK
(``elevenlabs/conversational_ai/**/raw_client.py``). This client only covers the
calls this project makes.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import httpx

BASE_URL = "https://api.elevenlabs.io"
API_KEY_ENV = "ELEVENLABS_API_KEY"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class ElevenLabsError(Exception):
    """An API call failed. The message never contains the API key."""

    def __init__(self, message: str, *, status_code: int | None = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class NotFoundError(ElevenLabsError):
    """The resource does not exist (HTTP 404)."""


def _detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:500] if text else "(empty body)"
    if isinstance(data, dict) and "detail" in data:
        data = data["detail"]
    text = str(data)
    return text[:500]


class ElevenLabsClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 4,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        if not key.strip():
            raise ElevenLabsError(
                f"{API_KEY_ENV} is not set. Add it to .env (see .env.example) or the environment."
            )
        self._key = key.strip()
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"xi-api-key": self._key, "accept": "application/json"},
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ElevenLabsClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # never show the key
        return f"ElevenLabsClient(base_url={str(self._http.base_url)!r})"

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def _wait(self, attempt: int, response: httpx.Response | None) -> None:
        delay = self._backoff * (2**attempt)
        if response is not None:
            retry_after = response.headers.get("retry-after", "")
            if retry_after.isdigit():
                delay = max(delay, float(retry_after))
        self._sleep(min(delay, 60.0))

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Send a request, retrying on 429, 5xx and network errors."""
        label = f"{method} {path}"
        for attempt in range(self._max_retries + 1):
            try:
                response = self._http.request(method, path, json=json, params=params)
            except httpx.TransportError as exc:
                if attempt < self._max_retries:
                    self._wait(attempt, None)
                    continue
                raise ElevenLabsError(
                    self._redact(f"{label} failed: network error: {exc.__class__.__name__}")
                ) from None
            if response.status_code in RETRY_STATUSES and attempt < self._max_retries:
                self._wait(attempt, response)
                continue
            if response.status_code >= 400:
                detail = self._redact(_detail(response))
                hint = ""
                if response.status_code == 401:
                    hint = f" Check {API_KEY_ENV}."
                elif response.status_code == 422:
                    hint = " The request body was rejected; see the detail above."
                message = f"{label} failed with HTTP {response.status_code}: {detail}.{hint}"
                cls = NotFoundError if response.status_code == 404 else ElevenLabsError
                raise cls(message, status_code=response.status_code, body=detail)
            if not response.content:
                return {}
            try:
                return response.json()
            except ValueError:
                return {"raw": response.text}
        raise ElevenLabsError(
            f"{label} failed after {self._max_retries} retries"
        )  # pragma: no cover

    # -- workspace secrets -------------------------------------------------
    def create_secret(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/v1/convai/secrets", json=payload)

    def update_secret(self, secret_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("PATCH", f"/v1/convai/secrets/{secret_id}", json=payload)

    def delete_secret(self, secret_id: str) -> None:
        self.request("DELETE", f"/v1/convai/secrets/{secret_id}")

    # -- tools ---------------------------------------------------------------
    def create_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/v1/convai/tools", json=payload)

    def update_tool(self, tool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("PATCH", f"/v1/convai/tools/{tool_id}", json=payload)

    def delete_tool(self, tool_id: str) -> None:
        self.request("DELETE", f"/v1/convai/tools/{tool_id}")

    # -- knowledge base ------------------------------------------------------
    def create_kb_from_url(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/v1/convai/knowledge-base/url", json=payload)

    def delete_kb_document(self, doc_id: str) -> None:
        self.request("DELETE", f"/v1/convai/knowledge-base/{doc_id}")

    # -- agents --------------------------------------------------------------
    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/v1/convai/agents/create", json=payload)

    def update_agent(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("PATCH", f"/v1/convai/agents/{agent_id}", json=payload)

    def delete_agent(self, agent_id: str) -> None:
        self.request("DELETE", f"/v1/convai/agents/{agent_id}")

    def get_signed_url(self, agent_id: str) -> str:
        """Signed WebSocket URL for a private agent (valid for 15 minutes to start a call)."""
        data = self.request(
            "GET", "/v1/convai/conversation/get-signed-url", params={"agent_id": agent_id}
        )
        return data["signed_url"]

    # -- tests ---------------------------------------------------------------
    def create_test(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/v1/convai/agent-testing/create", json=payload)

    def update_test(self, test_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("PUT", f"/v1/convai/agent-testing/{test_id}", json=payload)

    def delete_test(self, test_id: str) -> None:
        self.request("DELETE", f"/v1/convai/agent-testing/{test_id}")

    def run_tests(self, agent_id: str, test_ids: list[str]) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/v1/convai/agents/{agent_id}/run-tests",
            json={"tests": [{"test_id": t} for t in test_ids]},
        )

    def get_test_invocation(self, invocation_id: str) -> dict[str, Any]:
        return self.request("GET", f"/v1/convai/test-invocations/{invocation_id}")
