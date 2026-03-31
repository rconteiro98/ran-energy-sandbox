"""Minimal local LLM client for optional plain-English summaries.

The client assumes an Ollama-compatible local HTTP endpoint. The dashboard
must continue to work even when the endpoint or model is unavailable.
"""

from __future__ import annotations

import json
from urllib import error, request

DEFAULT_LOCAL_LLM_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_LOCAL_LLM_MODEL = "llama3.1:8b"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 20


class LocalLlmError(RuntimeError):
    """Raised when the local LLM endpoint cannot return a usable response."""


def list_local_llm_models(
    endpoint_url: str = DEFAULT_LOCAL_LLM_ENDPOINT,
    timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> list[str]:
    """Return the installed local model tags from an Ollama-compatible endpoint."""

    normalized_endpoint = endpoint_url.rstrip("/")
    http_request = request.Request(
        url=f"{normalized_endpoint}/api/tags",
        headers={"Content-Type": "application/json"},
        method="GET",
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raise LocalLlmError(
            f"Local LLM model lookup failed with HTTP {exc.code}."
        ) from exc
    except error.URLError as exc:
        raise LocalLlmError(
            "Local LLM endpoint is unavailable. Start the local runtime and try again."
        ) from exc
    except TimeoutError as exc:
        raise LocalLlmError("Local LLM model lookup timed out.") from exc

    try:
        parsed_response = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise LocalLlmError("Local LLM model list was not valid JSON.") from exc

    models = parsed_response.get("models", [])
    return [
        str(model_row.get("name", "")).strip()
        for model_row in models
        if str(model_row.get("name", "")).strip()
    ]


def generate_local_llm_text(
    prompt: str,
    model_name: str = DEFAULT_LOCAL_LLM_MODEL,
    endpoint_url: str = DEFAULT_LOCAL_LLM_ENDPOINT,
    timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> str:
    """Send a short prompt to the local LLM and return the generated text."""

    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }
    ).encode("utf-8")

    normalized_endpoint = endpoint_url.rstrip("/")
    http_request = request.Request(
        url=f"{normalized_endpoint}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raise LocalLlmError(
            f"Local LLM request failed with HTTP {exc.code}. Check the endpoint and model tag."
        ) from exc
    except error.URLError as exc:
        raise LocalLlmError(
            "Local LLM endpoint is unavailable. Start the local runtime and try again."
        ) from exc
    except TimeoutError as exc:
        raise LocalLlmError("Local LLM request timed out.") from exc

    try:
        parsed_response = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise LocalLlmError("Local LLM response was not valid JSON.") from exc

    generated_text = str(parsed_response.get("response", "")).strip()
    if not generated_text:
        raise LocalLlmError("Local LLM returned an empty response.")

    return generated_text
