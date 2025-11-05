"""Utilities for working with LLM providers that support OpenAI-compatible APIs.

This module exposes a thin wrapper around the OpenAI Python client so the
application can automatically fall back to a local LLM served via LM Studio
whenever the primary OpenAI API key is unavailable or returns an error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


def _normalize_base_url(base_url: str) -> str:
    """Ensure the provided base URL ends with /v1 as expected by OpenAI client."""
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return normalized
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


@dataclass
class LLMFallbackConfig:
    """Configuration for building a fallback-enabled OpenAI client."""

    primary_api_key: Optional[str]
    primary_model: Optional[str]
    request_timeout: Optional[float] = None
    local_enabled: bool = False
    local_base_url: Optional[str] = None
    local_model: Optional[str] = None
    local_api_key: str = "lm-studio"


class _ChatCompletionsAdapter:
    """Adapter that forwards chat completion calls with fallback handling."""

    def __init__(self, owner: "FallbackOpenAI") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner._call_with_fallback(
            lambda client, payload: client.chat.completions.create(**payload),
            kwargs,
        )


class _ChatAdapter:
    """Adapter that mimics the structure of the OpenAI client."""

    def __init__(self, owner: "FallbackOpenAI") -> None:
        self.completions = _ChatCompletionsAdapter(owner)


class _ResponsesAdapter:
    """Adapter for the Responses API used by advanced workflows."""

    def __init__(self, owner: "FallbackOpenAI") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner._call_with_fallback(
            lambda client, payload: client.responses.create(**payload),
            kwargs,
        )


class FallbackOpenAI:
    """OpenAI-compatible client that automatically falls back to a local LLM."""

    def __init__(self, config: LLMFallbackConfig) -> None:
        if not config.primary_api_key and not config.local_enabled:
            raise ValueError("Either primary OpenAI credentials or local LLM must be configured.")

        self._primary_model = config.primary_model
        self._fallback_model = config.local_model or config.primary_model
        self._primary_client: Optional[OpenAI] = None
        self._fallback_client: Optional[OpenAI] = None

        if config.primary_api_key:
            self._primary_client = OpenAI(
                api_key=config.primary_api_key,
                timeout=config.request_timeout,
            )

        if config.local_enabled and config.local_base_url:
            base_url = _normalize_base_url(config.local_base_url)
            self._fallback_client = OpenAI(
                api_key=config.local_api_key or "lm-studio",
                base_url=base_url,
                timeout=config.request_timeout,
            )

        if not self._primary_client and not self._fallback_client:
            raise ValueError("Failed to initialize OpenAI clients for both primary and local providers.")

        self.chat = _ChatAdapter(self)
        self.responses = _ResponsesAdapter(self)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def primary_model(self) -> Optional[str]:
        return self._primary_model

    @property
    def fallback_model(self) -> Optional[str]:
        return self._fallback_model

    @property
    def has_primary(self) -> bool:
        return self._primary_client is not None

    @property
    def has_fallback(self) -> bool:
        return self._fallback_client is not None

    def _call_with_fallback(
        self,
        caller: Callable[[OpenAI, dict], Any],
        payload: dict,
    ) -> Any:
        """
        Invoke the provided callable against the primary client first, falling
        back to the local client if a recoverable error occurs or the primary
        is not configured.
        """
        last_error: Optional[Exception] = None
        if self._primary_client is not None:
            try:
                return caller(self._primary_client, dict(payload))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Primary OpenAI call failed (%s). Trying local fallback if available.",
                    exc.__class__.__name__,
                )

        if self._fallback_client is not None:
            fallback_payload = dict(payload)
            if not fallback_payload.get("model"):
                fallback_payload["model"] = self._fallback_model
            elif (
                self._primary_model
                and self._fallback_model
                and fallback_payload["model"] == self._primary_model
            ):
                fallback_payload["model"] = self._fallback_model

            try:
                return caller(self._fallback_client, fallback_payload)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Local LLM call failed (%s).",
                    exc.__class__.__name__,
                )
                if last_error is not None:
                    raise last_error
                raise

        if last_error is not None:
            raise last_error

        raise RuntimeError("No LLM clients available for request.")

