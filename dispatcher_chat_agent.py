"""DispatcherChatAgent — общий чат с главным распределителем.

Используется для разговоров с универсальным ИИ, который подсказывает,
какой режим или инструмент включить в Construction AI Agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from llm_provider import FallbackOpenAI

logger = logging.getLogger(__name__)


DEFAULT_DISPATCHER_PROMPT = (
    "Ты — главный распределитель Construction AI Agent. "
    "Помогаешь пользователю понять, какой режим выбрать: универсальные команды, "
    "поиск материалов, Google Sheets, проверка/импорт/форматирование смет, база материалов, пакетный просчет. "
    "Отвечай кратко и структурированно на русском, предлагай следующий шаг списком. "
    "Не выдумывай функций, которых нет, если чего-то не хватает — честно скажи и предложи обход. "
    "Если для ответа нужны уточнения (файл, лист, материал) — запроси их."
)


@dataclass
class DispatcherMessage:
    """Сообщение в истории чата главного распределителя."""

    role: str
    content: str
    timestamp: str


class DispatcherChatAgent:
    """Чат с главным распределителем (общий ИИ)."""

    def __init__(
        self,
        client: FallbackOpenAI,
        preferred_model: Optional[str] = "qwen/qwen3-vl-4b",
        fallback_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_history_turns: int = 8,
        request_timeout: Optional[float] = None,
        temperature: float = 0.35,
    ) -> None:
        self._client = client
        self._preferred_model = preferred_model or "qwen/qwen3-vl-4b"
        self._fallback_model = fallback_model
        self._system_prompt = system_prompt or DEFAULT_DISPATCHER_PROMPT
        self._max_history_turns = max(1, max_history_turns)
        self._request_timeout = request_timeout
        self._temperature = temperature
        self._history: List[DispatcherMessage] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chat(self, message: str, *, reset_history: bool = False) -> str:
        """Отправить сообщение в чат и получить ответ."""
        cleaned = (message or "").strip()
        if reset_history:
            self.reset_history()
        if not cleaned:
            raise ValueError("message must be a non-empty string")

        prompt_messages = self._build_messages(cleaned)
        reply = self._generate_reply(prompt_messages)

        now = datetime.utcnow().isoformat()
        self._history.append(DispatcherMessage(role="user", content=cleaned, timestamp=now))
        self._history.append(
            DispatcherMessage(role="assistant", content=reply, timestamp=datetime.utcnow().isoformat())
        )
        self._trim_history()
        return reply

    def reset_history(self) -> None:
        """Очистить историю диалога."""
        self._history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Вернуть историю в сериализуемом виде."""
        return [msg.__dict__.copy() for msg in self._history]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _trim_history(self) -> None:
        limit = self._max_history_turns * 2
        if len(self._history) > limit:
            self._history = self._history[-limit:]

    def _build_messages(self, user_message: str) -> Sequence[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": self._system_prompt}]
        history_limit = self._max_history_turns * 2
        for msg in self._history[-history_limit:]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _iter_models(self):
        seen = set()
        for name in [
            self._preferred_model,
            self._fallback_model,
            getattr(self._client, "fallback_model", None),
            getattr(self._client, "primary_model", None),
        ]:
            if name and name not in seen:
                seen.add(name)
                yield name

    def _generate_reply(self, messages: Sequence[Dict[str, str]]) -> str:
        errors: List[str] = []
        for model_name in self._iter_models():
            try:
                completion = self._client.chat.completions.create(
                    model=model_name,
                    messages=list(messages),
                    temperature=self._temperature,
                    timeout=self._request_timeout,
                )
                content = completion.choices[0].message.content if completion.choices else ""
                if content:
                    return content.strip()
            except Exception as exc:  # noqa: BLE001
                error_text = str(exc)
                errors.append(error_text)
                logger.warning("Dispatcher chat failed with model %s: %s", model_name, error_text)
                continue

        last_error = errors[-1] if errors else "LLM unavailable for dispatcher chat"
        raise RuntimeError(last_error)
