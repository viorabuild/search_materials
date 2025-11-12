"""ProjectChatAgent - вспомогательный чат для обсуждения репозитория.

Позволяет вести диалог с LLM об устройстве проекта, опираясь на
предзагруженный контекст (README, документация) и историю беседы."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from openai import OpenAI

try:
    from openai import OpenAIError
except ImportError:  # pragma: no cover
    OpenAIError = Exception  # type: ignore

try:
    from openai import RateLimitError
except ImportError:  # pragma: no cover
    RateLimitError = ()  # type: ignore


logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "Ты - дружелюбный ассистент, помогающий разработчику ориентироваться в "
    "проекте. Объясняй устройство репозитория, его архитектуру и файлы. "
    "При необходимости ссылайся на конкретные файлы и подсказывай, где искать "
    "подробности. Не придумывай несуществующие файлы."
)


def _safe_read(path: Path, max_chars: int = 4000) -> Optional[str]:
    """Прочитать файл, ограничив размер, чтобы избежать перегрузки контекста."""
    try:
        text = path.read_text(encoding="utf-8")
        if len(text) > max_chars:
            return text[:max_chars] + "\n..."
        return text
    except Exception:
        return None


@dataclass
class ProjectContext:
    """Контекст проекта, который добавляется в системные сообщения."""

    summary: str


class ProjectChatError(RuntimeError):
    """Raised when LLM-backed project chat fails to respond."""

    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProjectChatAgent:
    """LLM-чат для обсуждения проекта."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        project_root: Path,
        context_files: Optional[Sequence[Path]] = None,
        max_history_turns: int = 6,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        request_timeout: Optional[float] = None,
    ) -> None:
        self._client = client
        self._model = model
        self._project_root = Path(project_root)
        self._max_history_turns = max(1, max_history_turns)
        self._system_prompt = system_prompt
        self._history: List[dict] = []
        self._context = self._build_context(context_files)
        self._request_timeout = request_timeout

    def chat(self, message: str, extra_context: Optional[str] = None) -> str:
        """Отправить сообщение в чат и получить ответ."""
        if not message or not message.strip():
            raise ValueError("message must be a non-empty string")

        messages = [{"role": "system", "content": self._system_prompt}]

        if self._context.summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Контекст проекта:\n{self._context.summary}",
                }
            )

        if extra_context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Дополнительный контекст:\n{extra_context}",
                }
            )

        # История (ограниченное число последних turn'ов)
        history_limit = self._max_history_turns * 2
        if len(self._history) > history_limit:
            self._history = self._history[-history_limit:]
        messages.extend(self._history)

        user_message = message.strip()
        messages.append({"role": "user", "content": user_message})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.3,
                timeout=self._request_timeout,
            )
        except OpenAIError as exc:  # pragma: no cover - network/UI layer
            status_code = getattr(exc, "status_code", None)
            if status_code is None:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 429 or (
                RateLimitError and isinstance(exc, RateLimitError)
            ):
                msg = "Лимит запросов к LLM исчерпан. Попробуйте снова через минуту."
                raise ProjectChatError(msg, status_code=429) from exc
            logger.error("Project chat OpenAI error: %s", exc)
            raise ProjectChatError("LLM сервис временно недоступен.", status_code=503) from exc
        except Exception as exc:  # pragma: no cover - unexpected runtime
            logger.exception("Unexpected project chat failure")
            raise ProjectChatError("Неожиданная ошибка при обращении к проектному чату.") from exc
        answer = (
            response.choices[0].message.content.strip()
            if response.choices
            else "Не удалось получить ответ от модели."
        )

        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": answer})

        return answer

    def reset_history(self) -> None:
        """Сбросить историю диалога."""
        self._history.clear()

    def _build_context(self, context_files: Optional[Sequence[Path]]) -> ProjectContext:
        """Сформировать текстовый контекст проекта."""
        files = list(context_files or [])
        if not files:
            files = [
                self._project_root / "README.md",
                self._project_root / "docs" / "architecture.md",
            ]

        snippets: List[str] = []
        for path in files:
            resolved = path if path.is_absolute() else self._project_root / path
            content = _safe_read(resolved)
            if content:
                try:
                    relative = resolved.relative_to(self._project_root)
                except ValueError:
                    relative = resolved.name
                snippets.append(f"# {relative}\n{content}")

        if not snippets:
            overview = self._build_overview()
            return ProjectContext(summary=overview)

        overview = self._build_overview()
        combined = overview + ("\n\n" + "\n\n".join(snippets) if snippets else "")
        return ProjectContext(summary=combined)

    def _build_overview(self) -> str:
        """Сформировать краткий список ключевых файлов для подсказок."""
        parts: List[str] = []

        top_level_py = sorted(
            p for p in self._project_root.glob("*.py") if p.is_file()
        )
        if top_level_py:
            items = ", ".join(p.name for p in top_level_py[:12])
            parts.append(f"Топ-уровень .py файлов: {items}")

        docs_dir = self._project_root / "docs"
        if docs_dir.exists():
            docs = sorted(docs_dir.glob("*.md"))
            if docs:
                items = ", ".join(doc.name for doc in docs[:12])
                parts.append(f"Документация /docs: {items}")

        frontend_dir = self._project_root / "frontend"
        if frontend_dir.exists():
            html_files = sorted(frontend_dir.glob("*.html"))
            if html_files:
                items = ", ".join(f.name for f in html_files[:8])
                parts.append(f"Фронтенд /frontend: {items}")

        return "\n".join(parts)
