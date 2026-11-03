"""OpenAI-совместимый LLM-клиент с ретраями и rate limiter."""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import requests

from app.config import ProviderConfig


class LLMError(RuntimeError):
    """Ошибка обращения к LLM (сеть, статус, таймаут, нет ключа)."""

    def __init__(self, message: str, retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    """Потокобезопасный лимитер: не больше `rpm` запросов в минуту равномерно."""

    def __init__(self, rpm: int = 0):
        self._interval = (60.0 / rpm) if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                delay = self._next_at - now
                self._next_at += self._interval
            else:
                delay = 0.0
                self._next_at = now + self._interval
        if delay > 0:
            time.sleep(delay)


class LLMProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        *,
        stage: str,
        json_mode: bool = False,
    ) -> str:
        ...


class OpenAICompatProvider(LLMProvider):
    """Реальный OpenAI-совместимый клиент через requests."""

    def __init__(self, cfg: ProviderConfig, limiter: RateLimiter | None = None):
        self._cfg = cfg
        self._limiter = limiter or RateLimiter(cfg.rpm)

    def chat(
        self,
        messages: list[dict],
        *,
        stage: str,
        json_mode: bool = False,
    ) -> str:
        if not self._cfg.api_key:
            raise LLMError(
                "LLM_API_KEY не задан. Настройте .env (см. .env.example) "
                "или запустите сервис с ключом провайдера."
            )
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "temperature": self._cfg.temperature,
        }
        if json_mode and self._cfg.json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        self._limiter.acquire()
        for attempt in range(self._cfg.max_retries):
            try:
                resp = requests.post(
                    f"{self._cfg.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._cfg.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self._cfg.timeout,
                )
                if resp.status_code == 429:
                    retry_after = _parse_retry_after(resp)
                    raise LLMError(
                        f"HTTP 429: {resp.text[:200]}", retry_after=retry_after
                    )
                if resp.status_code >= 500:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except LLMError as e:
                last_err = e
                if e.retry_after > 0:
                    time.sleep(e.retry_after)
                else:
                    time.sleep(min(2 ** attempt, 8))
            except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
                raise LLMError(f"LLM request failed: {e}") from e
        raise LLMError(
            f"LLM failed after {self._cfg.max_retries} attempts: {last_err}"
        )


def _parse_retry_after(resp: requests.Response) -> float:
    """Читает паузу при 429 из Retry-After / x-ratelimit-reset-requests."""
    ra = resp.headers.get("Retry-After")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    reset = resp.headers.get("x-ratelimit-reset-requests")
    if reset:
        try:
            return float(reset)
        except ValueError:
            pass
    return 0.0
