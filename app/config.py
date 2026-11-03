"""Конфигурация провайдера LLM из .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProviderConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_retries: int = 3
    timeout: float = 60.0
    json_mode: bool = True
    rpm: int = 0

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "ProviderConfig":
        if dotenv_path is None:
            dotenv_path = Path(__file__).resolve().parent.parent / ".env"
        env: dict[str, str] = {}
        if dotenv_path and dotenv_path.exists():
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")

        def get(key: str, default: str | None = None) -> str | None:
            return os.environ.get(key, env.get(key, default))

        base_url = get("LLM_BASE_URL", "https://api.openai.com/v1")
        api_key = get("LLM_API_KEY") or None
        model = get("LLM_MODEL", "gpt-4o-mini")
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=float(get("LLM_TEMPERATURE", "0.2")),
            max_retries=int(get("LLM_MAX_RETRIES", "3")),
            timeout=float(get("LLM_TIMEOUT", "60")),
            json_mode=get("LLM_JSON_MODE", "true").lower() not in ("0", "false", "off", "no"),
            rpm=int(get("LLM_RPM", "0") or 0),
        )
