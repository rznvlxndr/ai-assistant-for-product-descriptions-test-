"""Pydantic-модели входа и вывода."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------- ВХОД ----------

class Modification(BaseModel):
    """Одна модификация в составе товара-семейства (из реального каталога)."""

    code: str | None = None
    title: str | None = None
    fields: dict[str, str] = Field(default_factory=dict)


class ProductInput(BaseModel):
    """Входные данные о товаре. Все поля опциональны — система деградирует
    на любом подмножестве (это ядро устойчивости)."""

    name: str | None = None
    manufacturer: str | None = None
    specs: dict[str, Any] | str | None = None
    description: str | None = None

    @field_validator("specs")
    @classmethod
    def _normalize_specs(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, str):
            return {"_raw_string": v}
        if isinstance(v, dict):
            return v
        return {"_raw": str(v)}


# ---------- ВЫХОД ----------

class SpecFact(BaseModel):
    """Одна характеристика с указанием источника (откуда взято значение)."""

    key: str
    value: str
    source: Literal["specs", "description", "decoded", "other"] = "specs"


class ProductOutput(BaseModel):
    """Полный результат — каждый блок ТЗ видимым полем."""

    name: str = ""
    manufacturer: str | None = None
    title_seo: str = ""
    description_seo: str = ""
    specs: list[SpecFact] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    commercial_block: str = ""
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
