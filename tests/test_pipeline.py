"""Тесты пайплайна на фикстурах (без сети и ключа, LLM = test double)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import ProviderConfig
from app.models import ProductInput
from app.pipeline import TITLE_MAX, DESC_MIN, DESC_MAX, run_pipeline
from app.providers import LLMError, OpenAICompatProvider

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


class FakeLLMProvider:
    """Заглушка LLM с записанными ответами для запуска тестов без сети."""

    def __init__(self, fixture_key: str):
        self._key = fixture_key

    def chat(self, messages: list[dict], *, stage: str, json_mode: bool = False) -> str:
        fixture = _FAKE_RESPONSES.get(self._key, {})
        if stage not in fixture:
            stage = "generate"
        return fixture.get(stage, _FAKE_RESPONSES["default"]["generate"])


def _fake_extract(facts: list[tuple[str, str]]) -> str:
    return json.dumps({"facts": [{"key": k, "value": v} for k, v in facts], "other": []}, ensure_ascii=False)


_FAKE_RESPONSES: dict[str, dict[str, str]] = {
    "thermometer_specs_string": {
        "extract_string": _fake_extract([
            ("диапазон измерений", "-40...+600 °C"),
            ("класс точности", "2"),
            ("длина монтажной части", "250 мм"),
            ("материал защитной арматуры", "нержавеющая сталь 12Х18Н10Т"),
        ]),
    },
    "default": {
        "generate": json.dumps({
            "title": "Контрольно-измерительный прибор",
            "description": (
                "Прибор для измерения и контроля параметров технологических процессов. "
                "Технические характеристики предоставлены не полностью, рекомендуем "
                "обратиться к документации производителя или уточнить у менеджера. "
                "Прибор применяется в системах автоматизации и контроля. "
                "Для выбора конкретной модели и исполнения обратитесь к спецификации "
                "производителя. Подробное описание будет дополнено после уточнения "
                "параметров и условий эксплуатации."
            ),
            "commercial": "Прибор подходит для задач контроля и измерения параметров. Уточните у менеджера область применения.",
            "needs_clarification": ["модель", "характеристики", "применение"],
        }, ensure_ascii=False),
    },
}


def _run(stem: str, data: dict):
    product = ProductInput(**data)
    return run_pipeline(product, FakeLLMProvider(stem))


def _load_all_fixtures() -> list[tuple[str, dict]]:
    result = []
    for p in sorted(FIXTURES_DIR.glob("*.json")):
        result.append((p.stem, json.loads(p.read_text(encoding="utf-8"))))
    return result


@pytest.fixture(scope="module", params=_load_all_fixtures())
def fixture_case(request):
    return request.param


def test_does_not_crash_on_any_fixture(fixture_case):
    stem, data = fixture_case
    out = _run(stem, data)
    assert out is not None


def test_seo_lengths(fixture_case):
    stem, data = fixture_case
    out = _run(stem, data)
    assert len(out.title_seo) <= TITLE_MAX, f"title too long: {out.title_seo}"
    assert DESC_MIN <= len(out.description_seo) <= DESC_MAX, (
        f"description length {len(out.description_seo)} out of range"
    )


def test_queries_between_3_and_5(fixture_case):
    stem, data = fixture_case
    out = _run(stem, data)
    assert 3 <= len(out.queries) <= 5, f"queries count {len(out.queries)}"


def test_facts_from_input_are_preserved(fixture_case):
    """Каждый ключ specs из входа присутствует в выводе (verbatim, полнота)."""
    from app.pipeline import _normalize_key

    stem, data = fixture_case
    specs = data.get("specs")
    if not isinstance(specs, dict):
        pytest.skip("specs не dict (строка/отсутствует) — проверяется в другом тесте")
    if any(k.lower() in ("модификации", "варианты", "исполнения", "variants", "modifications") for k in specs):
        pytest.skip("товар-семейство — проверяется в test_modifications_not_lost")
    out = _run(stem, data)
    out_keys = {f.key.lower() for f in out.specs}
    for k in specs:
        if k.startswith("_"):
            continue
        assert _normalize_key(k).lower() in out_keys, f"ключ '{k}' из входа не найден в выводе (specs={out.specs})"


def test_no_trade_fields_in_specs(fixture_case):
    """Цена/наличие/срок поставки не должны быть отдельными характеристиками."""
    stem, data = fixture_case
    out = _run(stem, data)
    for f in out.specs:
        kl = f.key.lower()
        vl = f.value.lower()
        assert not any(t in kl for t in ("цена", "цена с ндс")), f"торговое поле в specs: {f.key}"
        assert not (kl == "наличие" and "в наличии" in vl), f"торговое наличие в specs: {f.key}"
        if "₽" in vl:
            assert "[торговое:" in vl, f"цена вне торгового блока: {f.key}: {f.value}"


def test_code_only_fixture_degrades_gracefully():
    """Минимальный вход (только код в name) не падает и помечает нехватку данных."""
    data = json.loads((FIXTURES_DIR / "pressure_controller_code_only.json").read_text())
    out = _run("pressure_controller_code_only", data)
    assert out.title_seo
    assert out.description_seo
    assert any("расшифров" in w or "данных нет" in w for w in out.warnings)


def test_contradiction_is_flagged_not_silently_fixed():
    """Противоречие specs (0–20 м) vs description (0–10 м) должно флагаться."""
    data = json.loads((FIXTURES_DIR / "levelmeter_contradiction.json").read_text())
    out = _run("levelmeter_contradiction", data)
    spec_values = [f.value for f in out.specs if "диапазон" in f.key.lower() and f.source == "specs"]
    assert any("20" in v for v in spec_values), f"0-20 м из specs не сохранено: {spec_values}"
    desc_values = [f.value for f in out.specs if "диапазон" in f.key.lower() and f.source == "description"]
    assert any("0-10" in v or "0–10" in v for v in desc_values), (
        f"0-10 м из описания потеряно: {desc_values}"
    )
    assert any("противореч" in w.lower() for w in out.warnings), f"нет предупреждения о противоречии: {out.warnings}"


def test_modifications_not_lost():
    """Товар-семейство: все модификации из входа присутствуют в выводе."""
    data = json.loads((FIXTURES_DIR / "trm1_catalog_slice.json").read_text())
    out = _run("trm1_catalog_slice", data)
    mods = [f for f in out.specs if f.key.startswith("модификация")]
    assert len(mods) == 3, f"ожидалось 3 модификации, получено {len(mods)}"
    for code in ("87780", "125242", "125174"):
        assert any(code in m.key for m in mods), f"модификация {code} потеряна"


def test_provider_without_key_raises_clear_error():
    """Без ключа реальный провайдер бросает понятную ошибку (а не имитирует)."""
    cfg = ProviderConfig(api_key=None)
    provider = OpenAICompatProvider(cfg)
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        provider.chat([{"role": "user", "content": "hi"}], stage="generate")


def test_fake_provider_is_test_only():
    """Убеждаемся: в app/ нет мок-провайдера (имитация не в проде)."""
    import app.providers as prov
    assert not hasattr(prov, "MockLLMProvider")
    assert not hasattr(prov, "build_provider")
    assert not hasattr(prov, "MOCK_RESPONSES")


def test_invalid_json_input_returns_422():
    """Некорректные данные (не JSON) — понятная ошибка, не падение."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.post("/api/generate", json={"product": {"specs": []}})
    assert resp.status_code == 422


def test_real_catalog_products_pass_invariants():
    """Реальные товары каталога (products/) проходят инварианты без сети.

    Живой провайдер не нужен: test double. Проверяем, что реальные данные
    (цены, коды, характеристики) не ломают пайплайн и не теряются.
    """
    products_dir = Path(__file__).resolve().parent.parent / "products"
    if not products_dir.exists():
        pytest.skip("products/ не существует")
    files = [p for p in sorted(products_dir.glob("*.json"))]
    assert files, "products/ пуст"
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        product = ProductInput(**data)
        out = run_pipeline(product, FakeLLMProvider(f.stem))
        assert len(out.title_seo) <= TITLE_MAX, f"{f.name}: title слишком длинный"
        assert DESC_MIN <= len(out.description_seo) <= DESC_MAX, f"{f.name}: desc вне диапазона"
        if isinstance(data.get("specs"), dict):
            out_keys = {x.key.lower() for x in out.specs}
            for k in data["specs"]:
                if not k.startswith("_"):
                    assert k.lower() in out_keys, f"{f.name}: {k} потерян"


def test_parse_json_response_handles_markdown_wrap():
    """gemma возвращает JSON в ```json ... ``` — парсер обязан это выдержать."""
    from app.pipeline import _parse_json_response

    raw = '```json\n{"title": "Тест", "description": "Описание"}\n```'
    assert _parse_json_response(raw)["title"] == "Тест"
    raw2 = 'Вот результат: {"a": 1} — конец.'
    assert _parse_json_response(raw2)["a"] == 1
    assert _parse_json_response('{"x": [1,2]}')["x"] == [1, 2]
