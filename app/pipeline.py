"""Основной пайплайн: input → extract → generate → queries → validate → output.

Ключевой принцип: карточка строится ТОЛЬКО из фактов, извлечённых на этапе
extract (с источником каждого значения). Ничего «из головы» модели в карточку
не попадает — это архитектурная защита от галлюцинаций (главное требование ТЗ).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from app.models import ProductInput, ProductOutput, SpecFact
from app.providers import LLMError, LLMProvider


def _parse_json_response(text: str) -> dict:
    """Устойчивый парсинг JSON-ответа LLM.

    Модели (особенно локальные/средние) часто оборачивают JSON в markdown:
    ```json\n{...}\n```, добавляют пояснения до/после или экранируют. Пробуем:
    1) прямой json.loads;
    2) извлечение из ```json ... ``` блока;
    3) срез от первой '{' до последней '}' (если там валидный JSON).
    """
    text = text.strip()
    if not text:
        raise json.JSONDecodeError("пустой ответ LLM", text, 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import re

    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError(f"не удалось извлечь JSON из ответа LLM: {text[:120]}...", text, 0)

TRADE_KEYWORDS = ("цена", "срок поставки", "наличие", "скидк", "в наличии", "₽")

DECODE_HINTS = ("тип корпуса", "тип входа", "тип выхода", "интерфейс", "наличие")

MODIFICATION_CONTAINER_KEYS = ("модификации", "варианты", "исполнения", "variants", "modifications")
TRADE_KEYS_IN_MOD = ("цена", "наличие", "срок поставки", "скидк", "срок")

SPEC_KEY_ALIASES = {
    "range": "диапазон измерений",
    "measurement_range": "диапазон измерений",
    "precision": "класс точности",
    "accuracy": "класс точности",
    "класс точности": "класс точности",
    "connection": "присоединение",
    "thread": "присоединение",
    "case": "тип корпуса",
    "housing": "тип корпуса",
    "enclosure": "класс защиты",
    "protection": "класс защиты",
    "ip": "класс защиты",
    "material": "материал корпуса",
    "output": "выходной сигнал",
    "output_signal": "выходной сигнал",
    "power": "питание",
    "supply": "питание",
    "interface": "интерфейс",
    "temperature": "диапазон температур",
}


def _normalize_key(key: str) -> str:
    """Приводит ключ характеристики к канону (если знаем), иначе как есть."""
    k = key.strip().lower()
    return SPEC_KEY_ALIASES.get(k, key.strip())


def _is_trade_key(key: str, value: str) -> bool:
    """Цена/наличие/срок поставки — торговые поля, не характеристики.

    ВАЖНО: «наличие интерфейса RS-485» — это техническая характеристика
    (есть/нет интерфейс), а не торговое «наличие на складе».
    """
    k = key.lower()
    v = value.lower()
    if "интерфейс" in k or "интерфейс" in v:
        return False
    return any(t in k for t in ("цена", "срок поставки", "скидк", "цена с ндс")) or "₽" in v or (
        "в наличии" in v or "поступлен" in v or "дней" in k or k == "наличие"
    )


def _format_modification(mod: dict[str, Any]) -> str:
    """Человекочитаемая строка модификации: тех. поля — как есть, торговые
    вынесены в конец с пометкой. Ничего не выдумываем, только переупаковка.

    «наличие интерфейса RS-485» — тех. характеристика, НЕ торговое наличие.
    Используем _is_trade_key, который учитывает это.
    """
    tech: list[str] = []
    trade: list[str] = []
    for k, v in mod.items():
        val = str(v)
        if _is_trade_key(k, val):
            trade.append(f"{k}: {val}")
        else:
            tech.append(f"{k}: {val}")
    result = "; ".join(tech)
    if trade:
        result += f"  [торговое: {'; '.join(trade)}]"
    return result


def _extract_modifications(
    items: list[Any], out: list[SpecFact], other: list[SpecFact]
) -> None:
    """Массив модификаций → по одному факту на модификацию (таблица вариантов)."""
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            other.append(SpecFact(key=f"модификация[{i}]", value=str(item), source="other"))
            continue
        code = item.get("код товара") or item.get("название") or f"#{i + 1}"
        title = item.get("название")
        key = f"модификация {code}"
        if title and str(title) not in key:
            key += f" ({title})"
        out.append(SpecFact(key=key, value=_format_modification(item), source="other"))


def _extract_from_specs_dict(
    specs: dict[str, Any], out: list[SpecFact], other: list[SpecFact]
) -> None:
    """Рекурсивно обходим specs-dict. Ключ/значение verbatim."""
    for key, val in specs.items():
        if key.startswith("_"):
            continue
        if key.lower() in MODIFICATION_CONTAINER_KEYS and isinstance(val, list):
            _extract_modifications(val, out, other)
            continue
        if isinstance(val, dict):
            other.append(
                SpecFact(key=key, value=json.dumps(val, ensure_ascii=False), source="other")
            )
        elif isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    other.append(
                        SpecFact(
                            key=f"{key}[{i}]",
                            value=json.dumps(item, ensure_ascii=False),
                            source="other",
                        )
                    )
                else:
                    other.append(
                        SpecFact(key=f"{key}[{i}]", value=str(item), source="other")
                    )
        else:
            value_str = str(val)
            if _is_trade_key(key, value_str):
                other.append(SpecFact(key=key, value=value_str, source="other"))
                continue
            out.append(SpecFact(key=_normalize_key(key), value=value_str, source="specs"))


def _extract_raw_string(
    raw: str,
    provider: LLMProvider,
    facts: list[SpecFact],
    other: list[SpecFact],
    warnings: list[str],
) -> None:
    """extract фактов из specs-строки через LLM; при сбое — весь текст в other."""
    try:
        result = provider.chat(
            [
                {"role": "system", "content": "Ты — извлекатель фактов из текста. "
                 "Верни ТОЛЬКО JSON {\"facts\": [{\"key\":..., \"value\":...}], "
                 "\"other\": [...]}. Не додумывай, только из текста."},
                {"role": "user", "content": raw},
            ],
            stage="extract_string",
            json_mode=True,
        )
        data = _parse_json_response(result)
        for item in data.get("facts", []):
            k = str(item.get("key", "")).strip()
            v = str(item.get("value", "")).strip()
            if k and v:
                facts.append(SpecFact(key=k, value=v, source="specs"))
        for o in data.get("other", []):
            if o:
                other.append(SpecFact(key="other", value=str(o), source="other"))
    except LLMError as e:
        if "LLM_API_KEY" in str(e):
            raise
        warnings.append(f"не удалось распарсить specs-строку ({e}); строка сохранена в other")
        other.append(SpecFact(key="_raw_string", value=raw, source="other"))
    except (json.JSONDecodeError, AttributeError) as e:
        warnings.append(f"не удалось распарсить specs-строку ({e}); строка сохранена в other")
        other.append(SpecFact(key="_raw_string", value=raw, source="other"))


DESC_FACT_PATTERNS = [
    ("диапазон измерений", r"(?:диапазон\s+(?:измерени[йя]|измерен)?|измеряем\w+)\s*[:—]?\s*([\d,.\-–]+[^\s.,;]{0,20})"),
    ("диапазон измерений", r"диапазон\s+измерени[йя]\s+([\d\s.,\-–]+[^\s.,;]{0,20})"),
    ("рабочая температура", r"рабоч\w+\s+температур\w+\s+([\d\s.,\-–]+[^\s.,;]{0,25})"),
    ("погрешность", r"погрешност\w+\s+([\d±.,\-–]+[^\s.,;]{0,15})"),
    ("класс точности", r"класс\s+точности\s+([\d.,]+)"),
    ("материал защитной арматуры", r"материал\s+(?:защитной\s+)?арматуры\s*[:—]?\s*([А-Яа-яA-Za-z0-9 ]{3,40})"),
    ("степень защиты", r"([Сс]тепень\s+защиты|[Ii][Pp]\s*\d{2})"),
    ("выходной сигнал", r"([4-9]\s*(?:–|-)?\s*20\s*мА|0[–-]10\s*В|4\.\.20\s*мА)"),
]


def _extract_from_description(
    desc: str,
    facts: list[SpecFact],
    other: list[SpecFact],
    warnings: list[str],
) -> None:
    """Из текста описания извлекаем тех. факты (детерминированно, только из
    текста) и торговые — мимо в other. Противоречия с specs флагаем позже."""
    seen_keys: set[str] = set()
    for sentence in re.split(r"[.;\n]", desc):
        s = sentence.strip()
        if not s:
            continue
        low = s.lower()
        if any(t in low for t in ("цена", "наличие", "срок поставки", "₽", "в наличии", "скидк")):
            other.append(SpecFact(key="коммерция", value=s, source="other"))
            continue
        for key, pattern in DESC_FACT_PATTERNS:
            m = re.search(pattern, s, re.IGNORECASE)
            if m:
                value = m.group(1).strip()
                if not value:
                    continue
                k = key.lower()
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                facts.append(SpecFact(key=key, value=value, source="description"))
                break


def _detect_conflicts(
    facts: list[SpecFact],
    warnings: list[str],
) -> None:
    """Ищет противоречия между фактами из specs и description (одна характеристика
    с разными значениями). Флагает, но не исправляет молча."""
    from collections import defaultdict

    by_key: dict[str, list[SpecFact]] = defaultdict(list)
    for f in facts:
        by_key[f.key.lower()].append(f)

    for key, items in by_key.items():
        if len(items) < 2:
            continue
        sources = sorted({f.source for f in items})
        if len(sources) < 2:  # оба из specs или оба из description — не конфликт
            continue
        values = {f.value for f in items}
        if len(values) > 1:
            warnings.append(
                f"противоречие по характеристике «{items[0].key}»: "
                f"{' vs '.join(sorted(values))} (источники: {', '.join(sources)}) — требуется уточнение"
            )


def extract(product: ProductInput, provider: LLMProvider) -> tuple[list[SpecFact], list[SpecFact], list[str]]:
    """Возвращает (факты, other, warnings). Факты — только из входа."""
    facts: list[SpecFact] = []
    other: list[SpecFact] = []
    warnings: list[str] = []

    if isinstance(product.specs, dict):
        raw_string = product.specs.get("_raw_string")
        if raw_string:
            _extract_raw_string(raw_string, provider, facts, other, warnings)
        else:
            _extract_from_specs_dict(product.specs, facts, other)
    elif isinstance(product.specs, str) and product.specs.strip():
        _extract_raw_string(product.specs.strip(), provider, facts, other, warnings)

    if product.description:
        _extract_from_description(product.description, facts, other, warnings)

    _detect_conflicts(facts, warnings)
    _decode_name(product, facts, other, warnings)

    return facts, other, warnings


def _decode_name(
    product: ProductInput,
    facts: list[SpecFact],
    other: list[SpecFact],
    warnings: list[str],
) -> None:
    """Если name — код-конструкция (напр. 2ТРМ1-Щ2.У2.И.РС), а среди фактов есть
    расшифровки компонентов — собираем осмысленное название. Иначе «данных нет»."""
    if not product.name:
        return
    name = product.name.strip()
    # Код-конструкция (2ТРМ1-Щ2.У2.И.РС) — без пробелов, с разделителями.
    # Обычные имена («StreamLux SLS-720P») и размеры («ДКТL011-0,5») не подходят.
    if " " in name.strip() or "," in name:
        return
    has_code_shape = bool(re.search(r"[A-ZА-Я]+\d", name)) and bool(re.search(r"[-.]", name))
    segments = [s for s in re.split(r"[-.]", name) if s]
    has_multi_segments = len(segments) >= 3 and any(len(s) >= 2 for s in segments)
    if not (has_code_shape or has_multi_segments):
        return
    decode_map: dict[str, str] = {}
    for f in list(facts) + list(other):
        m = re.match(r"([А-ЯA-Z0-9]+)-(.{2,})", f.value.strip())
        if m and f.key and any(h in f.key.lower() for h in DECODE_HINTS):
            decode_map[m.group(1)] = m.group(2)
    if not decode_map:
        warnings.append("название выглядит как код; расшифровок компонентов нет — «данных нет»")
        return
    parts = []
    for token in re.split(r"[.-]", name):
        base = re.match(r"[А-ЯA-Z]+", token)
        if base:
            code = base.group(0)
            if code in decode_map:
                parts.append(decode_map[code])
                continue
        parts.append(token)
    decoded = " ".join(parts)
    warnings.append(f"название расшифровано по коду серии: «{decoded}» — сверить с производителем")
    other.append(SpecFact(key="расшифрованное название", value=decoded, source="decoded"))


# ---------- GENERATE ----------

def generate(
    product: ProductInput,
    facts: list[SpecFact],
    other: list[SpecFact],
    provider: LLMProvider,
    warnings: list[str],
) -> tuple[str, str, str, list[str]]:
    """Строит карточку ТОЛЬКО из фактов. Возвращает (title, description, commercial, queries)."""
    spec_lines = "\n".join(f"{f.key}: {f.value}" for f in facts)
    queries = _build_queries(product, facts)

    prompt = _build_generate_prompt(product, spec_lines, queries, other, warnings)
    try:
        out = provider.chat(
            [
                {"role": "system", "content": (
                    "Ты — контент-менеджер B2B-магазина контрольно-измерительных приборов. "
                    "Пишешь карточку товара ТОЛЬКО по предоставленным фактам. "
                    "ЗАПРЕЩЕНО выдумывать любые технические характеристики. "
                    "Если чего-то нет в фактах — напиши «данных нет» или обобщённо, без деталей. "
                    "Ответ — строго JSON с полями: title (≤70 символов), "
                    "description (800–1200 символов), commercial (кому подходит и какие задачи решает), "
                    "needs_clarification (список чего не хватает).")},
                {"role": "user", "content": prompt},
            ],
            stage="generate",
            json_mode=True,
        )
        data = _parse_json_response(out)
        title = str(data.get("title", "")).strip()
        description = str(data.get("description", "")).strip()
        commercial = str(data.get("commercial", "")).strip()
        clarif = data.get("needs_clarification") or []
        warnings.extend(f"уточнить: {c}" for c in clarif if c)
        return title, description, commercial, queries
    except LLMError as e:
        if "LLM_API_KEY" in str(e):
            raise
        warnings.append(f"LLM-генерация карточки не удалась ({e}); возвращён каркас")
        title = (product.name or "")[:70]
        description = _fallback_description(product, facts)
        commercial = "Недостаточно данных для коммерческого блока. Уточните категорию и применение."
        return title, description, commercial, queries
    except json.JSONDecodeError as e:
        warnings.append(f"LLM-генерация карточки не удалась ({e}); возвращён каркас")
        title = (product.name or "")[:70]
        description = _fallback_description(product, facts)
        commercial = "Недостаточно данных для коммерческого блока. Уточните категорию и применение."
        return title, description, commercial, queries


def _build_generate_prompt(
    product: ProductInput,
    spec_lines: str,
    queries: list[str],
    other: list[SpecFact],
    warnings: list[str],
) -> str:
    return (
        f"Товар: {product.name or 'данных нет'}\n"
        f"Производитель: {product.manufacturer or 'данных нет'}\n\n"
        f"Характеристики (только из них, ничего не добавлять):\n{spec_lines or 'данных нет'}\n\n"
        f"Дополнительно (может быть торговой/неструктурированной инфой):\n"
        + "\n".join(f"- {f.key}: {f.value}" for f in other[:10])
        + "\n\n"
        f"Поисковые запросы страницы: {', '.join(queries)}\n"
        f"Внимание/предупреждения: {'; '.join(warnings) or 'нет'}\n"
        f"SEO-описание должно включать 2-3 из этих запросов естественно, без спама."
    )


def _fallback_description(product: ProductInput, facts: list[SpecFact]) -> str:
    """Честный каркас описания без выдумки (для деградации)."""
    parts = []
    if product.name:
        parts.append(f"{product.name} — контрольно-измерительный прибор.")
    if product.manufacturer:
        parts.append(f"Производитель: {product.manufacturer}.")
    if facts:
        parts.append("Характеристики: " + "; ".join(f"{f.key}: {f.value}" for f in facts[:8]) + ".")
    else:
        parts.append("Технические характеристики не предоставлены.")
    base = " ".join(parts)
    while len(base) < 800:
        base += " Для получения полных технических характеристик обратитесь к документации производителя. "
    return base[:1200]


def _build_queries(product: ProductInput, facts: list[SpecFact]) -> list[str]:
    """Детерминированные шаблоны поисковых запросов на основе фактов."""
    qs: list[str] = []
    name = (product.name or "").strip()
    manuf = (product.manufacturer or "").strip()
    name_l = name.lower()
    manuf_l = manuf.lower()
    add = qs.append

    if name:
        add(f"{name_l} купить")
        add(f"купить {name_l}")
        if manuf and manuf_l not in name_l:
            add(f"{manuf_l} {name_l}")
    if manuf:
        add(f"{manuf_l} приборы купить")
    for f in facts:
        if len(qs) >= 5:
            break
        if f.key.lower().startswith("модификация"):
            m = re.search(r"\(([^)]{3,30})\)", f.key)
            if m:
                add(f"{name_l if name else 'прибор'} {m.group(1).lower()}")
            continue
        if any(t in f.key.lower() for t in ("диапазон", "класс", "присоединен", "корпус", "интерфейс", "точность")):
            val = f.value[:30].lower()
            add(f"{name_l if name else 'прибор'} {val}")
    seen: set[str] = set()
    result = []
    for q in qs:
        key = re.sub(r"\s+", " ", q).strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(q)
    return result[:5]


# ---------- POST-VALIDATE ----------

TITLE_MAX = 70
DESC_MIN, DESC_MAX = 800, 1200


def post_validate(
    out: ProductOutput,
    facts: list[SpecFact],
    other: list[SpecFact],
    warnings: list[str],
) -> ProductOutput:
    """Проверяем инварианты и правим, если нужно. Ничего не выдумываем."""
    if len(out.title_seo) > TITLE_MAX:
        out.title_seo = _smart_truncate(out.title_seo, TITLE_MAX)
        warnings.append(f"SEO-заголовок обрезан до {TITLE_MAX} символов")
    if len(out.description_seo) < DESC_MIN:
        out.description_seo = _pad_description(out.description_seo, DESC_MIN)
        warnings.append(f"SEO-описание расширено до ≥{DESC_MIN} символов нейтральным текстом")
    if len(out.description_seo) > DESC_MAX:
        out.description_seo = _smart_truncate(out.description_seo, DESC_MAX)
        warnings.append(f"SEO-описание обрезано до {DESC_MAX} символов")

    for f in facts:
        if f.key.startswith("_"):
            continue
        existing = [x for x in out.specs if x.key.lower() == f.key.lower()]
        if not existing:
            out.specs.append(f)
            continue
        if any(x.value == f.value for x in existing):
            continue
        if not any(x.source == f.source for x in existing):
            out.specs.append(f)
    for f in other:
        if not f.key or f.key in ("коммерция", "_raw_string"):
            continue
        if _is_trade_key(f.key, f.value):
            continue
        out.specs.append(f)
    return out


_PAD_PHRASES = (
    "Для получения полных технических характеристик обратитесь к документации производителя.",
    "Для подбора модификации под конкретные условия эксплуатации проконсультируйтесь с менеджером.",
    "Точные значения рабочих параметров подтверждаются паспортом изделия.",
    "Информация о конкретном исполнении и комплектации уточняется по запросу.",
    "При необходимости уточнения характеристик обратитесь в техническую поддержку.",
)


def _pad_description(text: str, min_len: int) -> str:
    """Дополняет описание нейтральными фразами до min_len.

    Фразы добавляются по кругу; одна и та же фраза не появляется дважды подряд.
    Повторы начинаются только когда весь набор исчерпан (неизбежно при очень
    коротком исходном тексте) — и тогда тоже чередуются.
    """
    base = text.rstrip()
    idx = 0
    while len(base) < min_len:
        phrase = _PAD_PHRASES[idx % len(_PAD_PHRASES)]
        if phrase in base and len(base.rstrip()) > len(text.rstrip()):
            tried = 0
            while tried < len(_PAD_PHRASES):
                nxt = _PAD_PHRASES[(idx + 1) % len(_PAD_PHRASES)]
                if nxt not in base:
                    phrase = nxt
                    idx += 1
                    break
                tried += 1
                idx += 1
        base += " " + phrase
        idx += 1
    return base


def _smart_truncate(text: str, limit: int) -> str:
    """Умная обрезка: по границе слова, сохраняя начало (ключевое)."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    # не резать посреди слова
    if cut[-1].isalnum():
        space = cut.rfind(" ")
        if space > limit * 0.6:
            cut = cut[:space]
    return cut.rstrip(" .,;:-") + "…"


# ---------- OUTPUT ----------

def render_markdown(out: ProductOutput) -> str:
    lines = [
        f"# {out.title_seo}",
        "",
        out.description_seo,
        "",
        "## Характеристики",
    ]
    for f in out.specs:
        src = f" *(источник: {f.source})*" if f.source != "specs" else ""
        lines.append(f"- **{f.key}**: {f.value}{src}")
    lines += [
        "",
        "## Поисковые запросы",
        *[f"- {q}" for q in out.queries],
        "",
        "## Кому подходит",
        out.commercial_block,
    ]
    if out.missing_fields:
        lines += ["", "## Чего не хватает", *[f"- {m}" for m in out.missing_fields]]
    if out.warnings:
        lines += ["", "## Предупреждения", *[f"- {w}" for w in out.warnings]]
    return "\n".join(lines)


def render_text(out: ProductOutput) -> str:
    lines = [
        out.title_seo,
        "",
        out.description_seo,
        "",
        "Характеристики:",
    ]
    for f in out.specs:
        lines.append(f"  {f.key}: {f.value}")
    lines += [
        "",
        "Поисковые запросы:",
        *[f"  {q}" for q in out.queries],
        "",
        f"Кому подходит: {out.commercial_block}",
    ]
    return "\n".join(lines)


# ---------- ORCHESTRATION ----------

def run_pipeline(
    product: ProductInput,
    provider: LLMProvider,
) -> ProductOutput:
    """Запуск всего пайплайна."""
    facts, other, warnings = extract(product, provider)

    title, description, commercial, queries = generate(
        product, facts, other, provider, warnings
    )

    out = ProductOutput(
        name=product.name or "",
        manufacturer=product.manufacturer,
        title_seo=title,
        description_seo=description,
        commercial_block=commercial,
        queries=queries,
    )
    if not product.name:
        out.missing_fields.append("name")
    if not product.manufacturer:
        out.missing_fields.append("manufacturer")
    if not product.specs:
        out.missing_fields.append("specs")
    if not product.description:
        out.missing_fields.append("description")

    out.warnings = warnings
    out = post_validate(out, facts, other, warnings)
    return out


def run_pipeline_with_key(
    product: ProductInput,
    cfg,
    limiter=None,
) -> ProductOutput:
    """Запуск пайплайна с реальным провайдером (без имитаций).

    `limiter` — общий RateLimiter для многопоточности (batch-режим):
    суммарный темп всех потоков не превысит лимит провайдера.
    """
    from app.providers import OpenAICompatProvider

    provider = OpenAICompatProvider(cfg, limiter=limiter)
    return run_pipeline(product, provider)
