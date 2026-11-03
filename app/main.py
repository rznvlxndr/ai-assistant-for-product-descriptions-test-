"""Точка входа: CLI (uv run) и HTTP API (FastAPI /uvicorn).

Примеры:
    uv run main.py fixtures/manometer_basic.json --format md
    uv run main.py fixtures/trm1_catalog_slice.json --format txt
    uv run main.py products/ --batch --workers 8 --format md   # папка с JSON
    uvicorn app.main:app   (или uv run uvicorn app.main:app)
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import ProviderConfig
from app.models import ProductInput
from app.pipeline import (
    ProductOutput,
    render_markdown,
    render_text,
    run_pipeline_with_key,
)
from app.providers import LLMError, OpenAICompatProvider, RateLimiter

app = FastAPI(title="AI-помощник для товарных карточек", version="1.0.0")


class ApiRequest(BaseModel):
    product: dict


@app.post("/api/generate", response_model=ProductOutput)
def api_generate(req: ApiRequest) -> ProductOutput:
    """POST /api/generate  {"product": {...}}"""
    try:
        product = ProductInput(**req.product)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Некорректный вход: {e}") from e
    cfg = ProviderConfig.from_env()
    try:
        return run_pipeline_with_key(product, cfg)
    except LLMError as e:
        raise HTTPException(status_code=503, detail=f"LLM недоступен: {e}") from e


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _render(out: ProductOutput, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(out.model_dump(), ensure_ascii=False, indent=2)
    if fmt == "md":
        return render_markdown(out)
    return render_text(out)


def _load_product(path: Path) -> tuple[Path, ProductInput]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return path, ProductInput(**data)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Генерация карточки товара")
    p.add_argument("input", type=Path, help="JSON-файл или папка с JSON (batch)")
    p.add_argument("--format", choices=["json", "md", "txt"], default="json")
    p.add_argument("--env", type=Path, default=ROOT / ".env")
    p.add_argument("--batch", action="store_true", help="обработать папку как пакет")
    p.add_argument("--workers", type=int, default=8, help="параллельных потоков (batch)")
    p.add_argument("--out-dir", type=Path, default=None, help="куда писать карточки (batch)")
    args = p.parse_args(argv)

    cfg = ProviderConfig.from_env(dotenv_path=args.env)

    if args.batch or args.input.is_dir():
        files = sorted(args.input.glob("*.json")) if args.input.is_dir() else [args.input]
        if not files:
            print(f"Ошибка: в {args.input} нет *.json файлов", file=sys.stderr)
            return 2
        limiter = RateLimiter(cfg.rpm)
        out_dir = args.out_dir or (args.input / ".." / "out").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = {"json": ".json", "md": ".md", "txt": ".txt"}[args.format]

        results: list[tuple[Path, bool, str]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {}
            for f in files:
                try:
                    path, product = _load_product(f)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"✗ {f.name}: невалидный вход — {e}", file=sys.stderr)
                    continue
                fut = ex.submit(
                    run_pipeline_with_key, product, cfg, limiter
                )
                futures[fut] = path

            for fut in as_completed(futures):
                path = futures[fut]
                try:
                    out = fut.result()
                    (out_dir / (path.stem + ext)).write_text(
                        _render(out, args.format), encoding="utf-8"
                    )
                    results.append((path, True, ""))
                    print(f"✓ {path.name} → {out_dir / (path.stem + ext)}")
                except LLMError as e:
                    results.append((path, False, str(e)))
                    print(f"✗ {path.name}: LLM — {e}", file=sys.stderr)
                except Exception as e:
                    results.append((path, False, str(e)))
                    print(f"✗ {path.name}: {e}", file=sys.stderr)

        ok = sum(1 for _, s, _ in results if s)
        print(f"\nГотово: {ok}/{len(results)} карточек → {out_dir}")
        return 0 if ok == len(results) else 3

    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        product = ProductInput(**data)
    except json.JSONDecodeError as e:
        print(f"Ошибка: файл не является валидным JSON — {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Ошибка: некорректные данные товара — {e}", file=sys.stderr)
        return 2

    try:
        out = run_pipeline_with_key(product, cfg)
    except LLMError as e:
        print(f"Ошибка LLM: {e}", file=sys.stderr)
        return 3

    print(_render(out, args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
