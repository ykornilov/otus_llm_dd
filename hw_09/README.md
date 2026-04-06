# CI/CD с автотестами и проверкой на галлюцинации

RAG QA-бот по документации Yandex DataLens, оценённый через Ragas и интегрированный в CI/CD.

## Стек

| Компонент | Реализация |
|-----------|-----------|
| Документы | 15 markdown-файлов по Yandex DataLens (`docs/`) |
| Embeddings | `intfloat/multilingual-e5-large` (ChromaDB) |
| Sparse search | BM25Okapi (`rank-bm25`) |
| Retrieval | Hybrid RRF (dense + sparse) |
| LLM | `gpt-4.1-mini` via OpenAI API |
| Evaluation | Ragas 0.4.x |
| Tests | pytest |
| CI/CD | GitHub Actions |

## Структура

```
hw_09/
├── rag.py                        # RAG-пайплайн (модуль)
├── docs/                         # Исходные документы
├── requirements.txt
├── pytest.ini
└── tests/
    ├── goldens.json              # 15 вопросов с эталонными ответами
    ├── test_unit.py              # Unit-тесты компонентов (без LLM)
    ├── test_ragas.py             # Интеграционные тесты (quality gates)
    ├── results.json              # Результаты последнего прогона
    ├── report.html               # HTML-отчёт
    └── generate_report.py        # Генератор HTML-отчёта
```

## Метрики Ragas

### Faithfulness (верность контексту)
Проверяет, что каждое утверждение в ответе LLM можно вывести из retrieved документов.
Оценка 1.0 = нет галлюцинаций. Используется как основная защита от выдумок.

### Answer Relevancy (релевантность ответа)
Генерирует синтетические вопросы из ответа и измеряет их сходство с исходным вопросом.
Низкий балл означает, что ответ уходит в сторону или слишком общий.

### Context Recall (полнота контекста)
Проверяет, насколько retrieved документы покрывают эталонный ответ (ground truth).
Низкий балл = retrieval не нашёл нужные чанки.

## Пороговые значения (quality gates)

| Метрика | Порог | Обоснование |
|---------|-------|-------------|
| `faithfulness` | **0.9** | Галлюцинации критичны — допускается лишь погрешность LLM-судьи |
| `answer_relevancy` | **0.7** | Русскоязычные ответы иногда получают штраф от embedding-модели |
| `context_recall` | **0.8** | Hybrid RRF должен уверенно находить релевантные чанки |

Пороги задаются в `tests/test_ragas.py`:
```python
THRESHOLDS = {
    "faithfulness": 0.9,
    "answer_relevancy": 0.7,
    "context_recall": 0.8,
}
```

## Результаты последнего прогона

| Метрика | Score | Статус |
|---------|-------|--------|
| faithfulness | 0.933 | ✅ PASS |
| answer_relevancy | 0.837 | ✅ PASS |
| context_recall | 0.956 | ✅ PASS |

Детальный отчёт: `tests/report.html`

## Запуск тестов

```bash
# Установка зависимостей
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Unit-тесты (быстро, без LLM)
pytest tests/test_unit.py -v

# Ragas evaluation (quality gates, ~5–10 мин)
pytest tests/test_ragas.py -v -s

# Все тесты
pytest tests/ -v -s

# Генерация HTML-отчёта
python tests/generate_report.py
```

## CI/CD (GitHub Actions)

Файл: `.github/workflows/ragas-eval.yml`

```
push / pull_request → main
          │
          ▼
    unit-tests (~1 мин)
    pytest test_unit.py
          │ passed
          ▼
    ragas-eval (~5–10 мин)
    pytest test_ragas.py
          │
    ┌─────┴─────┐
  PASS ✅    FAIL ❌
 merge OK   merge blocked
```

**Артефакты:** `tests/results.json` сохраняется 30 дней.

**Secrets для GitHub Actions:**
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (если используется кастомный endpoint)
