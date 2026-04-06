"""
Generate HTML report from tests/results.json.
Run: python tests/generate_report.py
"""

import json
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results.json"
REPORT_PATH = Path(__file__).parent / "report.html"

THRESHOLDS = {
    "faithfulness": 0.9,
    "answer_relevancy": 0.7,
    "context_recall": 0.8,
}

METRIC_DESCRIPTIONS = {
    "faithfulness": "Верность контексту — насколько ответ основан только на retrieved документах (без галлюцинаций).",
    "answer_relevancy": "Релевантность ответа — насколько ответ соответствует заданному вопросу.",
    "context_recall": "Полнота контекста — насколько retrieved документы покрывают эталонный ответ.",
}


def score_color(score: float, threshold: float) -> str:
    if score >= threshold:
        return "#2ecc71"
    if score >= threshold * 0.8:
        return "#f39c12"
    return "#e74c3c"


def bar(score: float, threshold: float) -> str:
    pct = int(score * 100)
    color = score_color(score, threshold)
    tpct = int(threshold * 100)
    return f"""
    <div style="position:relative;background:#eee;border-radius:4px;height:20px;width:200px;display:inline-block;">
      <div style="width:{pct}%;background:{color};height:100%;border-radius:4px;"></div>
      <div style="position:absolute;top:0;left:{tpct}%;height:100%;border-left:2px dashed #333;"></div>
    </div>
    <span style="margin-left:8px;font-weight:bold;color:{color}">{score:.3f}</span>
    """


def generate(results: dict) -> str:
    scores = results["scores"]
    per_sample = results["per_sample"]

    summary_rows = ""
    for metric, score in scores.items():
        t = THRESHOLDS[metric]
        status = "✅ PASS" if score >= t else "❌ FAIL"
        desc = METRIC_DESCRIPTIONS.get(metric, "")
        summary_rows += f"""
        <tr>
          <td><b>{metric}</b><br><small style="color:#666">{desc}</small></td>
          <td>{bar(score, t)}</td>
          <td style="text-align:center">{t}</td>
          <td style="text-align:center;font-size:1.2em">{status}</td>
        </tr>"""

    detail_rows = ""
    for s in per_sample:
        q = s["user_input"]
        for metric in ["faithfulness", "answer_relevancy", "context_recall"]:
            score = s.get(metric)
            if score is None:
                s[metric] = 0.0

        row_color = ""
        if s["faithfulness"] < THRESHOLDS["faithfulness"] or \
           s["answer_relevancy"] < THRESHOLDS["answer_relevancy"] or \
           s["context_recall"] < THRESHOLDS["context_recall"]:
            row_color = "background:#fff3cd"

        def cell(score, metric):
            t = THRESHOLDS[metric]
            c = score_color(score, t)
            return f'<td style="text-align:center;color:{c};font-weight:bold">{score:.3f}</td>'

        detail_rows += f"""
        <tr style="{row_color}">
          <td style="max-width:350px">{q}</td>
          {cell(s['faithfulness'], 'faithfulness')}
          {cell(s['answer_relevancy'], 'answer_relevancy')}
          {cell(s['context_recall'], 'context_recall')}
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>Ragas Evaluation Report — hw_09</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 960px; margin: 40px auto; padding: 0 20px; color: #333; }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
    h2 {{ color: #2c3e50; margin-top: 40px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th {{ background: #3498db; color: white; padding: 10px 14px; text-align: left; }}
    td {{ padding: 10px 14px; border-bottom: 1px solid #eee; vertical-align: middle; }}
    tr:hover {{ background: #f8f9fa; }}
    .badge {{ display:inline-block; padding:3px 10px; border-radius:12px;
              font-size:.85em; font-weight:bold; color:white; }}
    .pass {{ background:#2ecc71; }} .fail {{ background:#e74c3c; }}
    .info-box {{ background:#eaf4fb; border-left:4px solid #3498db;
                 padding:14px 18px; border-radius:4px; margin:20px 0; }}
    code {{ background:#f4f4f4; padding:2px 6px; border-radius:3px; font-size:.9em; }}
  </style>
</head>
<body>
  <h1>Ragas Evaluation Report</h1>
  <p><b>Проект:</b> hw_09 — RAG QA-бот по документации Yandex DataLens</p>
  <p><b>Модель:</b> gpt-4.1-mini &nbsp;|&nbsp;
     <b>Embeddings:</b> intfloat/multilingual-e5-large &nbsp;|&nbsp;
     <b>Vector store:</b> ChromaDB + BM25 (Hybrid RRF)</p>
  <p><b>Goldens:</b> 15 вопросов по 15 документам</p>

  <h2>Итоговые метрики</h2>
  <table>
    <tr>
      <th>Метрика</th>
      <th>Score</th>
      <th>Порог</th>
      <th>Статус</th>
    </tr>
    {summary_rows}
  </table>

  <h2>Описание метрик</h2>
  <div class="info-box">
    <b>Faithfulness</b> — проверка на галлюцинации.<br>
    LLM-судья проверяет, что каждое утверждение в ответе можно вывести из retrieved контекста.
    Оценка 1.0 = все утверждения подтверждены источниками.
  </div>
  <div class="info-box">
    <b>Answer Relevancy</b> — релевантность ответа вопросу.<br>
    Генерирует синтетические вопросы из ответа и измеряет их сходство с исходным вопросом.
    Низкий балл = ответ уходит в сторону или слишком общий.
  </div>
  <div class="info-box">
    <b>Context Recall</b> — полнота поиска.<br>
    Проверяет, насколько retrieved документы покрывают эталонный ответ (ground truth).
    Низкий балл = retrieval не нашёл нужные чанки.
  </div>

  <h2>Пороговые значения и CI/CD</h2>
  <p>Пороги заданы в <code>tests/test_ragas.py</code>:</p>
  <table>
    <tr><th>Метрика</th><th>Порог</th><th>Обоснование</th></tr>
    <tr><td>faithfulness</td><td>0.9</td><td>Галлюцинации критичны — допускается минимальная погрешность LLM-судьи</td></tr>
    <tr><td>answer_relevancy</td><td>0.7</td><td>Ответы на русском иногда получают штраф от embedding-модели</td></tr>
    <tr><td>context_recall</td><td>0.8</td><td>Hybrid RRF должен уверенно находить релевантные чанки</td></tr>
  </table>
  <p>В GitHub Actions (<code>.github/workflows/ragas-eval.yml</code>) пайплайн падает
     (<code>exit code 1</code>), если хотя бы один тест не проходит порог. Merge в main блокируется.</p>
  <p>Запуск локально:</p>
  <pre><code>pytest hw_09/tests/test_ragas.py -v --tb=short</code></pre>

  <h2>Детализация по вопросам</h2>
  <table>
    <tr>
      <th>Вопрос</th>
      <th style="text-align:center">Faithfulness<br><small>≥ 0.9</small></th>
      <th style="text-align:center">Answer Relevancy<br><small>≥ 0.7</small></th>
      <th style="text-align:center">Context Recall<br><small>≥ 0.8</small></th>
    </tr>
    {detail_rows}
  </table>

  <p style="margin-top:40px;color:#999;font-size:.85em">
    Сгенерировано: <code>tests/generate_report.py</code> &nbsp;|&nbsp;
    Данные: <code>tests/results.json</code>
  </p>
</body>
</html>"""
    return html


if __name__ == "__main__":
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    html = generate(results)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Report saved to {REPORT_PATH}")
