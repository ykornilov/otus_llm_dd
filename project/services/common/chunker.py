"""
Split FORMULA_REFERENCE into semantic chunks for RAG retrieval.

Each chunk is a dict:
    {
        "title":      str,   # function name or section name
        "text":       str,   # full chunk text (sent to embedder + returned to LLM)
        "chunk_type": str,   # "function" | "overview"
    }
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prompts.formula import FORMULA_REFERENCE

# ── Russian keyword aliases ───────────────────────────────────────────────────
# Appended to each chunk so both BM25 (exact token match) and dense retrieval
# (semantic match for Russian queries) can find the right function.
# Format added to chunk text:  "Русские синонимы: ..."
CHUNK_ALIASES: dict[str, list[str]] = {
    # ── Arithmetic ────────────────────────────────────────────────────────────
    "SQRT":              ["квадратный корень", "корень числа", "извлечь корень", "square root"],
    "ROUND":             ["округление", "округлить", "знаки после запятой", "округлить до"],
    "MEDIAN":            ["медиана", "медианное значение", "медианный"],
    "ABS":               ["модуль", "абсолютное значение"],
    "CEILING":           ["округление вверх", "округлить вверх", "ceiling"],
    "FLOOR":             ["округление вниз", "округлить вниз", "floor"],
    "POWER":             ["степень", "возведение в степень", "в степени"],

    # ── String ────────────────────────────────────────────────────────────────
    "LEFT":              ["первые символы", "начало строки", "первые N символов",
                          "взять символы слева", "символы с начала"],
    "RIGHT":             ["последние символы", "конец строки", "символы с конца",
                          "последние N символов"],
    "REPLACE (string)":  ["заменить", "замена", "заменить подстроку", "подстановка"],
    "CONTAINS (string)": ["содержит", "включает", "входит ли", "проверить вхождение"],
    "LEN":               ["длина строки", "количество символов", "число символов"],
    "UPPER":             ["верхний регистр", "заглавные буквы", "прописные буквы"],
    "LOWER":             ["нижний регистр", "строчные буквы", "маленькие буквы"],
    "TRIM":              ["пробелы по краям", "убрать пробелы", "обрезать пробелы", "лишние пробелы"],
    "SUBSTR":            ["подстрока", "часть строки", "вырезать строку"],
    "SPLIT":             ["разбить строку", "разделить", "разделитель", "разбивка по символу"],
    "CONCAT":            ["соединить строки", "склеить", "объединить строки", "конкатенация"],
    "STARTSWITH (string)": ["начинается с", "начало строки проверить"],
    "ENDSWITH":          ["заканчивается на", "конец строки проверить"],
    "FIND":              ["найти позицию", "позиция подстроки", "поиск в строке"],
    "SPACE":             ["пробел", "добавить пробелы"],

    # ── Date / time ───────────────────────────────────────────────────────────
    "TODAY":             ["сегодня", "текущая дата", "сегодняшняя дата", "дата сегодня"],
    "NOW":               ["текущее время", "сейчас", "текущая дата и время"],
    "YEAR":              ["год", "текущий год", "извлечь год", "год события"],
    "MONTH":             ["месяц", "текущий месяц", "извлечь месяц", "номер месяца"],
    "WEEK":              ["неделя", "номер недели", "текущая неделя", "номер текущей недели"],
    "DAY":               ["день", "текущий день", "число месяца", "день даты"],
    "HOUR":              ["час", "текущий час", "часы"],
    "MINUTE":            ["минута", "минуты"],
    "QUARTER":           ["квартал", "номер квартала"],
    "DAYOFWEEK":         ["день недели", "номер дня недели"],
    "DATETRUNC":         ["начало периода", "усечь дату", "начало недели",
                          "начало месяца", "начало года", "начало квартала"],
    "DATEADD":           ["прибавить дни", "добавить дни", "N дней назад",
                          "дней назад", "сдвинуть дату", "добавить к дате"],
    "DATEPART":          ["часть даты", "компонент даты", "извлечь часть даты"],
    "AGO":               ["N периодов назад", "сравнить с прошлым периодом"],

    # ── Aggregation ───────────────────────────────────────────────────────────
    "AVG":               ["среднее", "среднее значение", "средний"],
    "MAX":               ["максимум", "максимальное значение", "наибольшее"],
    "MIN":               ["минимум", "минимальное значение", "наименьшее"],
    "COUNT":             ["количество", "число строк", "подсчёт записей"],
    "COUNTD":            ["уникальные", "количество уникальных", "уникальные значения",
                          "distinct"],
    "SUM_IF":            ["сумма с условием", "условная сумма", "сумма по условию",
                          "сумма если"],
    "COUNT_IF":          ["количество с условием", "подсчёт с условием",
                          "число записей по условию", "количество если"],
    "AVG_IF":            ["среднее с условием", "условное среднее", "среднее если"],
    "COUNTD_IF":         ["уникальные с условием", "количество уникальных по условию"],

    # ── LOD / Aggregate Overview ──────────────────────────────────────────────
    # FIXED is a keyword, not a standalone function — lives in the overview chunk
    "Aggregate Functions — Overview": [
        "FIXED", "LOD", "уровень детализации",
        "игнорируя другие измерения", "игнорируя измерения",
        "фиксировать измерение", "фиксированный контекст",
        "независимо от фильтра", "по всем данным",
        "агрегация с измерениями", "BEFORE FILTER BY",
        "INCLUDE измерения", "EXCLUDE измерения",
        "фиксировать по полю", "рассчитать по жанру игнорируя",
        "средний по жанру игнорируя", "максимальный по стране игнорируя",
    ],

    # ── Window functions ──────────────────────────────────────────────────────
    "RANK (window)":           ["ранг", "рейтинг", "место в рейтинге", "позиция",
                                 "присвоить ранг", "RANK TOTAL", "RANK WITHIN"],
    "RANK_DENSE (window)":     ["плотный ранг", "ранг без пропусков"],
    "RANK_UNIQUE (window)":    ["уникальный ранг", "ранг уникальный"],
    "RANK_PERCENTILE (window)":["процентиль ранга", "перцентиль"],
    "MAVG (window)":           ["скользящее среднее", "скользящий средний",
                                 "moving average", "среднее за период"],
    "RSUM (window)":           ["нарастающая сумма", "накопительная сумма",
                                 "нарастающий итог", "нарастающая", "cumulative sum"],
    "RCOUNT (window)":         ["нарастающее количество", "накопительный подсчёт"],
    "RAVG (window)":           ["нарастающее среднее", "накопительное среднее"],
    "SUM (window)":            ["сумма TOTAL", "общая сумма", "сумма по всем",
                                 "доля от общей суммы"],
    "AVG (window)":            ["среднее TOTAL", "общее среднее", "среднее по всем"],
    "COUNT (window)":          ["количество TOTAL", "общее количество"],
    "LAG (window)":            ["предыдущее значение", "значение за предыдущий период"],
    "MSUM (window)":           ["скользящая сумма", "сумма за период"],
    "MCOUNT (window)":         ["скользящее количество"],

    # ── Markup ────────────────────────────────────────────────────────────────
    "BOLD":     ["жирный шрифт", "жирным", "выделить жирным", "полужирный", "bold text"],
    "COLOR":    ["цвет", "раскрасить", "цветной", "цветовое выделение",
                 "изменить цвет текста", "цветная подсветка"],
    "URL":      ["ссылка", "гиперссылка", "URL-адрес", "кликабельная ссылка",
                 "ссылка на сайт"],
    "BR":       ["перенос строки", "новая строка", "разрыв строки", "line break"],
    "MARKUP":   ["разметка", "размеченный текст", "форматирование текста"],
    "ITALIC":   ["курсив", "курсивный шрифт", "наклонный текст"],
    "TOOLTIP":  ["подсказка", "tooltip", "всплывающая подсказка"],

    # ── Logic ─────────────────────────────────────────────────────────────────
    "IF":       ["условие", "если", "ветвление", "условный оператор"],
    "CASE":     ["перевод", "сопоставление", "switch", "классификация"],
    "ISNULL":   ["пустое значение", "проверить NULL", "является ли пустым"],
    "IFNULL":   ["подставить если пустое", "заменить NULL", "значение по умолчанию"],
    "ZN":       ["ноль вместо NULL", "заменить NULL на ноль"],
}


def _apply_aliases(chunk: dict) -> dict:
    """Append Russian synonym line to chunk text if aliases exist for its title."""
    aliases = CHUNK_ALIASES.get(chunk["title"])
    if aliases:
        chunk = dict(chunk)   # don't mutate original
        chunk["text"] = chunk["text"] + "\n\nРусские синонимы: " + ", ".join(aliases) + "."
    return chunk


def get_chunks() -> list[dict]:
    """
    Returns ~199 chunks:
      - 197 individual function/operator entries
      - 1  Aggregate Functions overview
      - 1  Window Functions overview
    """
    ref = (
        FORMULA_REFERENCE
        .replace("<Beginning of the reference>", "")
        .replace("<End of the reference>", "")
        .strip()
    )

    # Locate overview sections
    agg_match = re.search(r"^# AGGREGATE FUNCTIONS", ref, re.MULTILINE)
    win_match = re.search(r"^# WINDOW FUNCTIONS", ref, re.MULTILINE)

    func_part = ref[: agg_match.start()].strip() if agg_match else ref
    agg_part  = ref[agg_match.start() : win_match.start()].strip() if (agg_match and win_match) else ""
    win_part  = ref[win_match.start() :].strip() if win_match else ""

    chunks: list[dict] = []

    # ── 1. Individual function / operator chunks ───────────────────────────
    parts = re.split(r"(?=^## \[)", func_part, flags=re.MULTILINE)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        first_line = part.splitlines()[0]
        m = re.match(r"^## \[([^\]]+)\]", first_line)
        title = m.group(1) if m else first_line.lstrip("#").strip()
        chunks.append(_apply_aliases({"title": title, "text": part, "chunk_type": "function"}))

    # ── 2. Aggregate Functions overview ────────────────────────────────────
    if agg_part:
        chunks.append(_apply_aliases({
            "title": "Aggregate Functions — Overview",
            "text": agg_part,
            "chunk_type": "overview",
        }))

    # ── 3. Window Functions overview ────────────────────────────────────────
    if win_part:
        chunks.append(_apply_aliases({
            "title": "Window Functions — Overview",
            "text": win_part,
            "chunk_type": "overview",
        }))

    return chunks


if __name__ == "__main__":
    ch = get_chunks()
    from collections import Counter
    types = Counter(c["chunk_type"] for c in ch)
    print(f"Total chunks: {len(ch)}")
    for t, n in types.items():
        print(f"  {t}: {n}")
    print(f"\nLargest 5:")
    for c in sorted(ch, key=lambda x: len(x["text"]), reverse=True)[:5]:
        print(f"  {len(c['text']):5d} chars  {c['title']}")
    print(f"\nAliased chunks: {sum(1 for c in ch if 'Русские синонимы' in c['text'])}")
